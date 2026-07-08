import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import tifffile as tiff
import math
import random
import matplotlib.pyplot as plt

# ==========================================
# 0. 配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

TEACHER_MODEL_PATH = "Demura_Final_Opt_Gaussian/best_actor.pth"
SAVE_DIR = "Distillation_Output"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 蒸馏参数
EPOCHS = 2000
BATCH_SIZE = 16  # 显存够可以加大
TEACHER_STEPS = 10  # 老师迭代次数
TARGET_GRAY = 16  # 训练的目标灰阶
ROI_SIZE = 200  # 训练时的尺寸


# ==========================================
# 1. 复用之前的类定义 (Env, Actor, Utils)
# ==========================================
# (为了代码独立完整性，这里重新粘贴一遍必要的类，
#  实际工程中可以直接 import)

def get_gaussian_kernel(kernel_size=5, sigma=2.0, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel.to(device)


class RealMuraEnv:
    def __init__(self, data_dir="./", roi_size=200, target_gray=128, crosstalk_strength=0.0):
        self.roi_size = roi_size
        self.target_gray_val = float(target_gray)
        self.crosstalk_percent = crosstalk_strength
        self.load_data(data_dir)

        avg_scale = torch.mean(self.scale_map).item()
        avg_gamma = torch.mean(self.gamma_map).item()
        norm_g = self.target_gray_val / 255.0
        self.ideal_target_nit = avg_scale * pow(norm_g, avg_gamma)

        self.crosstalk_kernel = get_gaussian_kernel(5, 0.5, 1)
        self.pad_size = 2

    def load_data(self, data_dir):
        g_path = os.path.join(data_dir, "gammaImg.tiff")
        s_path = os.path.join(data_dir, "scaleImg.tiff")

        if not os.path.exists(g_path):  # 生成假数据保底
            H, W = 1000, 1000
            g = 2.2 + np.random.normal(0, 0.05, (H, W));
            s = 500 + np.random.normal(0, 10, (H, W))
            tiff.imwrite(g_path, g.astype(np.float32));
            tiff.imwrite(s_path, s.astype(np.float32))

        full_g = tiff.imread(g_path).astype(np.float32)
        full_s = tiff.imread(s_path).astype(np.float32)

        # 随机裁剪 (Data Augmentation) - 蒸馏时非常重要，让Student见过各种Mura形态
        h, w = full_g.shape
        # 这里为了简化，还是取中心，实际可以写个 random_crop
        cx, cy = w // 2, h // 2
        r = self.roi_size // 2
        g_roi = full_g[cy - r:cy + r, cx - r:cx + r]
        s_roi = full_s[cy - r:cy + r, cx - r:cx + r]

        self.gamma_map = torch.from_numpy(g_roi).unsqueeze(0).unsqueeze(0).to(device)
        self.scale_map = torch.from_numpy(s_roi).unsqueeze(0).unsqueeze(0).to(device)

    def _physics_model(self, gray_tensor):
        norm_g = gray_tensor / 255.0
        norm_g = torch.clamp(norm_g, 1e-6, 1.0)
        l_base = self.scale_map * torch.pow(norm_g, self.gamma_map)
        if self.crosstalk_percent > 0:
            l_padded = F.pad(l_base, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
            l_blur = F.conv2d(l_padded, self.crosstalk_kernel)
            l_final = l_base + (self.crosstalk_percent / 100.0) * l_blur
        else:
            l_final = l_base
        return l_final

    def get_state(self, current_luma, current_gray):
        rel_error = (current_luma - self.ideal_target_nit) / (self.ideal_target_nit + 1e-6)
        norm_gray = current_gray / 255.0
        state = torch.cat([rel_error, norm_gray], dim=1)
        return state


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x): return F.relu(x + self.conv(x))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, padding_mode='reflect'), nn.ReLU())
        self.res = nn.Sequential(ResBlock(32), ResBlock(32), ResBlock(32))
        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.entry(x)
        x = self.res(x)
        return self.exit(x)

    # ==========================================


# 2. 蒸馏训练逻辑
# ==========================================
def train_distillation():
    # 1. 环境准备
    # 注意：这里我们使用 1.0 的串扰，因为这是 Teacher 训练时的环境
    env = RealMuraEnv(roi_size=ROI_SIZE, target_gray=TARGET_GRAY, crosstalk_strength=0.0)

    # 2. 加载 Teacher
    teacher = Actor().to(device)
    if os.path.exists(TEACHER_MODEL_PATH):
        teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH))
        print("Teacher model loaded.")
    else:
        print("Error: Teacher model not found!")
        return
    teacher.eval()  # 冻结 Teacher
    for param in teacher.parameters():
        param.requires_grad = False

    # 3. 初始化 Student
    student = Actor().to(device)
    student.train()

    # 4. 优化器
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    loss_history = []

    print(f"Start Distillation. Epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # --- A. 生成训练数据 (Data Generation via Simulation) ---
        # 这一步不需要梯度，我们只关心 Teacher 的最终输出
        with torch.no_grad():
            # 初始灰阶 (Batch Size = 1，实际可以扩展为 Batch)
            # 为了增加鲁棒性，我们可以给初始灰阶加一点微小的随机扰动
            init_gray = torch.full((1, 1, ROI_SIZE, ROI_SIZE), float(TARGET_GRAY), device=device)
            # init_gray += torch.randn_like(init_gray) * 0.1

            # 计算 Teacher 的累积输出 (Ground Truth)
            current_gray = init_gray.clone()
            teacher_accumulated_action = torch.zeros_like(init_gray)

            # 记录初始状态 S0 (Student 的输入)
            luma_0 = env._physics_model(current_gray)
            state_0 = env.get_state(luma_0, current_gray)

            # Teacher 迭代 10 步
            for _ in range(TEACHER_STEPS):
                luma = env._physics_model(current_gray)
                state = env.get_state(luma, current_gray)

                action = teacher(state)

                current_gray = torch.clamp(current_gray + action, 0, 255)
                teacher_accumulated_action += action

        # --- B. Student 推理 (One-shot) ---
        # Student 输入的是 S0，直接预测 Teacher 的 Sum(Action)
        student_pred_action = student(state_0)

        # --- C. 计算 Loss 并更新 ---
        # 目标：Student 的单步输出 == Teacher 的 10 步累积输出
        loss = F.mse_loss(student_pred_action, teacher_accumulated_action)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")

            # 保存中间结果
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, "student_oneshot_latest.pth"))

    # 保存最终 Student 模型
    torch.save(student.state_dict(), os.path.join(SAVE_DIR, "student_oneshot_final.pth"))
    print("Distillation Finished. Model saved.")

    # 绘制 Loss 曲线
    plt.plot(loss_history)
    plt.title("Distillation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.show()


# ==========================================
# 3. 验证脚本 (Validation)
# ==========================================
def validate_oneshot():
    print("\nValidating One-shot Student Model...")

    env = RealMuraEnv(roi_size=ROI_SIZE, target_gray=TARGET_GRAY, crosstalk_strength=1.0)
    student = Actor().to(device)
    student.load_state_dict(torch.load(os.path.join(SAVE_DIR, "student_oneshot_final.pth")))
    student.eval()

    # 1. 初始状态
    current_gray = torch.full((1, 1, ROI_SIZE, ROI_SIZE), float(TARGET_GRAY), device=device)
    luma_orig = env._physics_model(current_gray)
    state_0 = env.get_state(luma_orig, current_gray)
    std_orig = torch.std(luma_orig).item()

    # 2. Student 单步推理
    with torch.no_grad():
        action = student(state_0)

    # 3. 应用补偿
    final_gray = torch.clamp(current_gray + action, 0, 255)
    luma_final = env._physics_model(final_gray)
    std_final = torch.std(luma_final).item()

    print(f"Original Std: {std_orig:.4f}")
    print(f"One-shot Std: {std_final:.4f}")

    # 保存对比图
    tiff.imwrite(os.path.join(SAVE_DIR, "Val_Orig_Luma.tiff"), luma_orig.cpu().numpy()[0, 0].astype(np.float32))
    tiff.imwrite(os.path.join(SAVE_DIR, "Val_Final_Luma.tiff"), luma_final.cpu().numpy()[0, 0].astype(np.float32))
    tiff.imwrite(os.path.join(SAVE_DIR, "Val_Action.tiff"), action.cpu().numpy()[0, 0].astype(np.float32))


if __name__ == "__main__":
    train_distillation()
    validate_oneshot()