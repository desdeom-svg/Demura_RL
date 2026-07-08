import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import cv2
import os
import tifffile as tiff
import math

# ==========================================
# 0. 全局设置
# ==========================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

SAVE_DIR = "Demura_Universal_RandomPatch"
DIR_ORIGIN = os.path.join(SAVE_DIR, "Originals")
DIR_BEST = os.path.join(SAVE_DIR, "Best_Correction")

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if not os.path.exists(DIR_ORIGIN): os.makedirs(DIR_ORIGIN)
if not os.path.exists(DIR_BEST): os.makedirs(DIR_BEST)


# ==========================================
# 1. 辅助工具
# ==========================================
def get_gaussian_kernel(kernel_size=5, sigma=2.0, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel.to(device)


def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_ssim_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_gradient_loss(img):
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (dy ** 2).mean() + (dx ** 2).mean()


# ==========================================
# 2. 多屏幕随机切片仿真环境 (Universal Env)
# ==========================================
class MultiScreenMuraEnv:
    def __init__(self, data_root="data", roi_size=200, crosstalk_strength=0.0):
        self.roi_size = roi_size
        self.crosstalk_percent = crosstalk_strength
        self.data_root = data_root

        # 1. 存储所有屏幕的数据 (CPU Tensor)
        # 格式: [{'gamma': tensor, 'scale': tensor, 'h': int, 'w': int}, ...]
        self.screens_db = []
        self.load_all_screens()

        # 2. 当前 Episode 的上下文变量 (GPU Tensor)
        self.current_screen_idx = 0
        self.current_roi = None  # (x, y)
        self.current_gamma_patch = None
        self.current_scale_patch = None

        self.target_gray_val = 0.0
        self.ideal_target_nit = 0.0
        self.current_gray_map = None
        self.current_luma_map = None

        # 3. 辅助
        self.ssim_win = create_ssim_window(5, 1)
        self.crosstalk_kernel = get_gaussian_kernel(kernel_size=5, sigma=0.5, channels=1)
        self.pad_size = 5

    def load_all_screens(self):
        """遍历 screen1 到 screen5 文件夹加载数据"""
        print(f"Loading screens from {self.data_root}...")

        # 定义屏幕文件夹列表
        screen_folders = [f"screen{i}" for i in range(1, 6)]

        for folder in screen_folders:
            folder_path = os.path.join(self.data_root, folder)
            g_path = os.path.join(folder_path, "gammaImg.tiff")
            s_path = os.path.join(folder_path, "scaleImg.tiff")

            # 如果文件不存在，生成带随机特征的虚拟数据（方便代码直接运行）
            if not os.path.exists(g_path) or not os.path.exists(s_path):
                if not os.path.exists(folder_path): os.makedirs(folder_path)
                print(f"  [Warning] {folder} files missing. Generating random dummy screen...")

                # 模拟不同屏幕尺寸 (例如 1000x1000 到 1200x1200)
                H, W = 1000, 1000

                # 模拟不同屏幕的 Mura 特征 (不同的噪声强度和基础Gamma)
                base_gamma = 2.2 + np.random.uniform(-0.1, 0.1)
                noise_std = 0.05 + np.random.uniform(0, 0.05)

                g_np = base_gamma + np.random.normal(0, noise_std, (H, W))
                # 增加一些低频 Mura 斑块
                k_size = 101 + np.random.randint(0, 100)
                k = cv2.getGaussianKernel(k_size, k_size // 4);
                k2d = k * k.T
                mura_blob = cv2.filter2D(np.random.normal(0, 1.0, (H, W)), -1, k2d)
                g_np += mura_blob * 0.5

                s_np = 500 + np.random.normal(0, 10, (H, W))

                # 保存以便下次使用
                tiff.imwrite(g_path, g_np.astype(np.float32))
                tiff.imwrite(s_path, s_np.astype(np.float32))
            else:
                g_np = tiff.imread(g_path).astype(np.float32)
                s_np = tiff.imread(s_path).astype(np.float32)
                print(f"  Loaded {folder}: {g_np.shape}")

            # 转换为 Tensor 存储在 CPU 内存中，避免占用过多显存
            # 只有在 reset 切片时才移动到 GPU
            g_tensor = torch.from_numpy(g_np).unsqueeze(0).unsqueeze(0)
            s_tensor = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0)
            h, w = g_np.shape

            self.screens_db.append({
                'gamma': g_tensor,
                'scale': s_tensor,
                'h': h,
                'w': w,
                'name': folder
            })

        print(f"Total {len(self.screens_db)} screens loaded into memory.")

    def _physics_model(self, gray_tensor):
        norm_g = gray_tensor / 255.0
        norm_g = torch.clamp(norm_g, 1e-6, 1.0)
        # 使用当前切片的参数计算亮度
        l_base = self.current_scale_patch * torch.pow(norm_g, self.current_gamma_patch)

        if self.crosstalk_percent > 0:
            l_padded = F.pad(l_base, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
            l_blur = F.conv2d(l_padded, self.crosstalk_kernel)
            l_final = l_base + (self.crosstalk_percent / 100.0) * l_blur
        else:
            l_final = l_base
        return l_final

    def _get_observation(self):
        rel_error = (self.current_luma_map - self.ideal_target_nit) / (self.ideal_target_nit + 1e-6)
        norm_gray = self.current_gray_map / 255.0
        state = torch.cat([rel_error, norm_gray], dim=1)
        return state.detach()

    def reset(self, target_gray):
        self.target_gray_val = float(target_gray)

        # === 1. 随机选择一块屏幕 ===
        self.current_screen_idx = random.randint(0, len(self.screens_db) - 1)
        screen = self.screens_db[self.current_screen_idx]

        # === 2. 随机选择切片区域 (Random Crop) ===
        max_y = screen['h'] - self.roi_size
        max_x = screen['w'] - self.roi_size

        if max_y < 0 or max_x < 0:
            raise ValueError(f"ROI size {self.roi_size} is larger than screen size {screen['h']}x{screen['w']}")

        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        self.current_roi = (start_x, start_y)

        # 提取切片并移动到 GPU
        self.current_gamma_patch = screen['gamma'][:, :, start_y:start_y + self.roi_size,
                                   start_x:start_x + self.roi_size].to(device)
        self.current_scale_patch = screen['scale'][:, :, start_y:start_y + self.roi_size,
                                   start_x:start_x + self.roi_size].to(device)

        # === 3. 计算该切片的理想目标亮度 ===
        # 这里使用局部平均值作为目标，确保局部平坦
        avg_scale = torch.mean(self.current_scale_patch).item()
        avg_gamma = torch.mean(self.current_gamma_patch).item()

        norm_g = self.target_gray_val / 255.0
        self.ideal_target_nit = avg_scale * pow(norm_g, avg_gamma)

        # === 4. 初始化图像 ===
        self.current_gray_map = torch.full((1, 1, self.roi_size, self.roi_size),
                                           self.target_gray_val, device=device)
        self.current_luma_map = self._physics_model(self.current_gray_map)

        return self._get_observation()

    def step(self, action_gray_diff):
        self.current_gray_map = self.current_gray_map + action_gray_diff
        self.current_gray_map = torch.clamp(self.current_gray_map, 0, 255)

        actual_delta = self.current_gray_map - self.target_gray_val  # 简单的delta计算用于punishment
        self.current_luma_map = self._physics_model(self.current_gray_map)

        # 归一化亮度 (Ratio Map)
        ratio_map = self.current_luma_map / (self.ideal_target_nit + 1e-6)

        # 指标计算
        norm_std = torch.std(ratio_map)
        target_plane = torch.ones_like(ratio_map)
        ssim_val = _ssim(ratio_map, target_plane, self.ssim_win, 5, 1)

        mae_loss = torch.mean(torch.abs(ratio_map - 1.0))
        grad_loss = calc_gradient_loss(ratio_map)

        # === 核心修改：Reward 灰阶加权 ===
        # 目的：平衡高低灰阶 Loss 的量级差异
        gray_weight = 1.0 + (self.target_gray_val / 50.0)

        r_std = - norm_std * 200.0 * gray_weight
        r_mse = - mae_loss * 500.0 * gray_weight
        r_grad = - grad_loss * 50.0
        r_action = - torch.mean(action_gray_diff ** 2) * 0.005  # 动作幅度惩罚

        reward = r_std + r_mse + r_grad + r_action

        next_state = self._get_observation()
        real_std = torch.std(self.current_luma_map).item()

        reward_info = {
            'r_mse': r_mse.item(),
            'r_std': r_std.item(),
            'weight': gray_weight,
            'screen_id': self.current_screen_idx
        }

        return next_state, reward.item(), self.current_luma_map.detach(), ssim_val.item(), real_std, norm_std.item(), reward_info


# ==========================================
# 3. 改进的网络结构 (Pad-Process-Crop 策略)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 内部还是用 reflect padding，但在大图上做，边缘影响被推远
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x): return F.relu(x + self.conv(x))


class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x): return F.relu(x + self.conv(x))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_size = 16  # 定义过采样边缘大小

        self.entry = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )

        # HDC: 1,2,5 循环
        self.global_feat = nn.Sequential(
            DilatedResBlock(32, dilation=1),
            DilatedResBlock(32, dilation=2),
            DilatedResBlock(32, dilation=5),
            DilatedResBlock(32, dilation=1),
            DilatedResBlock(32, dilation=2),
            DilatedResBlock(32, dilation=5)
        )

        # Local Refinement
        self.local_refine = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlock(32)
        )

        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode='reflect')
        nn.init.uniform_(self.exit.weight, -1e-5, 1e-5)
        nn.init.constant_(self.exit.bias, 0)

    def forward(self, x):
        # === 核心策略：Pad -> Process -> Crop ===

        # 1. Pad Input: 四周各填充 16 像素 (使用 reflect 模式保持纹理连续)
        # 输入 200x200 -> 232x232
        x_pad = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')

        # 2. Network Process (在大图上跑)
        feat = self.entry(x_pad)
        feat = self.global_feat(feat)
        feat = self.local_refine(feat)
        out_pad = self.exit(feat)

        # 3. Crop Output: 切掉四周的 16 像素，只留纯净的中心
        # 输出 232x232 -> 200x200
        out = out_pad[..., self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        return out


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, cap=15000): self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns): self.buf.append((s.cpu(), a.cpu(), r, ns.cpu()))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, ns = zip(*batch)
        return (torch.cat(s).to(device), torch.cat(a).to(device),
                torch.tensor(r).float().unsqueeze(1).to(device), torch.cat(ns).to(device))

    def __len__(self): return len(self.buf)


# ==========================================
# 4. 训练主循环 (通用模型训练)
# ==========================================
if __name__ == '__main__':
    TRAIN_GRAYS = [16, 32, 48, 64, 96, 128, 160, 192, 224, 240]
    # 增加训练轮次，因为任务变难了（需要适应不同屏幕和位置）
    MAX_EPISODES = 2000
    STEPS = 8
    BATCH_SIZE = 32

    # 初始化多屏幕环境 (假设数据在 data/ 目录下)
    env = MultiScreenMuraEnv(data_root="data", roi_size=200, crosstalk_strength=0.0)

    buffer = ReplayBuffer(20000)

    # 使用简单 Actor
    actor = Actor().to(device)
    target_actor = Actor().to(device);
    target_actor.load_state_dict(actor.state_dict())

    critic = Critic().to(device)
    target_critic = Critic().to(device);
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 记录每个灰阶的最佳成绩 (归一化Std)
    # 注意：不同屏幕、不同位置的RealStd不可直接比较，必须用 NormStd
    best_norm_stds = {g: 999.0 for g in TRAIN_GRAYS}
    saved_originals = set()

    print(f"Start Universal Training (Random Patch & Random Screen). Levels: {TRAIN_GRAYS}")

    for episode in range(MAX_EPISODES):
        # 1. 随机灰阶
        current_target_gray = random.choice(TRAIN_GRAYS)

        # 2. Reset (内部会随机选屏、随机切片)
        state = env.reset(target_gray=current_target_gray)

        # 3. 偶尔保存一下当前切片的原始图 (用于观察数据多样性)
        if episode % 50 == 0:
            orig_std = torch.std(env.current_luma_map).item()
            scr_id = env.current_screen_idx
            save_name = f"Ep{episode}_Scn{scr_id}_G{current_target_gray}.tiff"
            # tiff.imwrite(os.path.join(DIR_ORIGIN, save_name), env.current_luma_map.detach().cpu().numpy()[0,
            # 0].astype(np.float32))

        ep_real_std = 0
        ep_norm_std = 0
        current_weight = 0
        noise_scale = max(0.01, 0.2 * (1 - episode / 1000))

        for step in range(STEPS):
            with torch.no_grad():
                action = actor(state)
                noise = torch.randn_like(action) * noise_scale
                action = action + noise

            # Step
            next_state, reward, luma_map, ssim_val, real_std, norm_std, r_info = env.step(action)
            current_weight = r_info['weight']

            buffer.push(state, action, reward, next_state)
            state = next_state

            ep_real_std = real_std
            ep_norm_std = norm_std

            # DDPG Update
            if len(buffer) > BATCH_SIZE:
                sb, ab, rb, nsb = buffer.sample(BATCH_SIZE)
                with torch.no_grad():
                    tgt_a = target_actor(nsb)
                    target_q = rb + 0.9 * target_critic(nsb, tgt_a)
                curr_q = critic(sb, ab)
                loss_c = F.mse_loss(curr_q, target_q)
                critic_opt.zero_grad();
                loss_c.backward();
                critic_opt.step()
                pred_a = actor(sb)
                loss_a = -critic(sb, pred_a).mean()
                actor_opt.zero_grad();
                loss_a.backward();
                actor_opt.step()
                for p, tp in zip(actor.parameters(), target_actor.parameters()):
                    tp.data.copy_(0.01 * p.data + 0.99 * tp.data)
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.copy_(0.01 * p.data + 0.99 * tp.data)

            # Visualization (降低绘图频率，每10轮画一次)
            if step == STEPS - 1:  # and episode % 10 == 0
                luma_np = luma_map.cpu().numpy()[0, 0]
                gray_np = env.current_gray_map.cpu().numpy()[0, 0]
                comp_diff = gray_np - current_target_gray

                axes[0, 0].clear();
                axes[0, 0].imshow(luma_np, cmap='jet')
                axes[0, 0].set_title(
                    f"Scn{env.current_screen_idx} | G{current_target_gray} | NormStd:{ep_norm_std:.4f}")

                axes[0, 1].clear();
                axes[0, 1].imshow(comp_diff, cmap='coolwarm')
                axes[0, 1].set_title(f"Comp (W: {current_weight:.1f})")

                axes[1, 0].clear();
                axes[1, 0].hist(luma_np.ravel(), bins=50, alpha=0.7, color='green')
                axes[1, 0].axvline(env.ideal_target_nit, color='red', linestyle='--')
                axes[1, 0].set_title(f"Target: {env.ideal_target_nit:.2f} nits")

                axes[1, 1].clear()
                grays = list(best_norm_stds.keys())
                # 只显示有效的 best std (初始999不显示)
                stds = [best_norm_stds[g] if best_norm_stds[g] < 10.0 else 0 for g in grays]
                axes[1, 1].bar(range(len(grays)), stds, tick_label=grays)
                axes[1, 1].set_title("Best Norm Std per Gray")
                plt.pause(0.001)

        print(f"Ep {episode} [Scn{env.current_screen_idx} | G{current_target_gray}]: NormStd={ep_norm_std:.4f} | "
              f"RealStd={ep_real_std:.4f} | R_STD={r_info['r_std']:.1f}")

        # 保存逻辑：基于 归一化Std 更新最佳记录
        if ep_norm_std < best_norm_stds[current_target_gray]:
            best_norm_stds[current_target_gray] = ep_norm_std

            # 保存通用模型
            if episode > 50:
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "best_universal_actor.pth"))

                # 保存效果图
                final_luma = env.current_luma_map.detach().cpu().numpy()[0, 0]
                save_name = f"Luma_Scn{env.current_screen_idx}_G{current_target_gray}.tiff"
                tiff.imwrite(os.path.join(DIR_BEST, save_name), final_luma.astype(np.float32))

    plt.ioff()
    print("Training Finished.")
