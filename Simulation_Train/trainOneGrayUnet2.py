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

# ... (Global Settings & Save Dir 保持不变) ...
# ==========================================
# 0. 全局设置
# ==========================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

SAVE_DIR = "Demura_Smooth_HDC"  # 修改保存目录以区分
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)


# ... (辅助函数 get_gaussian_kernel, gaussian_window, create_ssim_window, _ssim 保持不变) ...
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
    mu1_sq = mu1.pow(2);
    mu2_sq = mu2.pow(2);
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2;
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# === 修改点 1：平滑损失函数 (Total Variation - L1) ===
def calc_smoothness_loss(img):
    """
    计算 Total Variation (TV) Loss。
    相比 L2 (Square)，L1 (Abs) 更能有效去除“椒盐噪声”和高频震荡，
    同时能较好地保留边缘（虽然 Demura 不需要锐利边缘）。
    """
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return dy.mean() + dx.mean()


# ... (RealMuraEnv 类保持不变，但在 Step 中调用新的 Loss) ...
class RealMuraEnv:
    def __init__(self, data_dir="./", roi_size=200, target_gray=128, crosstalk_strength=0.0):
        self.roi_size = roi_size
        self.target_gray_val = float(target_gray)
        self.crosstalk_percent = crosstalk_strength
        self.gamma_map = None
        self.scale_map = None
        self.load_data(data_dir)
        avg_scale = torch.mean(self.scale_map).item()
        avg_gamma = torch.mean(self.gamma_map).item()
        norm_g = self.target_gray_val / 255.0
        self.ideal_target_nit = avg_scale * pow(norm_g, avg_gamma)
        self.target_map_tensor = torch.full((1, 1, roi_size, roi_size), self.ideal_target_nit, device=device)
        self.ssim_win = create_ssim_window(5, 1)
        self.crosstalk_kernel = get_gaussian_kernel(kernel_size=5, sigma=0.5, channels=1)
        self.pad_size = 5
        print(f"Target Nit: {self.ideal_target_nit:.4f} (Gray {target_gray})")
        self.current_gray_map = None
        self.current_luma_map = None

    def load_data(self, data_dir):
        g_path = os.path.join(data_dir, "gammaImg.tiff")
        s_path = os.path.join(data_dir, "scaleImg.tiff")
        if not os.path.exists(g_path) or not os.path.exists(s_path):
            print("Generating dummy data...")
            H, W = 1000, 1000
            g = 2.2 + np.random.normal(0, 0.05, (H, W))
            s = 500 + np.random.normal(0, 10, (H, W))
            k = cv2.getGaussianKernel(101, 20);
            k2d = k * k.T
            g += cv2.filter2D(np.random.normal(0, 0.1, (H, W)), -1, k2d)
            tiff.imwrite(g_path, g.astype(np.float32))
            tiff.imwrite(s_path, s.astype(np.float32))
        full_g = tiff.imread(g_path).astype(np.float32)
        full_s = tiff.imread(s_path).astype(np.float32)
        h, w = full_g.shape
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

    def _get_observation(self):
        rel_error = (self.current_luma_map - self.ideal_target_nit) / (self.ideal_target_nit + 1e-6)
        norm_gray = self.current_gray_map / 255.0
        state = torch.cat([rel_error, norm_gray], dim=1)
        return state.detach()

    def reset(self):
        self.current_gray_map = torch.full((1, 1, self.roi_size, self.roi_size),
                                           self.target_gray_val, device=device)
        self.current_luma_map = self._physics_model(self.current_gray_map)
        return self._get_observation()

    def step(self, action_gray_diff):
        self.current_gray_map = self.current_gray_map + action_gray_diff
        self.current_gray_map = torch.clamp(self.current_gray_map, 0, 255)
        self.current_luma_map = self._physics_model(self.current_gray_map)

        # 归一化后计算 Loss
        norm_luma_map = self.current_luma_map / (self.ideal_target_nit + 1e-6)

        diff_raw = self.current_luma_map - self.ideal_target_nit
        mse_raw = torch.mean(diff_raw ** 2)
        std_norm = torch.std(norm_luma_map)

        # SSIM
        target_plane = torch.ones_like(norm_luma_map)
        ssim_val = _ssim(norm_luma_map, target_plane, self.ssim_win, 5, 1)

        # === 使用新的平滑 Loss ===
        # 注意：这里我们计算 Action 的平滑度，或者 Luma Map 的平滑度
        # 计算 Action 的平滑度可以更直接地抑制 Actor 生成噪点
        smooth_loss = calc_smoothness_loss(norm_luma_map)

        r_mse = - mse_raw * 2000.0
        r_std = - std_norm * 200.0
        r_ssim = (ssim_val - 1.0) * 20.0

        # 大幅增加平滑项的权重，强迫 Actor 输出光滑曲面
        r_smooth = - smooth_loss * 100.0

        reward = r_mse + r_std + r_smooth

        next_state = self._get_observation()
        real_std = torch.std(self.current_luma_map).item()

        reward_info = {
            'r_mse': r_mse.item(),
            'r_std': r_std.item(),
            'r_ssim': r_ssim.item(),
            'r_smooth': r_smooth.item(),
            'total': reward.item()
        }
        return next_state, reward.item(), self.current_luma_map.detach(), ssim_val.item(), real_std, reward_info


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
            DilatedResBlock(32, dilation=3),
            DilatedResBlock(32, dilation=1),
            DilatedResBlock(32, dilation=2),
            DilatedResBlock(32, dilation=3)
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
            nn.Conv2d(3, 32, 3, stride=2, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, cap=5000): self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns): self.buf.append((s.cpu(), a.cpu(), r, ns.cpu()))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, ns = zip(*batch)
        return (torch.cat(s).to(device), torch.cat(a).to(device),
                torch.tensor(r).float().unsqueeze(1).to(device), torch.cat(ns).to(device))

    def __len__(self): return len(self.buf)


# ==========================================
# 4. 训练主循环
# ==========================================
if __name__ == '__main__':
    TARGET_GRAY = 16
    CROSSTALK = 0
    MAX_EPISODES = 1000
    STEPS = 16
    BATCH_SIZE = 32

    env = RealMuraEnv(roi_size=200, target_gray=TARGET_GRAY, crosstalk_strength=CROSSTALK)

    print("Saving Original Mura Map...")
    initial_gray = torch.full((1, 1, env.roi_size, env.roi_size), env.target_gray_val, device=device)
    initial_luma = env._physics_model(initial_gray)
    orig_std = torch.std(initial_luma).item()
    tiff.imwrite(os.path.join(SAVE_DIR, f"Original_Mura_Gray{TARGET_GRAY}_Std{orig_std:.3f}.tiff"),
                 initial_luma.cpu().numpy()[0, 0].astype(np.float32))

    buffer = ReplayBuffer(8000)
    actor = Actor().to(device)
    target_actor = Actor().to(device);
    target_actor.load_state_dict(actor.state_dict())
    critic = Critic().to(device)
    target_critic = Critic().to(device);
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=5e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_opt, step_size=300, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_opt, step_size=300, gamma=0.5)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    best_std = orig_std
    history_std, hist_r_smooth = [], []

    print(f"Start Training (Smooth HDC). Target: {env.ideal_target_nit:.4f}")

    for episode in range(MAX_EPISODES):
        state = env.reset()
        ep_std = 0
        last_reward_info = {}
        noise_scale = max(0.01, 0.2 * (1 - episode / 300))

        for step in range(STEPS):
            with torch.no_grad():
                action = actor(state)
                noise = torch.randn_like(action) * noise_scale
                action = action + noise

            next_state, reward, luma_map, ssim_val, std_val, r_info = env.step(action)
            last_reward_info = r_info

            buffer.push(state, action, reward, next_state)
            state = next_state
            ep_std = std_val

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

            if step == STEPS - 1:
                luma_np = luma_map.cpu().numpy()[0, 0]
                gray_np = env.current_gray_map.cpu().numpy()[0, 0]
                comp_diff = gray_np - env.target_gray_val

                axes[0, 0].clear();
                axes[0, 0].imshow(luma_np, cmap='jet')
                axes[0, 0].set_title(f"Luma (Std:{std_val:.3f})")
                axes[0, 1].clear();
                axes[0, 1].imshow(comp_diff, cmap='coolwarm')
                axes[0, 1].set_title("Smoothed Delta Gray")
                axes[1, 0].clear();
                axes[1, 0].hist(luma_np.ravel(), bins=50, alpha=0.7, color='blue')
                axes[1, 0].axvline(env.ideal_target_nit, color='red', linestyle='--')
                axes[1, 0].set_title("Luma Histogram")
                axes[1, 1].clear()
                axes[1, 1].plot(hist_r_smooth, label='r_smooth')
                axes[1, 1].legend(fontsize='small')
                axes[1, 1].set_title("Smoothness Reward")
                plt.pause(0.001)

        actor_scheduler.step()
        critic_scheduler.step()

        history_std.append(ep_std)
        hist_r_smooth.append(last_reward_info['r_smooth'])

        print(
            f"Ep {episode}: Std={ep_std:.4f} | reward={reward:.1f},R_STD={last_reward_info['r_std']:.1f}, R_MSE={last_reward_info['r_mse']:.1f}, R_Smooth={last_reward_info['r_smooth']:.1f}")

        if ep_std < best_std and episode > 10:
            best_std = ep_std
            print(f"  >>> New Best! Std: {best_std:.4f}")
            torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "best_actor.pth"))
            final_gray = env.current_gray_map.detach().cpu().numpy()[0, 0]
            demura_table = final_gray - env.target_gray_val
            tiff.imwrite(os.path.join(SAVE_DIR, f"Demura_Result_Gray{TARGET_GRAY}_Std{best_std:.3f}.tiff"),
                         demura_table.astype(np.float32))
            final_luma = luma_map.detach().cpu().numpy()[0, 0]
            tiff.imwrite(os.path.join(SAVE_DIR, f"Luma_Result_Gray{TARGET_GRAY}_Std{best_std:.3f}.tiff"),
                         final_luma.astype(np.float32))

    plt.ioff()
    print("Training Finished.")
