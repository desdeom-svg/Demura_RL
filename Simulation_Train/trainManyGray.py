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

SAVE_DIR = "Demura_Reward_Weighted"
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
# 2. 真实 Mura 仿真环境 (只修改了 Reward 计算)
# ==========================================
class RealMuraEnv:
    def __init__(self, data_dir="./", roi_size=200, crosstalk_strength=0.0):
        self.roi_size = roi_size
        self.crosstalk_percent = crosstalk_strength
        self.gamma_map = None
        self.scale_map = None
        self.load_data(data_dir)
        self.avg_scale = torch.mean(self.scale_map).item()
        self.avg_gamma = torch.mean(self.gamma_map).item()
        self.ssim_win = create_ssim_window(5, 1)
        self.crosstalk_kernel = get_gaussian_kernel(kernel_size=5, sigma=0.5, channels=1)
        self.pad_size = 5

        self.target_gray_val = 0.0
        self.ideal_target_nit = 0.0
        self.current_gray_map = None
        self.current_luma_map = None

    def load_data(self, data_dir):
        g_path = os.path.join(data_dir, "gammaImg.tiff")
        s_path = os.path.join(data_dir, "scaleImg.tiff")
        if not os.path.exists(g_path) or not os.path.exists(s_path):
            print("Warning: Files not found. Generating dummy data...")
            H, W = 1000, 1000
            g = 2.2 + np.random.normal(0, 0.05, (H, W))
            s = 500 + np.random.normal(0, 10, (H, W))
            k = cv2.getGaussianKernel(101, 20)
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

    def reset(self, target_gray):
        self.target_gray_val = float(target_gray)
        norm_g = self.target_gray_val / 255.0
        self.ideal_target_nit = self.avg_scale * pow(norm_g, self.avg_gamma)
        self.current_gray_map = torch.full((1, 1, self.roi_size, self.roi_size),
                                           self.target_gray_val, device=device)
        self.current_luma_map = self._physics_model(self.current_gray_map)
        return self._get_observation()

    def step(self, action_gray_diff):
        prev_gray = self.current_gray_map.clone()
        self.current_gray_map = self.current_gray_map + action_gray_diff
        self.current_gray_map = torch.clamp(self.current_gray_map, 0, 255)

        actual_delta = self.current_gray_map - prev_gray
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
        # 假设 G16 的 NormStd 约为 0.1，G240 约为 0.02 (相差5倍)
        # 我们设计一个线性增长的权重：
        # G0   -> weight 1.0
        # G250 -> weight 1.0 + (250/50) = 6.0
        gray_weight = 1.0 + (self.target_gray_val / 50.0)

        # 将权重应用到主要的 Loss 上
        r_std = - norm_std * 200.0 * gray_weight
        r_mse = - mae_loss * 500.0 * gray_weight

        # 辅助 Loss 权重可以小一点
        r_grad = - grad_loss * 50.0
        r_action = - torch.mean(actual_delta ** 2) * 0.005

        reward = r_std + r_mse + r_grad + r_action

        next_state = self._get_observation()
        real_std = torch.std(self.current_luma_map).item()

        reward_info = {
            'r_mse': r_mse.item(),
            'r_std': r_std.item(),
            'weight': gray_weight  # 记录权重用于观察
        }

        return next_state, reward.item(), self.current_luma_map.detach(), ssim_val.item(), real_std, norm_std.item(), reward_info


# ==========================================
# 3. Actor (还原为简单结构，无 Scale Net)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x): return F.relu(x + self.conv(x))


class SimpleActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, padding_mode='reflect'), nn.ReLU())
        self.res = nn.Sequential(ResBlock(32), ResBlock(32), ResBlock(32))
        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode='reflect')
        # 初始化权重非常小，避免初始动作过大导致高灰阶直接起飞
        nn.init.uniform_(self.exit.weight, -1e-4, 1e-4)
        nn.init.constant_(self.exit.bias, 0)

    def forward(self, x):
        x = self.entry(x)
        x = self.res(x)
        return self.exit(x)


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
# 4. 训练主循环
# ==========================================
if __name__ == '__main__':
    TRAIN_GRAYS = [16, 32, 48, 64, 96, 128, 160, 192, 224, 240]
    MAX_EPISODES = 800
    STEPS = 8
    BATCH_SIZE = 64

    env = RealMuraEnv(roi_size=200, crosstalk_strength=0.0)
    buffer = ReplayBuffer(15000)

    # 使用简单 Actor
    actor = SimpleActor().to(device)
    target_actor = SimpleActor().to(device);
    target_actor.load_state_dict(actor.state_dict())

    critic = Critic().to(device)
    target_critic = Critic().to(device);
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    best_stds = {g: 9999.0 for g in TRAIN_GRAYS}
    saved_originals = set()

    print(f"Start Training (Simple Actor + Reward Weighted). Levels: {TRAIN_GRAYS}")
    print("Strategy: Balancing Loss Magnitude using Gray-Dependent Weights.")

    for episode in range(MAX_EPISODES):
        current_target_gray = random.choice(TRAIN_GRAYS)
        state = env.reset(target_gray=current_target_gray)

        # 保存原始图
        if current_target_gray not in saved_originals:
            orig_std = torch.std(env.current_luma_map).item()
            save_name = f"Original_G{current_target_gray}_Std{orig_std:.4f}.tiff"
            tiff.imwrite(os.path.join(DIR_ORIGIN, save_name),
                         env.current_luma_map.detach().cpu().numpy()[0, 0].astype(np.float32))
            saved_originals.add(current_target_gray)

        ep_real_std = 0
        ep_norm_std = 0
        current_weight = 0
        noise_scale = max(0.01, 0.2 * (1 - episode / 600))

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

            # Visualization
            if step == STEPS - 1:
                luma_np = luma_map.cpu().numpy()[0, 0]
                gray_np = env.current_gray_map.cpu().numpy()[0, 0]
                comp_diff = gray_np - current_target_gray

                axes[0, 0].clear();
                axes[0, 0].imshow(luma_np, cmap='jet')
                axes[0, 0].set_title(f"Luma G{current_target_gray} | RealStd:{real_std:.3f}")

                axes[0, 1].clear();
                axes[0, 1].imshow(comp_diff, cmap='coolwarm')
                axes[0, 1].set_title(f"Delta Gray (W: {current_weight:.1f})")

                axes[1, 0].clear();
                axes[1, 0].hist(luma_np.ravel(), bins=50, alpha=0.7, color='green')
                axes[1, 0].axvline(env.ideal_target_nit, color='red', linestyle='--')
                axes[1, 0].set_title(f"Target: {env.ideal_target_nit:.2f} | NormStd: {norm_std:.4f}")

                axes[1, 1].clear()
                grays = list(best_stds.keys())
                stds = [best_stds[g] if best_stds[g] < 1000 else 0 for g in grays]
                axes[1, 1].bar(range(len(grays)), stds, tick_label=grays)
                axes[1, 1].set_title("Best Real Std")
                plt.pause(0.001)

        print(f"Ep {episode} [G{current_target_gray}]: RealStd={ep_real_std:.4f} | "
              f"NormStd={ep_norm_std:.5f} | Weight={current_weight:.2f} | "
              f"Reward={r_info['r_std']:.1f}")

        if ep_real_std < best_stds[current_target_gray]:
            best_stds[current_target_gray] = ep_real_std
            if episode > 30:
                print(f"  >>> New Best for G{current_target_gray}! Saving...")
                final_luma = env.current_luma_map.detach().cpu().numpy()[0, 0]
                save_name = f"Luma_G{current_target_gray}_Std{ep_real_std:.4f}.tiff"
                tiff.imwrite(os.path.join(DIR_BEST, save_name), final_luma.astype(np.float32))
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "best_actor.pth"))

    plt.ioff()
    print("Training Finished.")