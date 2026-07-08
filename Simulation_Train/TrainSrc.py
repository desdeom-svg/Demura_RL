import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import cv2
import os

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# ==========================================
# 1. 物理仿真环境 (噪声可控)
# ==========================================
class OLEDScreenEnv:
    def __init__(self, height=200, width=200, mura_intensity=1.0, noise_level=0.0):
        self.h = height
        self.w = width
        self.max_nit = 500.0
        self.gamma = 2.2
        self.mura_intensity = mura_intensity
        self.noise_level = noise_level  # 环境（相机）噪声强度

        self.base_gray_val = 128.0
        self.target_nit = self.max_nit * pow(self.base_gray_val / 255.0, self.gamma)
        print(f"Target: {self.target_nit:.2f} nit | Mura: {self.mura_intensity} | Noise: {self.noise_level}")

        # 模拟 Mura
        noise = (torch.rand(1, 1, self.h, self.w).to(device) - 0.5) * 2
        self.mura_map = 1.0 + self.mura_intensity * noise

        # 串扰核
        kernel_size = 5
        sigma = 1.0
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        self.crosstalk_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)
        self.pad_size = kernel_size // 2

        self.current_gray = torch.full((1, 1, self.h, self.w), self.base_gray_val).to(device)
        # 初始干净Mura (不带随机噪声) 用于保存TIFF
        self.initial_brightness_clean = self._physics_model(self.current_gray, enable_noise=False).detach()

    def _physics_model(self, gray_map, enable_noise=True):
        norm_gray = gray_map / 255.0
        base_brightness = self.max_nit * torch.pow(norm_gray, self.gamma)
        mura_brightness = base_brightness * self.mura_map

        mura_padded = F.pad(mura_brightness, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                            mode='replicate')
        final_brightness = F.conv2d(mura_padded, self.crosstalk_kernel)

        if enable_noise and self.noise_level > 0:
            noise = torch.randn_like(final_brightness) * self.noise_level
            return final_brightness + noise
        else:
            return final_brightness

    def calculate_9_point_uniformity(self, brightness_map):
        h_step, w_step = self.h // 3, self.w // 3
        sample_ratio = 0.33
        h_margin, w_margin = int(h_step * (1 - sample_ratio) / 2), int(w_step * (1 - sample_ratio) / 2)
        means = []
        for r in range(3):
            for c in range(3):
                roi = brightness_map[:, :, r * h_step + h_margin:(r + 1) * h_step - h_margin,
                      c * w_step + w_margin:(c + 1) * w_step - w_margin]
                means.append(torch.mean(roi))
        means_tensor = torch.stack(means)
        min_val, max_val = torch.min(means_tensor), torch.max(means_tensor)
        if max_val == 0: return 0.0
        return (min_val / max_val).item()

    def calculate_tv_loss(self, img):
        h_diff = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        w_diff = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return h_diff + w_diff

    def reset(self):
        self.current_gray = torch.full((1, 1, self.h, self.w), self.base_gray_val).to(device)
        current_b = self._physics_model(self.current_gray)
        state = (current_b - self.target_nit) / 10.0
        return state.detach()

    def step(self, action_compensation, tv_weight_factor=1.0):
        action_compensation = action_compensation.detach()
        self.current_gray = torch.clamp(self.current_gray + action_compensation, 0, 255).detach()
        current_brightness = self._physics_model(self.current_gray)

        # 指标
        diff = current_brightness - self.target_nit
        mse = torch.mean(diff ** 2)
        std = torch.std(current_brightness)
        uni_9pt = self.calculate_9_point_uniformity(current_brightness)
        tv_val = self.calculate_tv_loss(current_brightness)

        # 奖励 (TV Loss 权重动态调整)
        reward_mse = - mse * 0.05
        reward_std = - std * 5.0
        reward_uni = (uni_9pt - 0.90) * 500.0
        reward_tv = - tv_val * 2.0 * tv_weight_factor  # 动态权重

        reward = reward_mse + reward_std + reward_uni + reward_tv

        next_state = (current_brightness - self.target_nit) / 10.0
        return next_state.detach(), reward.item(), current_brightness.detach(), uni_9pt, mse.item()


# ==========================================
# 2. Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state.cpu(), action.cpu(), reward, next_state.cpu()))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (torch.cat(state).to(device), torch.cat(action).to(device),
                torch.tensor(reward).float().unsqueeze(1).to(device), torch.cat(next_state).to(device))

    def __len__(self): return len(self.buffer)


# ==========================================
# 3. Networks (ResNet + Coord)
# ==========================================
def get_coord_grid(batch_size, h, w):
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return torch.cat([grid_x, grid_y], dim=1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate')
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.entry = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, padding_mode='replicate'), nn.ReLU())
        self.res_blocks = nn.Sequential(ResBlock(32), ResBlock(32), ResBlock(32))
        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode='replicate')
        nn.init.uniform_(self.exit.weight, -0.0001, 0.0001)
        nn.init.constant_(self.exit.bias, 0)

    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W)
        x_in = torch.cat([x, coords], dim=1)
        x = self.entry(x_in)
        x = self.res_blocks(x)
        return self.exit(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='replicate'), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, state, action):
        B, _, H, W = state.shape
        coords = get_coord_grid(B, H, W)
        state_in = torch.cat([state, coords], dim=1)
        x = torch.cat([state_in, action], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==========================================
# 4. Main Training Loop
# ==========================================
if __name__ == '__main__':
    # === 参数配置 ===
    MURA_INTENSITY = 0.5
    NOISE_LEVEL = 0.0  # 0.0 为纯净模式，0.01 为模拟相机噪声
    MAX_EPISODES = 2000
    STEPS_PER_EPISODE = 20
    BATCH_SIZE = 128  # 显存允许则用128，否则64
    SAVE_DIR = "Demura_Final_Clean"

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    env = OLEDScreenEnv(mura_intensity=MURA_INTENSITY, noise_level=NOISE_LEVEL)
    buffer = ReplayBuffer(capacity=8000)

    actor = Actor().to(device)
    target_actor = Actor().to(device)
    critic = Critic().to(device)
    target_critic = Critic().to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=5e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    history = {'reward': [], 'mse': [], 'std': [], 'uni': []}
    best_std = 100.0
    mura_saved = False

    # Visualization
    plt.ion()

    fig_env, axes = plt.subplots(2, 2, figsize=(10, 8))
    im_b = axes[0, 0].imshow(np.zeros((200, 200)), cmap='jet');
    axes[0, 0].set_title("Current Brightness")
    plt.colorbar(im_b, ax=axes[0, 0])
    im_err = axes[0, 1].imshow(np.zeros((200, 200)), cmap='coolwarm');
    axes[0, 1].set_title("Error Map")
    plt.colorbar(im_err, ax=axes[0, 1])
    im_act = axes[1, 0].imshow(np.zeros((200, 200)), cmap='gray');
    axes[1, 0].set_title("Compensation")
    plt.colorbar(im_act, ax=axes[1, 0])
    im_final_g = axes[1, 1].imshow(np.zeros((200, 200)), cmap='gray', vmin=0, vmax=255);
    axes[1, 1].set_title("Final Gray")
    plt.colorbar(im_final_g, ax=axes[1, 1])
    plt.tight_layout()

    fig_curve, axes_curve = plt.subplots(2, 2, figsize=(12, 8))

    print(f"=== Training Start (Noise: {NOISE_LEVEL} | Dynamic TV Loss) ===")

    try:
        for episode in range(MAX_EPISODES):
            state = env.reset()
            ep_reward = 0
            final_uni = 0
            final_mse = 0

            # 策略调整
            noise_scale = max(0.1, 2.0 * (1 - episode / 100))
            # TV Loss 权重衰减：前50轮保持1.0，之后逐渐减小到0.1，允许后期修细节
            tv_weight = 1.0 if episode < 50 else max(0.1, 1.0 - (episode - 50) / 100)

            for step in range(STEPS_PER_EPISODE):
                with torch.no_grad():
                    action = actor(state)
                    noise = torch.randn_like(action) * noise_scale
                    action = action + noise

                # 传入 TV 权重
                next_state, reward, real_b, uni_9pt, mse_val = env.step(action, tv_weight_factor=tv_weight)
                buffer.push(state, action, reward, next_state)
                state = next_state

                ep_reward += reward
                final_uni = uni_9pt
                final_mse = mse_val

                if len(buffer) > BATCH_SIZE:
                    s_b, a_b, r_b, ns_b = buffer.sample(BATCH_SIZE)

                    with torch.no_grad():
                        tgt_act = target_actor(ns_b)
                        tgt_q = target_critic(ns_b, tgt_act)
                        y = r_b + 0.8 * tgt_q

                    curr_q = critic(s_b, a_b)
                    c_loss = F.mse_loss(curr_q, y)
                    critic_opt.zero_grad();
                    c_loss.backward();
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0);
                    critic_opt.step()

                    pred_a = actor(s_b)
                    a_loss = -critic(s_b, pred_a).mean()
                    actor_opt.zero_grad();
                    a_loss.backward();
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0);
                    actor_opt.step()

                    for p, tp in zip(actor.parameters(), target_actor.parameters()):
                        tp.data.copy_(0.05 * p.data + 0.95 * tp.data)
                    for p, tp in zip(critic.parameters(), target_critic.parameters()):
                        tp.data.copy_(0.05 * p.data + 0.95 * tp.data)

                if step == STEPS_PER_EPISODE - 1:
                    b_np = real_b.cpu().numpy()[0, 0]
                    diff_np = b_np - env.target_nit
                    act_np = action.detach().cpu().numpy()[0, 0]
                    gray_np = env.current_gray.cpu().numpy()[0, 0]

                    im_b.set_data(b_np);
                    im_b.set_clim(env.target_nit - 20, env.target_nit + 20)
                    im_err.set_data(diff_np);
                    limit = max(1, np.max(np.abs(diff_np)));
                    im_err.set_clim(-limit, limit)
                    im_act.set_data(act_np);
                    im_act.set_clim(np.min(act_np), np.max(act_np))
                    im_final_g.set_data(gray_np)
                    fig_env.canvas.draw();
                    fig_env.canvas.flush_events();
                    plt.pause(0.001)

            current_std = torch.std(real_b).item()
            history['reward'].append(ep_reward)
            history['mse'].append(final_mse)
            history['std'].append(current_std)
            history['uni'].append(final_uni)

            print(
                f"Ep {episode}: R={ep_reward:.1f}, MSE={final_mse:.2f}, Std={current_std:.3f}, Uni={final_uni:.4f}, TV_W={tv_weight:.2f}")

            # 实时保存曲线
            axes_curve[0, 0].clear();
            axes_curve[0, 0].plot(history['reward']);
            axes_curve[0, 0].set_title('Reward')
            axes_curve[0, 1].clear();
            axes_curve[0, 1].plot(history['mse']);
            axes_curve[0, 1].set_title('MSE')
            axes_curve[1, 0].clear();
            axes_curve[1, 0].plot(history['std']);
            axes_curve[1, 0].set_title('Std (Lower Better)')
            axes_curve[1, 1].clear();
            axes_curve[1, 1].plot(history['uni']);
            axes_curve[1, 1].set_title('Uniformity')
            fig_curve.savefig(os.path.join(SAVE_DIR, "Training_Curve.png"))

            # 熔断
            if current_std > 100.0 and episode > 10:
                print("!!! Divergence Detected !!!")
                break

            # 保存更优结果
            if episode > 10 and current_std < best_std:
                best_std = current_std
                print(f"   >>> Improved! Saving snapshot (Std: {best_std:.3f})")

                # 1. 保存模型
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "best_actor_model.pth"))

                # 2. 【关键修改】保存 Mura 本体数据 (.npy)
                if not mura_saved:
                    # 将 Tensor 转为 numpy 数组保存
                    np.save(os.path.join(SAVE_DIR, "Env_Mura_Map.npy"), env.mura_map.cpu().numpy())

                    # 顺便也存一张图作为预览（仅供人看，不用于机器还原）
                    orig_data = env.initial_brightness_clean.cpu().numpy()[0, 0].astype(np.float32)
                    cv2.imwrite(os.path.join(SAVE_DIR, "Original_Mura_Preview.tiff"), orig_data)
                    mura_saved = True

                # 3. 保存 累计补偿图 (Total Compensation)
                # 确保是：Current_Gray - 128
                final_gray_tensor = env.current_gray.detach()
                total_comp_tensor = final_gray_tensor - env.base_gray_val
                comp_data = total_comp_tensor.cpu().numpy()[0, 0].astype(np.float32)
                cv2.imwrite(os.path.join(SAVE_DIR, f"Comp_Std_{best_std:.3f}.tiff"), comp_data)

            if current_std < 0.2 and final_uni > 0.9995:  # 纯净模式下标准可以更高
                print(">>> Perfect Target Achieved! <<<")
                break

    except KeyboardInterrupt:
        print("Interrupted.")

    plt.ioff()
    plt.close('all')
    print("Done.")