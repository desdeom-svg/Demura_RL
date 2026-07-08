import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import tifffile as tiff

# UI & Plotting
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ==========================================
# 0. 配置与设备
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference running on: {device}")

# === 对应训练代码中的设置 ===
SAVE_DIR = "Demura_Reward_Weighted"  # 训练结果文件夹
MODEL_NAME = "best_actor.pth"  # 请确保您在训练时保存了这个文件
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
DATA_DIR = "./"


# ==========================================
# 1. 辅助函数
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


# ==========================================
# 2. 网络结构 (SimpleActor - 必须与训练完全一致)
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
        # Input: 2 channels (Rel_Error, Norm_Gray)
        self.entry = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, padding_mode='reflect'), nn.ReLU())
        self.res = nn.Sequential(ResBlock(32), ResBlock(32), ResBlock(32))
        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.entry(x)
        x = self.res(x)
        return self.exit(x)


# ==========================================
# 3. 推理环境
# ==========================================
class InferenceMuraEnv:
    def __init__(self, data_dir, roi_size=200):
        self.roi_size = roi_size
        self.gamma_map = None
        self.scale_map = None

        self.load_data(data_dir)

        # 串扰模拟核心
        self.crosstalk_kernel = get_gaussian_kernel(kernel_size=5, sigma=0.5, channels=1)
        self.pad_size = 2  # kernel//2

        self.target_gray_val = 128.0
        self.crosstalk_percent = 0.0
        self.ideal_target_nit = 0.0

    def load_data(self, data_dir):
        g_path = os.path.join(data_dir, "gammaImg.tiff")
        s_path = os.path.join(data_dir, "scaleImg.tiff")

        if not os.path.exists(g_path) or not os.path.exists(s_path):
            # 如果没有文件，生成假数据用于测试
            print("TIFF files not found, generating dummy data...")
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

        self.avg_scale = torch.mean(self.scale_map).item()
        self.avg_gamma = torch.mean(self.gamma_map).item()

    def update_params(self, gray_val, crosstalk_val):
        self.target_gray_val = float(gray_val)
        self.crosstalk_percent = float(crosstalk_val)

        # 计算理想目标亮度 (与训练代码一致)
        norm_g = self.target_gray_val / 255.0
        self.ideal_target_nit = self.avg_scale * pow(norm_g, self.avg_gamma)

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
        # 归一化误差 + 归一化灰阶
        rel_error = (current_luma - self.ideal_target_nit) / (self.ideal_target_nit + 1e-6)
        norm_gray = current_gray / 255.0
        state = torch.cat([rel_error, norm_gray], dim=1)
        return state


# ==========================================
# 4. UI 界面
# ==========================================
class DemuraInferenceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多灰阶 Demura 推理演示 (SimpleActor)")
        self.resize(1300, 900)

        try:
            self.env = InferenceMuraEnv(DATA_DIR, roi_size=200)
            self.model = SimpleActor().to(device)

            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print(f"Model loaded successfully: {MODEL_PATH}")
            else:
                raise FileNotFoundError(
                    f"Model file not found at: {MODEL_PATH}\n请确认已在训练代码中保存 best_actor.pth")

            self.model.eval()
        except Exception as e:
            print(f"Error: {e}")
            # 这里简单处理，实际应用可以弹窗提示

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        ctrl_group = QGroupBox("控制面板")
        ctrl_layout = QHBoxLayout(ctrl_group)

        # 1. 灰阶控制
        vbox_g = QVBoxLayout()
        vbox_g.addWidget(QLabel("目标灰阶 (Gray Level):"))
        self.slider_gray = QSlider(Qt.Horizontal)
        self.slider_gray.setRange(1, 250)  # 避免255纯白溢出
        self.slider_gray.setValue(32)
        self.lbl_gray = QLabel("32")
        self.slider_gray.valueChanged.connect(self.run_inference)
        self.slider_gray.valueChanged.connect(lambda v: self.lbl_gray.setText(str(v)))
        vbox_g.addWidget(self.slider_gray)
        vbox_g.addWidget(self.lbl_gray)
        ctrl_layout.addLayout(vbox_g)

        # 2. 迭代次数
        vbox_i = QVBoxLayout()
        vbox_i.addWidget(QLabel("迭代步数 (Steps):"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 20)
        self.spin_iter.setValue(8)  # 训练时用了8步
        self.spin_iter.valueChanged.connect(self.run_inference)
        vbox_i.addWidget(self.spin_iter)
        ctrl_layout.addLayout(vbox_i)

        # 3. 串扰干扰 (可选)
        vbox_c = QVBoxLayout()
        vbox_c.addWidget(QLabel("模拟干扰 (Crosstalk %):"))
        self.slider_cross = QSlider(Qt.Horizontal)
        self.slider_cross.setRange(0, 50)
        self.slider_cross.setValue(0)
        self.lbl_cross = QLabel("0.0")
        self.slider_cross.valueChanged.connect(self.run_inference)
        self.slider_cross.valueChanged.connect(lambda v: self.lbl_cross.setText(str(v)))
        vbox_c.addWidget(self.slider_cross)
        vbox_c.addWidget(self.lbl_cross)
        ctrl_layout.addLayout(vbox_c)

        layout.addWidget(ctrl_group)
        self.run_inference()

    def run_inference(self):
        g_val = self.slider_gray.value()
        c_val = self.slider_cross.value()
        iters = self.spin_iter.value()

        self.env.update_params(g_val, c_val)

        # 1. 初始状态
        # 初始灰阶图 = 目标灰阶 (全平)
        current_gray = torch.full((1, 1, self.env.roi_size, self.env.roi_size),
                                  float(g_val), device=device)

        # 记录初始物理亮度 (Raw Mura)
        luma_orig_tensor = self.env._physics_model(current_gray)

        # 2. 迭代推理
        for i in range(iters):
            # 物理渲染
            current_luma = self.env._physics_model(current_gray)

            # 获取状态
            state = self.env.get_state(current_luma, current_gray)

            # 模型预测 (SimpleActor 输出)
            with torch.no_grad():
                action = self.model(state)

            # 更新灰阶
            current_gray = torch.clamp(current_gray + action, 0, 255)

        # 3. 最终结果
        luma_final_tensor = self.env._physics_model(current_gray)

        # 计算 Demura Table (灰阶差值)
        demura_table = (current_gray - float(g_val)).cpu().numpy()[0, 0]

        # 4. 准备绘图数据
        luma_orig = luma_orig_tensor.cpu().numpy()[0, 0]
        luma_final = luma_final_tensor.cpu().numpy()[0, 0]

        # 计算指标
        std_orig = np.std(luma_orig)
        std_final = np.std(luma_final)

        # 归一化 Std (用于衡量优化程度，消除亮度影响)
        target_nit = self.env.ideal_target_nit
        norm_std_orig = std_orig / target_nit
        norm_std_final = std_final / target_nit

        improve = (1 - std_final / std_orig) * 100

        # 5. 绘图
        self.fig.clear()

        # Original Luma
        ax1 = self.fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(luma_orig, cmap='jet')
        ax1.set_title(f"Original (G{g_val})\nStd: {std_orig:.3f} (Norm: {norm_std_orig:.4f})")
        self.fig.colorbar(im1, ax=ax1)
        ax1.axis('off')

        # Final Luma
        ax2 = self.fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(luma_final, cmap='jet')
        ax2.set_title(f"Corrected (Steps: {iters})\nStd: {std_final:.3f} (Imp: {improve:.1f}%)")
        self.fig.colorbar(im2, ax=ax2)
        ax2.axis('off')

        # Demura Table
        ax3 = self.fig.add_subplot(2, 3, 3)
        limit = max(1.0, np.max(np.abs(demura_table)))
        im3 = ax3.imshow(demura_table, cmap='coolwarm', vmin=-limit, vmax=limit)
        ax3.set_title("Demura Table (Delta Gray)")
        self.fig.colorbar(im3, ax=ax3)
        ax3.axis('off')

        # Histogram
        ax4 = self.fig.add_subplot(2, 1, 2)
        ax4.hist(luma_orig.ravel(), bins=100, alpha=0.5, label='Original', color='red', density=True)
        ax4.hist(luma_final.ravel(), bins=100, alpha=0.5, label='Corrected', color='green', density=True)
        ax4.axvline(target_nit, color='black', linestyle='--', label='Target Nit')
        ax4.legend()
        ax4.set_title(f"Luma Distribution (Target: {target_nit:.2f} nits)")
        ax4.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemuraInferenceUI()
    window.show()
    sys.exit(app.exec_())