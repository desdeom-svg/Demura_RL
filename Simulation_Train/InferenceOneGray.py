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
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QSpinBox,
                             QPushButton, QLineEdit, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ==========================================
# 0. 配置与设备
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference running on: {device}")

# 请确认路径是否正确
SAVE_DIR = "Demura_Smooth_HDC"
MODEL_NAME = "best_actor.pth"
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
DATA_DIR = "./"


# ==========================================
# 1. 辅助函数 & 网络结构
# ==========================================
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


class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        # 修正 padding_mode='replicate'
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate')
        )

    def forward(self, x): return F.relu(x + self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 修正 padding_mode='replicate'
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate')
        )

    def forward(self, x): return F.relu(x + self.conv(x))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # 修正 padding_mode='replicate'
        self.entry = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, padding_mode='replicate'), nn.ReLU())
        self.global_feat = nn.Sequential(
            DilatedResBlock(32, dilation=1), DilatedResBlock(32, dilation=2), DilatedResBlock(32, dilation=5),
            DilatedResBlock(32, dilation=1), DilatedResBlock(32, dilation=2), DilatedResBlock(32, dilation=5)
        )
        self.local_refine = nn.Sequential(ResBlock(32), ResBlock(32), ResBlock(32))
        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode='replicate')

    def forward(self, x):
        x = self.entry(x)
        x = self.global_feat(x)
        x = self.local_refine(x)
        return self.exit(x)


# ==========================================
# 2. 推理环境 (支持动态 ROI)
# ==========================================
class InferenceMuraEnv:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.roi_size = 200  # 默认大小
        self.gamma_map = None
        self.scale_map = None
        self.full_gamma = None
        self.full_scale = None
        self.img_h = 0
        self.img_w = 0

        self.preload_full_images()

        self.crosstalk_kernel = get_gaussian_kernel(kernel_size=5, sigma=0.5, channels=1)
        self.pad_size = 2
        self.target_gray_val = 128.0
        self.crosstalk_percent = 0.0
        self.ideal_target_nit = 0.0

    def preload_full_images(self):
        g_path = os.path.join(self.data_dir, "gammaImg.tiff")
        s_path = os.path.join(self.data_dir, "scaleImg.tiff")

        # 自动生成假数据以防文件缺失
        if not os.path.exists(g_path) or not os.path.exists(s_path):
            print("TIFF files not found, generating dummy data (1000x1000)...")
            H, W = 1000, 1000
            g = 2.2 + np.random.normal(0, 0.05, (H, W))
            s = 500 + np.random.normal(0, 10, (H, W))
            tiff.imwrite(g_path, g.astype(np.float32))
            tiff.imwrite(s_path, s.astype(np.float32))

        self.full_gamma = tiff.imread(g_path).astype(np.float32)
        self.full_scale = tiff.imread(s_path).astype(np.float32)
        self.img_h, self.img_w = self.full_gamma.shape
        print(f"Full image loaded. Size: {self.img_w}x{self.img_h}")

    def set_roi(self, x, y, w, h):
        # 边界检查
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > self.img_w: w = self.img_w - x
        if y + h > self.img_h: h = self.img_h - y

        if w <= 0 or h <= 0:
            print("Warning: Invalid ROI, using default center crop.")
            x, y, w, h = self.img_w // 2 - 100, self.img_h // 2 - 100, 200, 200

        self.roi_size = w
        g_roi = self.full_gamma[y:y + h, x:x + w]
        s_roi = self.full_scale[y:y + h, x:x + w]

        self.gamma_map = torch.from_numpy(g_roi).unsqueeze(0).unsqueeze(0).to(device)
        self.scale_map = torch.from_numpy(s_roi).unsqueeze(0).unsqueeze(0).to(device)

        self.avg_scale = torch.mean(self.scale_map).item()
        self.avg_gamma = torch.mean(self.gamma_map).item()

        return x, y, w, h

    def update_params(self, gray_val, crosstalk_val):
        self.target_gray_val = float(gray_val)
        self.crosstalk_percent = float(crosstalk_val)
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
        rel_error = (current_luma - self.ideal_target_nit) / (self.ideal_target_nit + 1e-6)
        norm_gray = current_gray / 255.0
        state = torch.cat([rel_error, norm_gray], dim=1)
        return state


# ==========================================
# 3. 后台线程
# ==========================================
class InferenceWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, model, env, g_val, c_val, iters):
        super().__init__()
        self.model = model
        self.env = env
        self.g_val = g_val
        self.c_val = c_val
        self.iters = iters

    def run(self):
        with torch.no_grad():
            self.env.update_params(self.g_val, self.c_val)
            h, w = self.env.gamma_map.shape[2], self.env.gamma_map.shape[3]

            current_gray = torch.full((1, 1, h, w), float(self.g_val), device=device)
            luma_orig_tensor = self.env._physics_model(current_gray)

            for i in range(self.iters):
                current_luma = self.env._physics_model(current_gray)
                state = self.env.get_state(current_luma, current_gray)
                action = self.model(state)
                current_gray = torch.clamp(current_gray + action, 0, 255)

            luma_final_tensor = self.env._physics_model(current_gray)
            demura_table = (current_gray - float(self.g_val)).cpu().numpy()[0, 0]

            result_data = {
                "luma_orig": luma_orig_tensor.cpu().numpy()[0, 0],
                "luma_final": luma_final_tensor.cpu().numpy()[0, 0],
                "demura_table": demura_table,
                "target_nit": self.env.ideal_target_nit,
                "g_val": self.g_val,
                "iters": self.iters
            }
            self.finished.emit(result_data)


# ==========================================
# 4. UI 界面
# ==========================================
class DemuraInferenceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLED De-mura (动态 ROI 推理)")
        self.resize(1400, 900)

        # 初始化变量，防止 AttributeError
        self.env = None
        self.model = None

        try:
            self.env = InferenceMuraEnv(DATA_DIR)
            self.model = Actor().to(device)

            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print(f"Model loaded: {MODEL_PATH}")
            else:
                print(f"Warning: Model not found at {MODEL_PATH}, using random init.")
            self.model.eval()

            # 默认 ROI
            cx, cy = self.env.img_w // 2, self.env.img_h // 2
            self.env.set_roi(cx - 100, cy - 100, 200, 200)

        except Exception as e:
            # 如果初始化失败，弹出错误框并退出，防止后续崩溃
            print(f"Init Error: {e}")
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)

        # === 界面布局 ===
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧：绘图
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=3)

        # 右侧：控制面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # ROI 设置
        grp_roi = QGroupBox(f"ROI 区域设置 (Max: {self.env.img_w}x{self.env.img_h})")
        layout_roi = QFormLayout()

        self.input_x = QSpinBox();
        self.input_x.setRange(0, self.env.img_w);
        self.input_x.setValue(self.env.img_w // 2 - 100)
        self.input_y = QSpinBox();
        self.input_y.setRange(0, self.env.img_h);
        self.input_y.setValue(self.env.img_h // 2 - 100)
        self.input_w = QSpinBox();
        self.input_w.setRange(10, 2000);
        self.input_w.setValue(200)
        self.input_h = QSpinBox();
        self.input_h.setRange(10, 2000);
        self.input_h.setValue(200)

        btn_update_roi = QPushButton("更新区域并推理")
        btn_update_roi.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        btn_update_roi.clicked.connect(self.on_roi_update)

        layout_roi.addRow("Start X:", self.input_x)
        layout_roi.addRow("Start Y:", self.input_y)
        layout_roi.addRow("Width:", self.input_w)
        layout_roi.addRow("Height:", self.input_h)
        layout_roi.addRow(btn_update_roi)
        grp_roi.setLayout(layout_roi)
        right_layout.addWidget(grp_roi)

        # 参数控制
        grp_param = QGroupBox("推理参数")
        layout_param = QVBoxLayout()

        layout_param.addWidget(QLabel("Gray Level:"))
        self.slider_gray = QSlider(Qt.Horizontal)
        self.slider_gray.setRange(1, 250)
        self.slider_gray.setValue(32)
        self.lbl_gray = QLabel("32")
        self.slider_gray.valueChanged.connect(lambda v: self.lbl_gray.setText(str(v)))
        self.slider_gray.valueChanged.connect(self.on_param_change)
        layout_param.addWidget(self.slider_gray)
        layout_param.addWidget(self.lbl_gray)

        layout_param.addWidget(QLabel("Iterations:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 20)
        self.spin_iter.setValue(10)
        self.spin_iter.valueChanged.connect(self.on_param_change)
        layout_param.addWidget(self.spin_iter)

        grp_param.setLayout(layout_param)
        right_layout.addWidget(grp_param)
        right_layout.addStretch()

        # 防抖
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(200)
        self.debounce_timer.timeout.connect(self.start_inference)

        # 启动
        self.start_inference()

    def on_roi_update(self):
        if not self.model: return
        x = self.input_x.value()
        y = self.input_y.value()
        w = self.input_w.value()
        h = self.input_h.value()
        try:
            rx, ry, rw, rh = self.env.set_roi(x, y, w, h)
            self.input_x.setValue(rx)
            self.input_y.setValue(ry)
            self.input_w.setValue(rw)
            self.input_h.setValue(rh)
            self.start_inference()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_param_change(self):
        self.debounce_timer.start()

    def start_inference(self):
        if not self.model: return
        g_val = self.slider_gray.value()
        c_val = 0.0
        iters = self.spin_iter.value()

        self.worker = InferenceWorker(self.model, self.env, g_val, c_val, iters)
        self.worker.finished.connect(self.update_plots)
        self.worker.start()

    def update_plots(self, data):
        luma_orig = data["luma_orig"]
        luma_final = data["luma_final"]
        demura_table = data["demura_table"]
        target_nit = data["target_nit"]
        g_val = data["g_val"]

        std_orig = np.std(luma_orig)
        std_final = np.std(luma_final)
        improve = (1 - std_final / std_orig) * 100 if std_orig > 0 else 0

        self.fig.clear()

        ax1 = self.fig.add_subplot(2, 2, 1)
        im1 = ax1.imshow(luma_orig, cmap='jet')
        ax1.set_title(f"Original (Std: {std_orig:.4f})")
        self.fig.colorbar(im1, ax=ax1)
        ax1.axis('off')

        ax2 = self.fig.add_subplot(2, 2, 2)
        im2 = ax2.imshow(luma_final, cmap='jet')
        ax2.set_title(f"Corrected (Std: {std_final:.4f} | Imp: {improve:.1f}%)")
        self.fig.colorbar(im2, ax=ax2)
        ax2.axis('off')

        ax3 = self.fig.add_subplot(2, 2, 3)
        limit = max(1.0, np.max(np.abs(demura_table)))
        im3 = ax3.imshow(demura_table, cmap='coolwarm', vmin=-limit, vmax=limit)
        ax3.set_title(f"Demura Table (Delta Gray)")
        self.fig.colorbar(im3, ax=ax3)
        ax3.axis('off')

        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.hist(luma_orig.ravel(), bins=80, alpha=0.5, label='Original', color='red', density=True)
        ax4.hist(luma_final.ravel(), bins=80, alpha=0.5, label='Corrected', color='green', density=True)
        ax4.axvline(target_nit, color='black', linestyle='--', label='Target')
        ax4.legend()
        ax4.set_title("Luma Distribution")
        ax4.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemuraInferenceUI()
    window.show()
    sys.exit(app.exec_())