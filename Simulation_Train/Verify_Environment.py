import sys
import os
import numpy as np
import tifffile as tiff
import cv2

# PyQt5 UI 库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QCheckBox, QGroupBox, QSpinBox, QFormLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

# Matplotlib 图表库
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MuraSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLED Mura 仿真环境")
        self.resize(1200, 800)

        # === 1. 全局数据 ===
        self.data_dir = "./"
        self.full_gamma = None  # 原始大图
        self.full_scale = None  # 原始大图
        self.gamma_roi = None  # 当前 ROI
        self.scale_roi = None  # 当前 ROI

        # === 2. 初始化界面 ===
        self.init_ui()

        # === 3. 加载数据 ===
        self.load_data()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左侧：图像显示区 ---
        img_group = QGroupBox("Mura 仿真视图 (Nit)")
        img_layout = QVBoxLayout(img_group)
        self.lbl_display = QLabel("No Data")
        self.lbl_display.setAlignment(Qt.AlignCenter)
        self.lbl_display.setStyleSheet("background-color: black;")
        self.lbl_display.setMinimumSize(600, 600)
        img_layout.addWidget(self.lbl_display)
        layout.addWidget(img_group, stretch=6)

        # --- 右侧：控制与图表 ---
        right_panel = QWidget()
        vbox_right = QVBoxLayout(right_panel)

        # 1. ROI 设置
        roi_group = QGroupBox("ROI 区域设置")
        form_roi = QFormLayout(roi_group)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(50, 2000)
        self.spin_w.setValue(200)
        self.spin_w.setSingleStep(50)
        self.spin_w.valueChanged.connect(self.update_roi_crop)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(50, 2000)
        self.spin_h.setValue(200)
        self.spin_h.setSingleStep(50)
        self.spin_h.valueChanged.connect(self.update_roi_crop)

        form_roi.addRow("宽度 (Width):", self.spin_w)
        form_roi.addRow("高度 (Height):", self.spin_h)
        vbox_right.addWidget(roi_group)

        # 2. 仿真参数
        sim_group = QGroupBox("仿真参数")
        vbox_sim = QVBoxLayout(sim_group)

        # 灰阶滑块
        vbox_sim.addWidget(QLabel("灰阶 (Gray Level):"))
        hbox_g = QHBoxLayout()
        self.slider_gray = QSlider(Qt.Horizontal)
        self.slider_gray.setRange(0, 255)
        self.slider_gray.setValue(128)
        self.slider_gray.valueChanged.connect(self.update_simulation)
        self.lbl_gray = QLabel("128")
        hbox_g.addWidget(self.slider_gray)
        hbox_g.addWidget(self.lbl_gray)
        vbox_sim.addLayout(hbox_g)

        vbox_sim.addSpacing(10)

        # 串扰滑块
        vbox_sim.addWidget(QLabel("串扰/Flare 强度 (Crosstalk %):"))
        hbox_c = QHBoxLayout()
        self.slider_cross = QSlider(Qt.Horizontal)
        self.slider_cross.setRange(0, 100)  # 0% - 100%
        self.slider_cross.setValue(0)  # 默认 0
        self.slider_cross.valueChanged.connect(self.update_simulation)
        self.lbl_cross = QLabel("0%")
        hbox_c.addWidget(self.slider_cross)
        hbox_c.addWidget(self.lbl_cross)
        vbox_sim.addLayout(hbox_c)

        vbox_right.addWidget(sim_group)

        # 3. 显示设置
        view_group = QGroupBox("显示设置")
        vbox_view = QVBoxLayout(view_group)
        self.chk_color = QCheckBox("伪彩色 (Jet)")
        self.chk_color.setChecked(True)
        self.chk_color.stateChanged.connect(self.update_simulation)

        self.chk_auto = QCheckBox("自动归一化 (Auto Level)")
        self.chk_auto.setChecked(True)
        self.chk_auto.stateChanged.connect(self.update_simulation)

        vbox_view.addWidget(self.chk_color)
        vbox_view.addWidget(self.chk_auto)
        vbox_right.addWidget(view_group)

        # 4. 直方图
        hist_group = QGroupBox("相对均匀性分布 (Uniformity Deviation %)")
        vbox_hist = QVBoxLayout(hist_group)
        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor='#F0F0F0')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
        vbox_hist.addWidget(self.canvas)
        vbox_right.addWidget(hist_group, stretch=3)

        layout.addWidget(right_panel, stretch=4)

    def load_data(self):
        g_path = os.path.join(self.data_dir, "gammaImg.tiff")
        s_path = os.path.join(self.data_dir, "scaleImg.tiff")

        if not os.path.exists(g_path) or not os.path.exists(s_path):
            self.create_dummy_data()

        try:
            # 加载全图到内存
            self.full_gamma = tiff.imread(g_path).astype(np.float32)
            self.full_scale = tiff.imread(s_path).astype(np.float32)
            print(f"Data Loaded. Size: {self.full_gamma.shape}")

            # 初始裁剪
            self.update_roi_crop()

        except Exception as e:
            self.lbl_display.setText(f"Load Error: {e}")

    def create_dummy_data(self):
        H, W = 1000, 1000
        # 模拟数据
        g = 2.2 + np.random.normal(0, 0.05, (H, W))
        s = 500 + np.random.normal(0, 5, (H, W))
        # 加上一些低频波动
        Y, X = np.ogrid[:H, :W]
        s += 20 * np.sin(X / 100.0) * np.cos(Y / 100.0)

        tiff.imwrite("gammaImg.tiff", g.astype(np.float32))
        tiff.imwrite("scaleImg.tiff", s.astype(np.float32))

    def update_roi_crop(self):
        """ 根据 SpinBox 的值裁剪数据 """
        if self.full_gamma is None: return

        h_full, w_full = self.full_gamma.shape
        w_req = self.spin_w.value()
        h_req = self.spin_h.value()

        # 居中裁剪
        cx, cy = w_full // 2, h_full // 2

        # 边界保护
        x1 = max(0, cx - w_req // 2)
        y1 = max(0, cy - h_req // 2)
        x2 = min(w_full, x1 + w_req)
        y2 = min(h_full, y1 + h_req)

        self.gamma_roi = self.full_gamma[y1:y2, x1:x2]
        self.scale_roi = self.full_scale[y1:y2, x1:x2]

        self.update_simulation()

    def update_simulation(self):
        if self.gamma_roi is None: return

        # 获取参数
        gray = self.slider_gray.value()
        cross_strength = self.slider_cross.value()  # 0 - 100

        self.lbl_gray.setText(str(gray))
        self.lbl_cross.setText(f"{cross_strength}%")

        # === 1. 理论计算 (Theoretical) ===
        norm_g = max(gray, 0.001) / 255.0
        # L_base = Scale * (G/255)^Gamma
        l_base = self.scale_roi * np.power(norm_g, self.gamma_roi)

        # === 2. 串扰模拟 (Crosstalk Simulation) ===
        if cross_strength > 0:
            # 使用高斯滤波模拟光/电扩散
            # sigma 决定扩散范围，这里设为 15，也可以做成可调参数
            l_blur = cv2.GaussianBlur(l_base, (0, 0), sigmaX=15, sigmaY=15)

            # 混合模型：
            # L_final = L_base + (Strength%) * L_blur
            # 这种“加法”模型模拟了漏光或电流扩散导致的额外亮度
            l_final = l_base + (cross_strength / 100.0) * l_blur
        else:
            # 强度为0时，等于完全没有串扰
            l_final = l_base

        # === 3. 更新直方图 ===
        self.update_histogram(l_final)

        # === 4. 渲染图像 ===
        self.render_view(l_final)

    def update_histogram(self, data):
        self.ax.clear()
        flat_data = data.ravel()
        mean_val = np.mean(flat_data)

        if mean_val < 1e-5:
            self.canvas.draw()
            return

        # 计算相对偏差百分比: (Value - Mean) / Mean * 100
        rel_data = (flat_data - mean_val) / mean_val * 100.0

        # 过滤极值
        rel_data = rel_data[np.abs(rel_data) < 50]

        # 绘制
        self.ax.hist(rel_data, bins=60, color='#2ca02c', alpha=0.7)

        # 锁定 X 轴：-30% ~ +30%
        # 这样可以直观对比低灰阶(宽)和高灰阶(窄)的区别
        self.ax.set_xlim(-30, 30)

        self.ax.set_title(f"Uniformity Distribution (Mean: {mean_val:.1f} nit)", fontsize=9)
        self.ax.set_xlabel("Deviation (%)", fontsize=8)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

    def render_view(self, img_float):
        h, w = img_float.shape

        if self.chk_auto.isChecked():
            vmin, vmax = np.min(img_float), np.max(img_float)
            if vmax > vmin + 1e-6:
                img_norm = (img_float - vmin) / (vmax - vmin) * 255.0
            else:
                img_norm = np.zeros_like(img_float)
        else:
            # 归一化到理论最大亮度 (Scale 的最大值)
            # 假设 Scale 约为 Nit 值
            global_max = np.max(self.full_scale) if self.full_scale is not None else 600
            img_norm = (img_float / global_max) * 255.0

        img_u8 = np.clip(img_norm, 0, 255).astype(np.uint8)

        if self.chk_color.isChecked():
            display = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            ch = 3
        else:
            display = img_u8
            fmt = QImage.Format_Grayscale8
            ch = 1

        qimg = QImage(display.data, w, h, w * ch, fmt)
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl_display.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.lbl_display.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MuraSimulator()
    win.show()
    sys.exit(app.exec_())