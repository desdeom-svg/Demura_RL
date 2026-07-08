# -*- coding: utf-8 -*-
"""实机 Demura 强化学习交互环境及硬件操作桥梁。

实现了与 C# 自动化客户端的 TCP 通信协议、显示图案生成、外置 PreDemura.exe
图像对齐与提取流程、以及 Gym 风格的强化学习环境包装（RealWorldEnv）。
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff

from config import (
    CommConfig,
    GrayConfig,
    PanelConfig,
    PathConfig,
    SafetyConfig,
    TrainConfig,
    get_capture_mim_path,
    get_capture_params,
    get_pattern_index,
    get_result_tiff_candidates,
    gray_to_capture_name,
    gray_to_display_name,
)

# 使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(log_dir: str) -> logging.Logger:
    """初始化并配置日志记录器，同时输出到控制台与本地日志文件。"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("RealWorldEnv")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # 写入文件（DEBUG 级别，记录详细数据帧与执行路径）
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # 打印到控制台（INFO 级别，记录主要训练进度）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# 初始化全局日志
logger = setup_logger(PathConfig.LOG_DIR)


class DeviceClient:
    """TCP 局域网套接字客户端，用于同 C# 自动化控制端交互。

    Python 端启动 Server 监听端口，C# 程序（DemuraAIDemo）作为 Client 接入。
    通过发送带有指令格式的 JSON 数据帧来驱动 C# 执行相机曝光调整、灰度图切换以及快门抓拍。
    """

    def __init__(
        self,
        ip: str = CommConfig.SERVER_IP,
        port: int = CommConfig.SERVER_PORT,
        timeout: int = CommConfig.SOCKET_TIMEOUT,
    ) -> None:
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None

    def start_server(self) -> None:
        """启动 TCP 监听服务，阻塞等待 C# 自动化系统连接。"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(self.timeout)
        logger.info("TCP server listening on %s:%s", self.ip, self.port)

        self._accept_client("Automation client connected")

    def _accept_client(self, log_message: str) -> None:
        try:
            self.client_socket, addr = self.server_socket.accept()
        except socket.timeout as exc:
            raise TimeoutError(
                f"Automation client did not connect to {self.ip}:{self.port} within {self.timeout} seconds."
            ) from exc
        self.client_socket.settimeout(self.timeout)
        logger.info("%s: %s", log_message, addr)

    def reconnect(self, reason: str, timeout: int = 15) -> None:
        if self.server_socket is None:
            raise RuntimeError("Socket server is not running.")

        if self.client_socket is not None:
            try:
                self.client_socket.close()
            except OSError:
                pass
            finally:
                self.client_socket = None

        previous_timeout = self.server_socket.gettimeout()
        self.server_socket.settimeout(timeout)
        logger.warning(
            "Automation socket dropped (%s); waiting up to %ss for client to reconnect.",
            reason,
            timeout,
        )
        try:
            self._accept_client("Automation client reconnected")
        finally:
            self.server_socket.settimeout(previous_timeout)

    def _request_once(self, message: str) -> Dict[str, object]:
        if not self.client_socket:
            raise RuntimeError("Socket client is not connected.")
        self.client_socket.sendall(message.encode("utf-8"))
        data = self.client_socket.recv(4096)
        if not data:
            raise RuntimeError("No response from automation client.")
        text = data.decode("utf-8", errors="ignore").strip()
        logger.debug("Automation response: %s", text)
        return self._parse_response(text)

    def request(self, payload: Dict[str, object]) -> Dict[str, object]:
        """向 C# 端发送 JSON 形式的控制指令，并阻塞接收响应。"""
        message = json.dumps(payload, ensure_ascii=True)
        logger.debug("Send automation command: %s", message)
        try:
            return self._request_once(message)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout, OSError, RuntimeError) as exc:
            recoverable_runtime = isinstance(exc, RuntimeError) and str(exc) == "No response from automation client."
            if not recoverable_runtime and not isinstance(
                exc,
                (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout, OSError),
            ):
                raise
            self.reconnect(str(exc))
            logger.debug("Retrying automation command after reconnect.")
            return self._request_once(message)

    @staticmethod
    def _parse_response(text: str) -> Dict[str, object]:
        """解析来自 C# 客户端的响应字符串。

        兼容 JSON 数据帧及传统的 legacy 协议（如 START/END 消息或带有 ERROR 的异常消息）。
        """
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        upper_text = text.upper()
        if upper_text.startswith(CommConfig.LEGACY_ERROR_PREFIX):
            return {"ok": False, "message": text}
        if upper_text.startswith(CommConfig.LEGACY_END_CMD):
            return {"ok": True, "message": text}
        return {"ok": False, "message": f"Unexpected response: {text}"}

    def close(self) -> None:
        """关闭通信套接字连接。"""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        logger.info("Socket connection closed.")


class ImageProcessor:
    """图像渲染与预处理流程助手。

    主要负责将 2D 灰度补偿矩阵拼接到完整的屏幕大图（Panel BMP）中，
    调起 `PreDemura.exe` 处理相机图像以完成亚像素级别的几何对齐与畸变矫正，
    并最终载入亮度（Luma）TIFF 结果。
    """

    def __init__(self) -> None:
        os.makedirs(PathConfig.TRANSFER_BMP_DIR, exist_ok=True)
        os.makedirs(PathConfig.CAMERA_IMAGE_DIR, exist_ok=True)
        os.makedirs(PathConfig.RESULT_DIR, exist_ok=True)
        self.quantization_residuals: Dict[str, np.ndarray] = {}
        self.last_quantized_roi: Optional[np.ndarray] = None
        self.last_step_quantized_delta: Optional[np.ndarray] = None
        self.last_render_stats: Dict[str, float] = {}
        self.previous_quantized_rois: Dict[str, np.ndarray] = {}
        self.last_pre_demura_stdout: str = ""
        self.last_pre_demura_stderr: str = ""
        self.last_pre_demura_failure_reason: Optional[str] = None
        self.last_render_bmp_path: Optional[str] = None

    @staticmethod
    def _decode_process_output(raw: bytes) -> str:
        if not raw:
            return ""
        for encoding in ("utf-8", "gbk", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def reset_quantization_state(self, display_name: Optional[str] = None) -> None:
        """Reset quantization carry-over so each episode starts from a clean panel state."""
        if display_name is None:
            self.quantization_residuals.clear()
            self.previous_quantized_rois.clear()
        else:
            self.quantization_residuals.pop(display_name, None)
            self.previous_quantized_rois.pop(display_name, None)
        self.last_quantized_roi = None
        self.last_step_quantized_delta = None
        self.last_render_stats = {}
        self.last_render_bmp_path = None

    def render_panel_bmp(self, gray_map_2d: np.ndarray, display_name: str) -> str:
        """生成一个全屏幕分辨率的 BMP 图片，并在指定的 ROI 感兴趣区域填入当前的补偿灰阶值。

        参数:
            gray_map_2d  : 2D 浮点型灰度矩阵，尺寸为 (ROI_HEIGHT, ROI_WIDTH)
            display_name : 图片文件名标识（如 "W16"）
        返回:
            生成的 24位 RGB (实际全灰度通道一致) BMP 文件的绝对物理路径
        """
        roi_h, roi_w = gray_map_2d.shape
        gray_map_2d = gray_map_2d.astype(np.float32, copy=False)
        residual = self.quantization_residuals.get(display_name)
        if residual is None or residual.shape != gray_map_2d.shape:
            residual = np.zeros_like(gray_map_2d, dtype=np.float32)

        previous_quantized = self.previous_quantized_rois.get(display_name)
        quant_input = gray_map_2d + residual
        clipped = np.clip(quant_input, 0.0, 255.0)
        quantized_roi = np.round(clipped).astype(np.uint8)
        new_residual = clipped - quantized_roi.astype(np.float32)
        self.quantization_residuals[display_name] = new_residual.astype(np.float32, copy=False)
        self.last_quantized_roi = quantized_roi
        self.previous_quantized_rois[display_name] = quantized_roi.copy()
        if previous_quantized is None or previous_quantized.shape != quantized_roi.shape:
            step_delta = np.zeros_like(quantized_roi, dtype=np.int16)
            step_changed_ratio = 0.0
            step_delta_abs_mean = 0.0
        else:
            step_delta = quantized_roi.astype(np.int16) - previous_quantized.astype(np.int16)
            step_changed_ratio = float(np.mean(step_delta != 0))
            step_delta_abs_mean = float(np.mean(np.abs(step_delta)))
        self.last_step_quantized_delta = step_delta.astype(np.float32)

        digits = "".join(ch for ch in display_name if ch.isdigit())
        target_gray = int(digits) if digits else int(round(float(np.mean(gray_map_2d))))
        target_gray = int(np.clip(target_gray, 0, 255))
        quantized_delta = quantized_roi.astype(np.int16) - target_gray
        self.last_render_stats = {
            "quantized_gray_mean": float(np.mean(quantized_roi)),
            "quantized_gray_std": float(np.std(quantized_roi)),
            "quantized_delta_abs_mean": float(np.mean(np.abs(quantized_delta))),
            "quantized_changed_ratio": float(np.mean(quantized_roi != target_gray)),
            "quantized_step_delta_abs_mean": step_delta_abs_mean,
            "quantized_step_changed_ratio": step_changed_ratio,
            "render_residual_abs_mean": float(np.mean(np.abs(new_residual))),
            "render_rounding_abs_mean": float(np.mean(np.abs(quantized_roi.astype(np.float32) - gray_map_2d))),
        }
        # 初始化一个全黑的屏幕画幅
        panel = np.zeros((PanelConfig.PANEL_HEIGHT, PanelConfig.PANEL_WIDTH), dtype=np.uint8)
        # 将传入的 ROI 灰度矩阵进行四舍五入并夹逼裁剪至 [0, 255] 整数灰度区间，写入指定坐标区域
        panel[
            PanelConfig.ROI_START_Y : PanelConfig.ROI_START_Y + roi_h,
            PanelConfig.ROI_START_X : PanelConfig.ROI_START_X + roi_w,
        ] = quantized_roi

        bmp_path = os.path.join(PathConfig.TRANSFER_BMP_DIR, f"{display_name}.bmp")
        # 将单通道灰度图转换为三通道 BGR 图像以符合显示设备读取要求
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(bmp_path, panel_bgr)
        self.last_render_bmp_path = bmp_path
        return bmp_path

    def find_static_bmp(self, gray: int) -> Optional[str]:
        """定位静态参考图像 BMP 文件的存储路径，W253 自动生成。"""
        if gray == 253:
            factory_w253 = os.path.join("E:\\pgIn", PanelConfig.PRODUCT_NAME, "W253.bmp")
            if os.path.exists(factory_w253):
                bmp_path = os.path.join(PathConfig.TRANSFER_BMP_DIR, "W253.bmp")
                shutil.copy2(factory_w253, bmp_path)
                logger.info("Copied factory W253 BMP: %s -> %s", factory_w253, bmp_path)
                return bmp_path
            return self._generate_w253_locator()
        display_name = gray_to_display_name(gray)
        candidates = [
            os.path.join(PathConfig.STATIC_BMP_DIR, f"{display_name}.bmp"),
            os.path.join(PathConfig.TRANSFER_BMP_DIR, f"{display_name}.bmp"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _generate_w253_locator() -> str:
        """Generate W253 locator BMP with corner circles at ROI corners for PreDemura alignment."""
        panel_h, panel_w = PanelConfig.PANEL_HEIGHT, PanelConfig.PANEL_WIDTH
        rx, ry = PanelConfig.ROI_START_X, PanelConfig.ROI_START_Y
        rw, rh = PanelConfig.ROI_WIDTH, PanelConfig.ROI_HEIGHT
        roi_corners = [
            (rx, ry), (rx + rw - 1, ry),
            (rx, ry + rh - 1), (rx + rw - 1, ry + rh - 1),
        ]
        img = np.full((panel_h, panel_w, 3), 128, dtype=np.uint8)
        for cx, cy in roi_corners:
            cv2.circle(img, (cx, cy), 25, (0, 0, 0), -1, cv2.LINE_8)
            cv2.circle(img, (cx, cy), 5, (0, 200, 0), -1, cv2.LINE_8)
        # edge reference circles
        edge_x = rx + 200
        for ex in [edge_x, rx + rw - 1 - edge_x]:
            cv2.circle(img, (ex, ry), 30, (0, 0, 0), -1, cv2.LINE_8)
            cv2.circle(img, (ex, ry + rh - 1), 30, (0, 0, 0), -1, cv2.LINE_8)
        bmp_path = os.path.join(PathConfig.TRANSFER_BMP_DIR, "W253.bmp")
        os.makedirs(PathConfig.TRANSFER_BMP_DIR, exist_ok=True)
        cv2.imwrite(bmp_path, img)
        logger.info("Generated W253 locator BMP: %s (panel %dx%d, ROI %dx%d@%d,%d)",
                     bmp_path, panel_w, panel_h, rw, rh, rx, ry)
        return bmp_path

    def run_pre_demura(self) -> bool:
        """通过子进程同步执行外部图像几何及畸变对齐程序（PreDemura.exe）。

        该工具会将工业相机抓拍的原始 MIM 图像或静态参考图转化并裁剪为亚像素配准的 TIFF 格式亮度结果。
        """
        exe_path = PathConfig.PRE_DEMURA_EXE
        if not os.path.exists(exe_path):
            logger.error("PreDemura.exe not found: %s", exe_path)
            return False

        try:
            # 调起子进程并阻塞等待完成，设置超时时间为 120 秒
            process = subprocess.Popen(
                exe_path,
                cwd=PathConfig.CUR_DIR,
            )
            process.wait(timeout=120)
            logger.debug(
                "PreDemura completed. cwd=%s returncode=%s",
                PathConfig.CUR_DIR,
                process.returncode,
            )
            return process.returncode == 0
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("PreDemura timed out.")
            return False
        except Exception as exc:
            logger.error("Failed to execute PreDemura: %s", exc, exc_info=True)
            return False

    def run_pre_demura_checked(self) -> bool:
        """执行 PreDemura，并把外部输出纳入失败判定。"""
        exe_path = PathConfig.PRE_DEMURA_EXE
        self.last_pre_demura_stdout = ""
        self.last_pre_demura_stderr = ""
        self.last_pre_demura_failure_reason = None
        if not os.path.exists(exe_path):
            logger.error("PreDemura.exe not found: %s", exe_path)
            self.last_pre_demura_failure_reason = f"PreDemura.exe not found: {exe_path}"
            return False

        try:
            completed = subprocess.run(
                [exe_path],
                cwd=PathConfig.CUR_DIR,
                capture_output=True,
                timeout=120,
            )
            self.last_pre_demura_stdout = self._decode_process_output(completed.stdout).strip()
            self.last_pre_demura_stderr = self._decode_process_output(completed.stderr).strip()
            combined_output = "\n".join(
                part for part in (self.last_pre_demura_stdout, self.last_pre_demura_stderr) if part
            )
            if self.last_pre_demura_stdout:
                logger.debug("PreDemura stdout: %s", self.last_pre_demura_stdout)
            if self.last_pre_demura_stderr:
                logger.warning("PreDemura stderr: %s", self.last_pre_demura_stderr)
            logger.debug(
                "PreDemura checked. cwd=%s returncode=%s",
                PathConfig.CUR_DIR,
                completed.returncode,
            )
            if completed.returncode != 0:
                self.last_pre_demura_failure_reason = (
                    f"PreDemura returned {completed.returncode}: {combined_output or 'no output'}"
                )
                logger.error(self.last_pre_demura_failure_reason)
                return False
            lowered_output = combined_output.lower()
            if "process fail" in lowered_output or "定位图" in combined_output:
                self.last_pre_demura_failure_reason = combined_output or "PreDemura locator process failed."
                logger.error("PreDemura reported locator failure: %s", self.last_pre_demura_failure_reason)
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("PreDemura timed out.")
            self.last_pre_demura_failure_reason = "PreDemura timed out."
            return False
        except Exception as exc:
            logger.error("Failed to execute PreDemura: %s", exc, exc_info=True)
            self.last_pre_demura_failure_reason = f"Failed to execute PreDemura: {exc}"
            return False

    def dump_failure_artifacts(self, save_dir: Optional[str], phase: str, gray: Optional[int] = None) -> None:
        if not save_dir:
            return
        artifacts_dir = os.path.join(save_dir, "failure_artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        copied_file_metadata = {}
        if self.last_render_bmp_path and os.path.exists(self.last_render_bmp_path):
            panel_target = os.path.join(artifacts_dir, f"{phase}_panel.bmp")
            shutil.copy2(self.last_render_bmp_path, panel_target)
            panel_stat = os.stat(panel_target)
            copied_file_metadata[os.path.basename(panel_target)] = {
                "size_bytes": int(panel_stat.st_size),
                "mtime": datetime.fromtimestamp(panel_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
        copied_inputs = []
        if gray is not None:
            for input_gray in (gray, 252, 253):
                mim_path = get_capture_mim_path(input_gray)
                if os.path.exists(mim_path):
                    target_name = f"{phase}_{os.path.basename(mim_path)}"
                    target_path = os.path.join(artifacts_dir, target_name)
                    shutil.copy2(mim_path, target_path)
                    copied_inputs.append(target_name)
                    target_stat = os.stat(target_path)
                    copied_file_metadata[target_name] = {
                        "size_bytes": int(target_stat.st_size),
                        "mtime": datetime.fromtimestamp(target_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
            for result_path in get_result_tiff_candidates(gray):
                if os.path.exists(result_path):
                    target_name = f"{phase}_{os.path.basename(result_path)}"
                    target_path = os.path.join(artifacts_dir, target_name)
                    shutil.copy2(result_path, target_path)
                    copied_inputs.append(target_name)
                    target_stat = os.stat(target_path)
                    copied_file_metadata[target_name] = {
                        "size_bytes": int(target_stat.st_size),
                        "mtime": datetime.fromtimestamp(target_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
        summary = {
            "phase": phase,
            "gray": gray,
            "pre_demura_failure_reason": self.last_pre_demura_failure_reason,
            "pre_demura_stdout": self.last_pre_demura_stdout,
            "pre_demura_stderr": self.last_pre_demura_stderr,
            "render_stats": self.last_render_stats,
            "render_bmp_path": self.last_render_bmp_path,
            "copied_inputs": copied_inputs,
            "copied_file_metadata": copied_file_metadata,
        }
        with open(os.path.join(artifacts_dir, f"{phase}_summary.json"), "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    def read_result(self, gray: int) -> Optional[np.ndarray]:
        """读取外部 PreDemura.exe 对齐输出的特定灰阶 TIFF 亮度数据图（2D 浮点矩阵）。"""
        for result_path in get_result_tiff_candidates(gray):
            if not os.path.exists(result_path):
                logger.debug("Result TIFF does not exist yet: %s", result_path)
                continue

            # 使用 OpenCV 读取原生 TIFF 图像（-1 参数代表以原生深度/多通道格式读取，保留 float32 或 uint16 精度）
            result = cv2.imread(result_path, -1)
            if result is not None:
                logger.debug("Loaded result image: %s", result_path)
                return result.astype(np.float32)
            logger.warning("Failed to read result TIFF even though it exists: %s", result_path)

        logger.error("No result TIFF found for gray %s.", gray)
        return None

    def read_result_with_wait(
        self,
        gray: int,
        wait_attempts: int = 24,
        wait_interval_s: float = 0.25,
    ) -> Optional[np.ndarray]:
        for attempt in range(1, wait_attempts + 1):
            result = self.read_result(gray)
            if result is not None:
                return result
            if attempt < wait_attempts:
                time.sleep(wait_interval_s)
        logger.error(
            "No result TIFF found for gray %s after waiting %.2fs.",
            gray,
            wait_attempts * wait_interval_s,
        )
        return None


def calc_gradient_loss(img: torch.Tensor) -> torch.Tensor:
    """计算图像的空间总变差（Total Variation, 梯度损失）。

    物理意义：衡量补偿亮度图的局部平滑度，通过惩罚相邻像素间过大的亮度差异，
    来防止强化学习 Actor 补偿表产生局部空间上的高频噪点与斑马纹效应。
    """
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return dy.mean() + dx.mean()


def _adaptive_odd_kernel(img: torch.Tensor, requested: int) -> int:
    height = int(img.shape[-2])
    width = int(img.shape[-1])
    kernel = min(int(requested), height, width)
    if kernel < 3:
        return 1
    if kernel % 2 == 0:
        kernel -= 1
    return max(1, kernel)


def _separable_box_filter(img: torch.Tensor, requested_kernel: int) -> torch.Tensor:
    kernel = _adaptive_odd_kernel(img, requested_kernel)
    if kernel <= 1:
        return img
    pad = kernel // 2
    filtered = F.avg_pool2d(
        img,
        kernel_size=(1, kernel),
        stride=1,
        padding=(0, pad),
        count_include_pad=False,
    )
    filtered = F.avg_pool2d(
        filtered,
        kernel_size=(kernel, 1),
        stride=1,
        padding=(pad, 0),
        count_include_pad=False,
    )
    return filtered


def compute_uniformity_reward_from_rel_error(rel_error: torch.Tensor):
    """Multi-band visual reward for mura reduction.

    Low-frequency mura, mid-frequency cloud/block artifacts, high-frequency
    speckle, and local tail errors are penalized separately. This prevents
    optimizing only global std while leaving visible bright/dark patches.
    """
    rel_error = rel_error.float()
    mean_error = torch.mean(rel_error)
    mean_loss = torch.abs(mean_error)
    std_norm = torch.std(rel_error)
    grad_loss = calc_gradient_loss(rel_error)
    abs_error = torch.abs(rel_error)
    p95_abs_error = torch.quantile(abs_error.flatten(), 0.95)
    tail_abs_p99 = torch.quantile(abs_error.flatten(), 0.99)
    max_abs_error = torch.max(abs_error)
    mse_raw = torch.mean(rel_error ** 2)

    low_band = _separable_box_filter(rel_error, 101)
    mid_reference = _separable_box_filter(rel_error, 31)
    mid_band = mid_reference - low_band
    high_band = rel_error - mid_reference

    low_std = torch.std(low_band)
    mid_abs_p99 = torch.quantile(torch.abs(mid_band).flatten(), 0.99)
    high_abs_p99 = torch.quantile(torch.abs(high_band).flatten(), 0.99)
    profile_loss = calc_profile_loss(low_band)

    r_std = -std_norm * 22.0
    r_low = -low_std * 120.0
    r_mid = -mid_abs_p99 * 420.0
    r_high = -high_abs_p99 * 220.0
    r_tail = -tail_abs_p99 * 110.0
    r_profile = -profile_loss * 110.0
    r_grad = -grad_loss * 55.0
    r_mean = -mean_loss * 20.0
    total = r_std + r_low + r_mid + r_high + r_tail + r_profile + r_grad + r_mean
    reward_info = {
        "r_mse": 0.0,
        "r_std": r_std.item(),
        "r_ssim": 0.0,
        "r_low": r_low.item(),
        "r_mid": r_mid.item(),
        "r_high": r_high.item(),
        "r_tail": r_tail.item(),
        "r_profile": r_profile.item(),
        "r_grad": r_grad.item(),
        "r_mean": r_mean.item(),
        "mse_raw": float(mse_raw.item()),
        "mean_error": float(mean_error.item()),
        "mean_loss": float(mean_loss.item()),
        "std_norm": float(std_norm.item()),
        "low_std": float(low_std.item()),
        "mid_abs_p99": float(mid_abs_p99.item()),
        "high_abs_p99": float(high_abs_p99.item()),
        "tail_abs_p99": float(tail_abs_p99.item()),
        "profile_loss": float(profile_loss.item()),
        "p95_abs_error": float(p95_abs_error.item()),
        "max_abs_error": float(max_abs_error.item()),
        "visual_std": float(std_norm.item()),
        "raw_std": float(std_norm.item()),
        "grad_loss": float(grad_loss.item()),
        "total": total.item(),
    }
    return total, reward_info


def calc_lowpass_std(img: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """对亮度图做低通平滑后计算标准差，更贴近人眼对低频 mura 的感知。"""
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = F.avg_pool2d(img, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.std(blurred)


def calc_profile_loss(img: torch.Tensor) -> torch.Tensor:
    """计算按行/按列均值轮廓的起伏，用于抑制 banding 与块状不均。"""
    row_profile = torch.mean(img, dim=3)
    col_profile = torch.mean(img, dim=2)
    return 0.5 * (torch.std(row_profile) + torch.std(col_profile))


def calc_center_mean(img: torch.Tensor, window_size: int) -> torch.Tensor:
    """Use the center patch mean as the physical target luma baseline for a pure-gray reset."""
    _, _, height, width = img.shape
    size = max(1, int(window_size))
    size = min(size, height, width)
    half = size // 2
    center_y = height // 2
    center_x = width // 2
    y0 = max(0, center_y - half)
    x0 = max(0, center_x - half)
    y1 = min(height, y0 + size)
    x1 = min(width, x0 + size)
    patch = img[:, :, y0:y1, x0:x1]
    return torch.mean(patch)


def apply_multi_pass_lowpass(img: torch.Tensor, kernel_size: int, passes: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    passes = max(0, int(passes))
    if kernel_size <= 1 or passes <= 0:
        return img
    if kernel_size % 2 == 0:
        kernel_size += 1
    filtered = img
    padding = kernel_size // 2
    for _ in range(passes):
        filtered = F.avg_pool2d(filtered, kernel_size=kernel_size, stride=1, padding=padding)
    return filtered


class RealWorldEnv:
    """实机 Demura 强化学习闭环训练环境（Gym 接口封装）。

    管理状态构建、动作检查与限幅、多灰阶亮度加权奖励的计算，并调度通信与图像处理管道。
    """

    def __init__(
        self,
        gray_candidates: Optional[Iterable[int]] = None,
        freeze_target_per_run: bool = True,
        no_quantization_carryover: bool = False,
        train_roi_size: int = 0,
        reward_roi_size: int = 0,
    ) -> None:
        self.roi_h = PanelConfig.ROI_HEIGHT
        self.roi_w = PanelConfig.ROI_WIDTH
        self.gray_candidates = list(gray_candidates or [GrayConfig.DEFAULT_SINGLE_GRAY])
        self.freeze_target_per_run = bool(freeze_target_per_run)
        # 当前训练的控制目标基准灰阶值（浮点与整型表示）
        self.target_gray = float(self.gray_candidates[0])
        self.current_gray_int = int(self.gray_candidates[0])
        self.display_name = gray_to_display_name(self.current_gray_int)

        self.device_client = DeviceClient()
        self.image_processor = ImageProcessor()
        self.displayed_gray_map: Optional[torch.Tensor] = None

        # 状态张量（在 device/GPU 上运算）
        self.current_gray_map: Optional[torch.Tensor] = None  # 当前写入屏幕的绝对灰阶地图: (1, 1, H, W)
        self.current_luma_map: Optional[torch.Tensor] = None  # 从相机提取到的实际亮度地图: (1, 1, H, W)
        self.target_mean_nit: Optional[float] = None          # 目标亮度的基准均值（Nit，第0步的平均亮度）
        self.initial_std: Optional[float] = None              # 校正前亮度的原始空间标准差 (表征初始 Mura 严重程度)
        self.total_delta: Optional[torch.Tensor] = None       # 累计动作调整量 (Delta Gray) 地图，用于安全限制校验

        self.baseline_gray_map: Optional[torch.Tensor] = None
        self.target_luma_map: Optional[torch.Tensor] = None
        self.run_target_mean_nit: Optional[float] = None
        self.run_target_luma_map: Optional[torch.Tensor] = None
        self.run_target_norm_std: Optional[float] = None
        self.last_capture_used_retry = False
        self.last_capture_retry_count = 1
        self.baseline_penalty_weight: float = float(getattr(TrainConfig, "BASELINE_DEVIATION_W", 0.0))
        self.episode_count = 0
        self.step_count = 0
        self._static_refs_ok: bool = False
        self.no_quantization_carryover: bool = bool(no_quantization_carryover)
        self._train_roi: Optional[Tuple[int, int, int, int]] = None  # (top, left, h, w)
        if int(train_roi_size) > 0:
            self._train_roi = self._compute_train_roi(int(train_roi_size))
        resolved_reward_roi_size = int(reward_roi_size) if int(reward_roi_size) > 0 else int(train_roi_size)
        self._reward_roi: Optional[Tuple[int, int, int, int]] = None
        if resolved_reward_roi_size > 0:
            self._reward_roi = self._compute_center_roi(resolved_reward_roi_size, "Reward ROI")

    def _compute_center_roi(self, size: int, label: str):
        h, w = self.roi_h, self.roi_w
        crop_h = min(size, h)
        crop_w = min(size, w)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        logger.info("%s: top=%d left=%d size=%dx%d (full ROI %dx%d)", label, top, left, crop_h, crop_w, h, w)
        return (top, left, crop_h, crop_w)

    def _compute_train_roi(self, size: int):
        return self._compute_center_roi(size, "Train ROI")

    def _crop_to_train_roi(self, tensor_4d: torch.Tensor) -> torch.Tensor:
        if self._train_roi is None:
            return tensor_4d
        if tensor_4d.shape[-2:] != (self.roi_h, self.roi_w):
            return tensor_4d
        top, left, h, w = self._train_roi
        return tensor_4d[..., top:top + h, left:left + w].contiguous()

    def _crop_to_reward_roi(self, tensor_4d: torch.Tensor) -> torch.Tensor:
        if self._reward_roi is None:
            return tensor_4d
        if tensor_4d.shape[-2:] != (self.roi_h, self.roi_w):
            return tensor_4d
        top, left, h, w = self._reward_roi
        return tensor_4d[..., top:top + h, left:left + w].contiguous()

    def _update_baseline_std_metrics(self) -> float:
        luma = self._crop_to_reward_roi(self.current_luma_map)
        target = self._crop_to_reward_roi(self.target_luma_map)
        norm_luma = luma / (target + 1e-6)
        self.initial_std = float(torch.std(luma).item())
        initial_norm_std = float(torch.std(norm_luma).item())
        if self.run_target_norm_std is None or not self.freeze_target_per_run:
            self.run_target_norm_std = initial_norm_std
        return initial_norm_std

    def _expand_luma_to_full_roi_if_needed(self, luma_np: np.ndarray, fill_value: Optional[float] = None) -> np.ndarray:
        luma = np.asarray(luma_np, dtype=np.float32)
        if luma.shape == (self.roi_h, self.roi_w):
            return luma
        if luma.ndim != 2:
            raise ValueError(f"Luma result must be 2D, got shape {luma.shape}")

        target_roi = self._reward_roi
        if target_roi is None:
            crop_h = min(int(luma.shape[0]), self.roi_h)
            crop_w = min(int(luma.shape[1]), self.roi_w)
            top = (self.roi_h - crop_h) // 2
            left = (self.roi_w - crop_w) // 2
            target_roi = (top, left, crop_h, crop_w)
        top, left, h, w = target_roi
        if luma.shape != (h, w):
            raise ValueError(
                f"Luma crop shape {luma.shape} does not match reward ROI {(h, w)}. "
                "Use a matching reward ROI or restore full-ROI PreDemura output."
            )
        fill = float(fill_value) if fill_value is not None else float(np.nanmean(luma))
        expanded = np.full((self.roi_h, self.roi_w), fill, dtype=np.float32)
        expanded[top:top + h, left:left + w] = luma
        logger.info(
            "Expanded luma crop %sx%s into full ROI %sx%s at top=%s left=%s.",
            h,
            w,
            self.roi_h,
            self.roi_w,
            top,
            left,
        )
        return expanded

    def connect(self) -> None:
        """建立 TCP 服务器并等待 C# 客户端连接上线。"""
        self.device_client.start_server()

    def close(self) -> None:
        """断开连接并关闭通信服务。"""
        self.device_client.close()

    def set_target_gray(self, gray: int) -> None:
        """切换当前 Episode 优化的目标灰阶。"""
        self.current_gray_int = int(gray)
        self.target_gray = float(gray)
        self.display_name = gray_to_display_name(self.current_gray_int)

    def ensure_static_references(self, force_refresh: bool = False) -> None:
        """确保定位用的静态基准参考图像（W253）已拍照并被 PreDemura 提取。

        如果强制刷新或本地文件不存在，则驱动 C# 重新进行定位拍照。
        """
        for ref_gray in GrayConfig.STATIC_REFERENCE_GRAYS:
            capture_path = get_capture_mim_path(ref_gray)
            if os.path.exists(capture_path) and not force_refresh:
                continue

            logger.info("Capturing static reference W%s.", ref_gray)
            bmp_path = self.image_processor.find_static_bmp(ref_gray)
            self._send_capture_command(
                gray=ref_gray,
                bmp_path=bmp_path,
                save_path=capture_path,
                exposure=get_capture_params(ref_gray)[0],
                gain=get_capture_params(ref_gray)[1],
                delay_ms=500,
            )
        self._static_refs_ok = True

    def _send_capture_command(
        self,
        gray: int,
        bmp_path: Optional[str],
        save_path: str,
        exposure: float,
        gain: float,
        delay_ms: int,
    ) -> None:
        """构造并向 C# 自动化客户端下发 SNAP 控制帧。"""
        payload: Dict[str, object] = {
            "cmd": "SNAP",
            "pattern_name": gray_to_display_name(gray),
            "pattern_index": get_pattern_index(gray),
            "save_path": save_path,
            "exposure": exposure,
            "gain": gain,
            "delay_ms": delay_ms,  # 给显示屏切换图案及液晶分子响应留出的延迟（毫秒）
        }
        if bmp_path and os.path.exists(bmp_path):
            payload["bmp_path"] = bmp_path  # 若为动态渲染图，则提供本地 BMP 的绝对路径以供 C# 传输显示

        response = self.device_client.request(payload)
        if not response.get("ok"):
            raise RuntimeError(f"Capture failed for gray {gray}: {response}")

    def _capture_real_luma(self, delay_ms: int = 400) -> Optional[np.ndarray]:
        """驱动硬件抓拍当前补偿图，并运行外部几何对准提取亮度数据。"""
        t0 = time.perf_counter()
        # 1. 确保定位参考图就绪（缓存后跳过文件存在性检查）
        if not self._static_refs_ok:
            self.ensure_static_references(force_refresh=False)

        if self.current_gray_map is None:
            raise RuntimeError("Gray map is not initialized.")

        # 2. 渲染并生成面板要显示的全分辨率 BMP 图像
        if self.no_quantization_carryover:
            self.image_processor.reset_quantization_state(self.display_name)
        gray_np = self.current_gray_map.detach().cpu().numpy()[0, 0]
        bmp_path = self.image_processor.render_panel_bmp(gray_np, self.display_name)
        t1 = time.perf_counter()
        save_path = get_capture_mim_path(self.current_gray_int)
        exposure, gain = get_capture_params(self.current_gray_int)

        # 3. 向 C# 下发 SNAP 抓拍命令
        self._send_capture_command(
            gray=self.current_gray_int,
            bmp_path=bmp_path,
            save_path=save_path,
            exposure=exposure,
            gain=gain,
            delay_ms=delay_ms,
        )
        t2 = time.perf_counter()

        # 4. 同步运行 PreDemura 进行图像纠偏对齐
        if not self.image_processor.run_pre_demura_checked():
            if self.image_processor.last_pre_demura_failure_reason:
                raise RuntimeError(self.image_processor.last_pre_demura_failure_reason)
            return None
        t3 = time.perf_counter()
        # 5. 读取纠偏后的亮度对齐 TIFF 图像
        wait_attempts = 24 if delay_ms >= 1000 else 20
        result = self.image_processor.read_result_with_wait(
            self.current_gray_int,
            wait_attempts=wait_attempts,
            wait_interval_s=0.25,
        )
        t4 = time.perf_counter()
        logger.debug(
            "[perf] ep=%d step=%d gray=%d delay=%dms render=%.3fs capture=%.3fs predemura=%.3fs read=%.3fs total=%.3fs",
            self.episode_count, self.step_count, self.current_gray_int, delay_ms,
            t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0,
        )
        return result

    def _capture_real_luma_with_retry(
        self,
        phase: str,
        retries: int = 3,
        retry_delay_s: float = 1.0,
        delay_ms: int = 250,
    ) -> Optional[np.ndarray]:
        self.last_capture_used_retry = False
        self.last_capture_retry_count = 1
        for attempt in range(1, retries + 1):
            try:
                luma_np = self._capture_real_luma(delay_ms=delay_ms)
                if luma_np is not None:
                    self.last_capture_retry_count = attempt
                    self.last_capture_used_retry = attempt > 1
                    if attempt > 1:
                        logger.info("%s succeeded on retry %s/%s.", phase, attempt, retries)
                    return luma_np
                logger.warning(
                    "%s attempt %s/%s returned no luma result for gray %s.",
                    phase,
                    attempt,
                    retries,
                    self.current_gray_int,
                )
            except Exception as exc:
                logger.warning(
                    "%s attempt %s/%s failed for gray %s: %s",
                    phase,
                    attempt,
                    retries,
                    self.current_gray_int,
                    exc,
                )
                error_text = str(exc).lower()
                if "process fail" in error_text or "定位图" in str(exc):
                    logger.warning(
                        "%s encountered locator failure on attempt %s/%s; aborting remaining retries in this phase.",
                        phase,
                        attempt,
                        retries,
                    )
                    break

            if attempt < retries:
                time.sleep(retry_delay_s)

        return None

    def reset(
        self,
        target_gray: Optional[int] = None,
        initial_delta_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """重置环境（Episode 初始步骤）。

        将屏幕输出为没有进行任何补偿的纯净目标灰阶画面，拍摄得到第一帧亮度图以提取该灰阶的目标基准亮度。
        """
        if target_gray is not None:
            self.set_target_gray(target_gray)

        self.episode_count += 1
        self.step_count = 0
        logger.info("=== Episode %s reset at gray %s ===", self.episode_count, self.current_gray_int)

        # 初始灰度图填充为纯净目标灰度 (如 16)
        self.current_gray_map = torch.full(
            (1, 1, self.roi_h, self.roi_w),
            self.target_gray,
            device=device,
        )
        # 初始累计 Delta 置为全 0
        if initial_delta_map is not None:
            if initial_delta_map.shape != self.current_gray_map.shape:
                raise ValueError(
                    f"Initial delta shape mismatch: got {tuple(initial_delta_map.shape)}, "
                    f"expected {tuple(self.current_gray_map.shape)}"
                )
            self.current_gray_map = torch.clamp(self.current_gray_map + initial_delta_map.to(device), 0, 255)
        self.total_delta = self.current_gray_map - self.target_gray
        self.baseline_gray_map = self.current_gray_map.detach().clone()
        self.image_processor.reset_quantization_state(self.display_name)
        self.displayed_gray_map = None

        luma_np = self._capture_real_luma_with_retry(
            phase=f"Episode {self.episode_count} reset",
            delay_ms=1000,
        )
        if luma_np is None:
            logger.warning(
                "Episode %s reset could not capture initial luma; refreshing locator frames and retrying once more.",
                self.episode_count,
            )
            self.ensure_static_references(force_refresh=True)
            luma_np = self._capture_real_luma_with_retry(
                phase=f"Episode {self.episode_count} reset post-refresh",
                delay_ms=1800,
            )
        if luma_np is None:
            self.image_processor.dump_failure_artifacts(
                PathConfig.ACTIVE_SAVE_DIR,
                "reset_failure",
                gray=self.current_gray_int,
            )
            detail = self.image_processor.last_pre_demura_failure_reason
            if detail:
                raise RuntimeError(
                    f"Failed to capture initial luma image after retries at gray {self.current_gray_int}: {detail}"
                )
            raise RuntimeError(f"Failed to capture initial luma image after retries at gray {self.current_gray_int}.")

        # 存储初始状态与目标亮度基准
        luma_np = self._expand_luma_to_full_roi_if_needed(luma_np)
        self.current_luma_map = torch.from_numpy(luma_np).float().unsqueeze(0).unsqueeze(0).to(device)
        if self.image_processor.last_quantized_roi is not None:
            self.displayed_gray_map = (
                torch.from_numpy(self.image_processor.last_quantized_roi.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
        else:
            self.displayed_gray_map = torch.round(self.current_gray_map.detach()).to(device)
        if self.run_target_mean_nit is None or not self.freeze_target_per_run:
            self.run_target_mean_nit = float(
                calc_center_mean(self.current_luma_map, TrainConfig.TARGET_CENTER_SIZE).item()
            )
            self.run_target_luma_map = torch.full_like(self.current_luma_map, self.run_target_mean_nit).detach()
        self.target_mean_nit = self.run_target_mean_nit
        self.target_luma_map = self.run_target_luma_map
        initial_norm_std = self._update_baseline_std_metrics()

        # 保存初始文件以供追踪
        logger.info(
            "Episode %s baseline at gray %s: target_nit=%.4f real_std=%.4f norm_std=%.4f",
            self.episode_count,
            self.current_gray_int,
            self.target_mean_nit,
            float(self.initial_std),
            initial_norm_std,
        )
        return self._get_observation()

    def step(self, action_gray_diff: torch.Tensor):
        """执行一步 RL 动作更新。

        参数:
            action_gray_diff: Actor 输出的单步灰阶增量 map -> (1, 1, H, W)
        返回:
            next_observation, reward, info
        """
        self.step_count += 1

        # 1. 经过安全钳夹与累计防烧屏校验
        if self._train_roi is not None and action_gray_diff.shape[-2:] != (self.roi_h, self.roi_w):
            top, left, h, w = self._train_roi
            full_action = torch.zeros((1, 1, self.roi_h, self.roi_w), device=action_gray_diff.device, dtype=action_gray_diff.dtype)
            full_action[..., top:top + h, left:left + w] = action_gray_diff
            action_gray_diff = full_action
        action_gray_diff = self._safety_check(action_gray_diff)
        action_mean = float(torch.mean(action_gray_diff).item())
        action_abs_mean = float(torch.mean(torch.abs(action_gray_diff)).item())
        action_min = float(torch.min(action_gray_diff).item())
        action_max = float(torch.max(action_gray_diff).item())
        
        # 2. 将动作加到当前灰度图上并做 [0, 255] 像素限幅
        previous_gray_map = self.current_gray_map
        self.current_gray_map = torch.clamp(self.current_gray_map + action_gray_diff, 0, 255)
        effective_float_action = self.current_gray_map - previous_gray_map
        self.total_delta = self.current_gray_map - self.target_gray
        # 更新累计 Delta 变化量

        # 3. 硬件抓拍与前置 PreDemura 处理
        luma_np = self._capture_real_luma_with_retry(
            phase=f"Episode {self.episode_count} step {self.step_count}",
        )
        if luma_np is None:
            logger.error("Step %s capture failed after retries.", self.step_count)
            return self._get_observation(), 0.0, {"error": True, "gray": self.current_gray_int}

        luma_np = self._expand_luma_to_full_roi_if_needed(
            luma_np,
            fill_value=self.target_mean_nit,
        )
        self.current_luma_map = torch.from_numpy(luma_np).float().unsqueeze(0).unsqueeze(0).to(device)
        if self.image_processor.last_quantized_roi is not None:
            self.displayed_gray_map = (
                torch.from_numpy(self.image_processor.last_quantized_roi.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
        effective_action_tensor = None
        if self.image_processor.last_step_quantized_delta is not None:
            effective_action_tensor = (
                torch.from_numpy(self.image_processor.last_step_quantized_delta.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

        # 4. 亮度突变安全防御：如果当前读数均值相比最初设定值发生极端异常变化（如大于 50% 跌落或飙升），则触发警告
        current_mean = torch.mean(self.current_luma_map).item()

        # 5. 计算当前的奖励回报
        reward, reward_info = self._compute_reward()
        visual_std = float(reward_info["visual_std"])
        raw_std = float(reward_info["real_std"])
        std_ratio = (
            visual_std / (self.initial_std + 1e-6)
            if self.initial_std is not None
            else float("nan")
        )
        render_stats = dict(self.image_processor.last_render_stats)
        info = {
            "step": self.step_count,
            "gray": self.current_gray_int,
            "std": visual_std,
            "raw_std": raw_std,
            "std_ratio": std_ratio,
            "mean": current_mean,
            "action_mean": action_mean,
            "action_abs_mean": action_abs_mean,
            "action_min": action_min,
            "action_max": action_max,
            "effective_float_action_abs_mean": float(torch.mean(torch.abs(effective_float_action)).item()),
            "total_delta_abs_mean": float(torch.mean(torch.abs(self.total_delta)).item()),
            "reward_info": reward_info,
            "error": False,
            "capture_used_retry": self.last_capture_used_retry,
            "capture_retry_count": self.last_capture_retry_count,
        }
        if effective_action_tensor is not None:
            info["effective_action_tensor"] = effective_action_tensor.detach()
            info["effective_action_abs_mean"] = float(torch.mean(torch.abs(effective_action_tensor)).item())
        info.update(render_stats)
        if effective_action_tensor is not None:
            train_roi_effective = self._crop_to_train_roi(effective_action_tensor)
            reward_roi_effective = self._crop_to_reward_roi(effective_action_tensor)
            info["effective_action_train_roi_abs_mean"] = float(torch.mean(torch.abs(train_roi_effective)).item())
            info["quantized_step_train_roi_changed_ratio"] = float(
                torch.mean((train_roi_effective != 0).float()).item()
            )
            info["quantized_step_train_roi_delta_abs_mean"] = float(
                torch.mean(torch.abs(train_roi_effective)).item()
            )
            info["effective_action_reward_roi_abs_mean"] = float(torch.mean(torch.abs(reward_roi_effective)).item())
            info["quantized_step_reward_roi_changed_ratio"] = float(
                torch.mean((reward_roi_effective != 0).float()).item()
            )
            info["quantized_step_reward_roi_delta_abs_mean"] = float(
                torch.mean(torch.abs(reward_roi_effective)).item()
            )

        return self._get_observation(), reward, info

    def _safety_check(self, action: torch.Tensor) -> torch.Tensor:
        """硬件安全防护。

        1. 将单步变动范围严格 clip 在 [-MAX_ACTION_DELTA, MAX_ACTION_DELTA] (默认 +-3.0 灰阶)。
        2. 判断叠加此动作后，是否会突破单像素累计的最大变化量限制 MAX_TOTAL_DELTA (默认 +-30 灰阶)。
           若有任一位置越界，强行将该超限位置的单步动位置 0 (禁止继续变化)，其余位置正常响应。
        """
        action = torch.clamp(action, -SafetyConfig.MAX_ACTION_DELTA, SafetyConfig.MAX_ACTION_DELTA)

        projected_total = self.total_delta + action
        over_limit = torch.abs(projected_total) > SafetyConfig.MAX_TOTAL_DELTA
        if over_limit.any():
            logger.warning("Cumulative gray delta exceeded the safety limit; clipping action.")
            # 将超限像素处的步进动位置为 0
            action = torch.where(over_limit, torch.zeros_like(action), action)
        return action

    def _get_observation(self) -> torch.Tensor:
        """构建强化学习环境的 State 输入（双通道 Tensor）。

        通道 0: 相对亮度误差 (Relative Error) = (当前亮度 - 目标均值) / 目标均值
        通道 1: 归一化当前绝对灰度 (Norm Gray) = 当前灰度地图 / 255.0
        """
        target_map = self.target_luma_map if self.target_luma_map is not None else self.current_luma_map
        rel_error = (self.current_luma_map - target_map) / (target_map + 1e-6)
        gray_source = self.displayed_gray_map if self.displayed_gray_map is not None else self.current_gray_map
        norm_gray = gray_source / 255.0
        rel_error = self._crop_to_train_roi(rel_error)
        norm_gray = self._crop_to_train_roi(norm_gray)
        state = torch.cat([rel_error, norm_gray], dim=1).detach()
        return state

    def _compute_reward(self, gray_weight: float = 1.0):
        """计算当前的加权奖励函数。

        奖励设计思想为：均匀的亮度场应该让各处相对亮度均对齐至 1.0 (即相对误差为 0)，
        并且全局标准差 (STD) 极小、邻域梯度变化极平滑。均取负值，值越接近 0 奖励越高。
        """
        # 归一化当前亮度：以目标亮度 nit 作为 1.0 准则
        target_map = self.target_luma_map if self.target_luma_map is not None else self.current_luma_map
        # Crop to reward ROI so action support can be larger than the optimized area.
        luma = self._crop_to_reward_roi(self.current_luma_map)
        tgt = self._crop_to_reward_roi(target_map)
        norm_luma = luma / (tgt + 1e-6)
        diff_raw = norm_luma - 1.0
        real_std = torch.std(luma)

        total, reward_info = compute_uniformity_reward_from_rel_error(diff_raw)
        reward_info["real_std"] = float(real_std.item())
        return total.item(), reward_info


        # 视觉导向 reward：强约束整体亮度偏移，优先压低低频 mura，再轻量抑制高频纹理

    def _save_step_data(self, tag: str, luma_np: np.ndarray) -> None:
        """将当前的亮度矩阵和补偿 Delta 灰度图保存为 32位 浮点数 TIFF 格式，以便追溯。"""
        save_dir = getattr(PathConfig, "ACTIVE_SAVE_DIR", PathConfig.SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)
        gray_np = self.current_gray_map.detach().cpu().numpy()[0, 0]
        delta_np = gray_np - self.target_gray

        tiff.imwrite(
            os.path.join(save_dir, f"{tag}_luma.tiff"),
            luma_np.astype(np.float32),
        )
        tiff.imwrite(
            os.path.join(save_dir, f"{tag}_delta_gray.tiff"),
            delta_np.astype(np.float32),
        )
        quantized_roi = self.image_processor.last_quantized_roi
        if quantized_roi is not None:
            quantized_delta_np = quantized_roi.astype(np.float32) - self.target_gray
            tiff.imwrite(
                os.path.join(save_dir, f"{tag}_quantized_delta_gray.tiff"),
                quantized_delta_np.astype(np.float32),
            )

    def get_best_result(self) -> np.ndarray:
        """获取当前训练后沉淀出来的 Delta 灰度补偿表矩阵。"""
        gray_np = self.current_gray_map.detach().cpu().numpy()[0, 0]
        return gray_np - self.target_gray

    def get_quantized_result(self) -> Optional[np.ndarray]:
        if self.image_processor.last_quantized_roi is None:
            return None
        return self.image_processor.last_quantized_roi.astype(np.float32) - self.target_gray

    def get_current_gray_snapshot(self) -> np.ndarray:
        return self.current_gray_map.detach().cpu().numpy()[0, 0].astype(np.float32, copy=True)

    def get_displayed_gray_snapshot(self) -> Optional[np.ndarray]:
        if self.displayed_gray_map is None:
            return None
        return self.displayed_gray_map.detach().cpu().numpy()[0, 0].astype(np.float32, copy=True)
