# -*- coding: utf-8 -*-
"""实机 Demura 强化学习训练的中央配置模块。

包含硬件面板参数、多灰阶曝光与显示参数、通信协议、路径管理、训练超参以及安全确认逻辑。
"""

import os
from typing import Dict, Iterable, List, Tuple


class PanelConfig:
    """显示屏面板与 ROI (Region of Interest) 感兴趣区域参数配置。"""
    # 面板物理分辨率
    PANEL_HEIGHT = 2652
    PANEL_WIDTH = 1200
    # 相机图像中对应的 ROI 起始坐标与宽高（用于 Demura 补偿的目标区域）
    ROI_START_X = 100
    ROI_START_Y = 200
    ROI_HEIGHT = 2000
    ROI_WIDTH = 1000
    # 当前产品型号名称
    PRODUCT_NAME = "HD033"


class GrayConfig:
    """多灰阶与采集参数配置。"""
    # 默认单灰阶训练的目标灰阶值
    DEFAULT_SINGLE_GRAY = 16
    # 联合训练的多灰阶列表
    DEFAULT_MULTI_GRAYS = [16] # [128, 64, 32, 16, 8, 5, 3]
    # 定位和亮度参考用的静态基准灰阶（W252, W253 分别对应极高灰阶，用于画面配准与定位）
    STATIC_REFERENCE_GRAYS = [252, 253]

    # 各灰阶拍照时的相机采集参数：{ 灰阶值: (曝光时间 exposure_us, 增益 gain_db) }
    # 曝光时间单位为微秒，低灰阶（如3、8）亮度极低，因此需要很长的曝光时间（15000us）来捕获足够的进光量；
    # 高灰阶（如64、128）亮度极高，曝光时间需要大幅缩短（300us）以防相机过曝饱和。
    CAPTURE_PARAMS: Dict[int, Tuple[float, float]] = {
        3: (15000.0, 10.0),
        5: (6000.0, 10.0),
        8: (15000.0, 1.0),
        16: (2000.0, 3.0),
        32: (1500.0, 1.0),
        64: (300.0, 1.0),
        128: (300.0, 1.0),
        252: (200.0, 1.0),
        253: (200.0, 1.0),
    }

    # 对应灰阶在 C# 客户端 Pattern 发生器中的画面索引值 (Pattern Index)
    DISPLAY_PATTERN_INDEX: Dict[int, int] = {
        16: 204,
        32: 205,
        64: 206,
        128: 207,
        8: 208,
        5: 209,
        3: 210,
        252: 252,
        253: 253,
    }


class CommConfig:
    """与 C# 自动化测试软件的 TCP 通信配置。"""
    # TCP 服务器绑定的本地 IP 与端口（Python 端作为 Server 监听，C# 作为 Client 连接）
    SERVER_IP = "localhost"
    SERVER_PORT = 12345
    # TCP 通信超时时间（单位：秒）
    SOCKET_TIMEOUT = int(os.environ.get("DEMURA_SOCKET_TIMEOUT", "120"))

    # 遗留/传统通信协议命令及标识符
    LEGACY_START_CMD = "START"
    LEGACY_END_CMD = "END"
    LEGACY_ERROR_PREFIX = "ERROR"


class PathConfig:
    """本地文件系统路径配置。"""
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))

    # 临时传输 BMP 的目录（用于向屏幕发送包含补偿的图案）
    TRANSFER_BMP_DIR = os.path.join(CUR_DIR, "TransferBmp")
    # 静态参考图目录（如 Static W252/W253 的定位图）
    STATIC_BMP_DIR = os.path.join(CUR_DIR, "StaticPatterns")
    # 相机抓拍到的原始图像保存目录
    CAMERA_IMAGE_DIR = os.path.join(CUR_DIR, "images")
    # 外部算法（PreDemura.exe）处理后输出的 TIFF 结果图目录
    RESULT_DIR = os.path.join(CUR_DIR, "ResultData")
    # 训练日志记录目录
    LOG_DIR = os.path.join(CUR_DIR, "TrainLogs")
    # 强化学习微调过程中，中间产物与最终模型、TIFF 补偿表的保存目录
    SAVE_DIR = os.path.join(CUR_DIR, "RealWorld_Train")
    ACTIVE_SAVE_DIR = SAVE_DIR

    # 图像对齐与 Demura 前置处理的可执行文件路径
    PRE_DEMURA_EXE = os.path.join(CUR_DIR, "PreDemura.exe")
    # 预训练 Actor 模型权重路径
    PRETRAINED_MODEL = os.path.join(
        CUR_DIR, "TrainData", "Demura_Final_Opt_Gaussian", "best_actor.pth"
    )

    # 离线蒸馏所使用的仿真面板映射数据目录列表
    DISTILL_DATA_DIRS = [
        os.path.join(CUR_DIR, "data", "screen5"),
        os.path.join(CUR_DIR, "data", "screen4"),
        os.path.join(CUR_DIR, "data", "screen3"),
        os.path.join(CUR_DIR, "data", "screen1"),
    ]


class TrainConfig:
    """强化学习与知识蒸馏训练超参数配置。"""
    # ---- 基础训练超参 ----
    PHYSICAL_GAMMA = 2.2        # 屏幕物理 Gamma 值（用于自适应反函数物理先验动作计算）
    PRIOR_LOWPASS_KERNEL = 31
    PRIOR_LOWPASS_PASSES = 2
    TARGET_SURFACE_LOWPASS_KERNEL = 31
    TARGET_SURFACE_LOWPASS_PASSES = 2
    TARGET_CENTER_SIZE = 200    # 纯色首帧中心区域均值窗口
    PRIOR_LOWPASS_KERNEL = 31   # 物理先验只修正低频 mura，先对相对亮度误差做低通
    PRIOR_LOWPASS_PASSES = 2
    MAX_EPISODES = 50          # 最大训练 Episode 数
    STEPS_PER_EPISODE = 10      # 每个 Episode 的交互步数（单步对应一次硬件拍照及反馈）
    BATCH_SIZE = 4             # 神经网络训练的批大小
    LEARN_START_SIZE = 16      # 实机上至少累计多少条样本后再开始 DDPG 更新，避免极小样本把预训练模型带偏
    BUFFER_CAPACITY = 2000      # 经验回放区（Replay Buffer）的最大容量

    ACTOR_LR = 5e-6             # Actor (策略网络) 的学习率
    CRITIC_LR = 5e-5            # Critic (估值网络) 的学习率
    GAMMA = 0.9                 # 强化学习折扣因子
    TAU = 0.01                  # 软更新 (Soft Update) 的混合系数
    NOISE_SCALE_INIT = 0.05     # 探索噪声（动作扰动）的初始标准差
    NOISE_DECAY = 0.99          # 探索噪声的每 episode 衰减率
    PRETRAINED_NOISE_SCALE_FACTOR = 0.1  # 载入预训练权重后，将实机探索噪声默认缩小到 10%
    COLD_START_NOISE_SCALE_INIT = 0.5    # 从零开始实机训练时，使用更强的初始探索幅度
    COLD_START_NOISE_DECAY = 0.995       # 从零开始实机训练时，探索衰减更慢

    MIN_ACTION_ABS_MEAN = 0.08
    MAX_ACTION_SCALE_BOOST = 12.0
    RESIDUAL_ACTION_CLIP = 0.50
    BASELINE_DEVIATION_W = 0.0
    SKIP_REPLAY_STD_RATIO = 1.08
    DISCARD_INITIAL_EPISODES = 1
    LR_STEP_SIZE = 20           # 学习率衰减周期（每 20 episodes 衰减一次）
    LR_GAMMA = 0.5              # 学习率衰减率（减半）
    MAX_STD_RATIO = 1.2         # 相对 reset 基线的 std 恶化阈值，超过后提前结束当前 episode
    RANDOM_WARMUP_EPISODES = 2  # 从零开始时前若干个 episode 使用随机平滑动作收集经验
    POLICY_ACTION_LOWPASS_KERNEL = 31  # learned actor 动作在下发到实机前的低通平滑核
    POLICY_ACTION_LOWPASS_PASSES = 2   # learned actor 动作低通平滑的重复次数

    # ---- 多灰阶训练 ----
    # 训练时使用的灰阶列表，每个 episode 会随机从中挑选一个灰阶值作为控制目标
    ACTOR_UPDATE_EVERY = 4      # Actor delayed update cadence on real hardware
    ACTOR_LEARN_START_UPDATES = 20   # Critic-only warmup updates after random warmup episodes
    TRAIN_GRAYS = [128, 64, 32, 16, 8, 5, 3]

    # ---- 灰阶加权 Reward 机制 ----
    # 动机：低灰阶（如 G3、G5）本身的发光亮度极低（约几 nit 或更小），因此计算出的绝对亮度误差/Loss 非常微弱；
    # 而高灰阶（如 G128）亮度极高，相同的相对误差会产生巨大的 Loss 绝对值。
    # 为了防止多灰阶联合优化时，大灰阶的梯度完全掩盖小灰阶的梯度，需要设计加权公式。
    # 权重计算公式：gray_weight = GRAY_WEIGHT_BASE + gray / GRAY_WEIGHT_SCALE
    # 这样低灰阶的 loss 会经过适当的放大，以平衡其梯度更新信号。
    GRAY_WEIGHT_BASE = 1.0
    GRAY_WEIGHT_SCALE = 50.0

    # 奖励函数中，各项损失分量的基础权重比
    REWARD_MEAN_W = 220.0       # 全局平均亮度偏移权重，防止通过整体变亮/变暗“伪改善”
    REWARD_LOWPASS_W = 180.0    # 低频视觉 mura 权重，优先压制人眼最敏感的大块/条纹不均
    REWARD_PROFILE_W = 140.0    # 行列轮廓起伏权重，专门约束纵向/横向 banding 与云斑
    REWARD_GRAD_W = 25.0        # 高频纹理惩罚权重，只做轻约束，避免过度打压细颗粒
    VISUAL_LOWPASS_KERNEL = 31  # 低通平滑核尺寸，越大越偏向观察低频视觉不均

    # ---- 蒸馏参数 ----
    # 蒸馏初始化模式："single" = 单教师单灰阶；"multi" = 多教师多灰阶联合蒸馏
    DISTILL_MODE = "single"

    DISTILL_EPOCHS = 500        # 蒸馏迭代轮数
    DISTILL_LR = 1e-4           # 蒸馏优化的学习率
    DISTILL_STEPS = 8           # 教师模型在仿真环境下 rollout 迭代的步数（累积以生成最终的总动作 GT）
    DISTILL_ROI_SIZE = 200      # 蒸馏时使用的 ROI 边长（200x200）
    DISTILL_BATCH_SIZE = 1      # 蒸馏批大小
    DISTILL_SAVE_NAME = "distilled_actor_init.pth" # 蒸馏后的 Actor 权重文件名

    # 单教师蒸馏下的随机 ROI 裁剪数据增强（True 时，每个 epoch 裁剪不同位置以提升学生网络的空间泛化力）
    DISTILL_RANDOM_CROP = True
    DISTILL_SINGLE_GRAY = 16    # 单教师蒸馏的目标灰阶值


class SafetyConfig:
    """硬件安全防护参数配置。"""
    # 单步内，任意像素点允许的最大灰阶调整量增量（防止突变过大损坏屏幕或相机捕获异常）
    MAX_ACTION_DELTA = 3.0
    # 在整个 Episode 中，任意像素点允许的最大累计灰阶调整量（防止过度补偿导致局部烧屏或完全失真）
    MAX_TOTAL_DELTA = 30.0
    # 亮度异常检测阈值（若当前捕获亮度的均值与目标亮度偏差超过此比例，触发警告，防相机被遮挡或故障）
    LUMA_ANOMALY_RATIO = 0.5
    # 是否开启人工确认模式
    ENABLE_HUMAN_CONFIRM = True
    # 训练开始前几个 Episode 结束后，暂停并要求人工确认是否继续
    HUMAN_CONFIRM_EPISODES = 3


def parse_gray_list(gray_text: str) -> List[int]:
    """解析由逗号分隔的灰阶文本字符串（如 "128,64,32" -> [128, 64, 32]）。"""
    values = [item.strip() for item in gray_text.split(",") if item.strip()]
    grays = [int(item) for item in values]
    if not grays:
        raise ValueError("At least one gray level is required.")
    return grays


def unique_gray_list(grays: Iterable[int]) -> List[int]:
    """对灰阶列表进行去重，并保持原有顺序。"""
    ordered: List[int] = []
    seen = set()
    for gray in grays:
        value = int(gray)
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def gray_to_display_name(gray: int) -> str:
    """生成用于显示的灰阶命名格式，如 16 -> "W16"。"""
    return f"W{int(gray)}"


def gray_to_capture_name(gray: int) -> str:
    """生成用于硬件抓拍图像保存时的灰阶命名格式，如 16 -> "W016" (补齐三位)。"""
    return f"W{int(gray):03d}"


def gray_to_result_name(gray: int) -> str:
    """生成外部 PreDemura 输出 TIFF 结果时的灰阶命名格式，如 16 -> "W16"。"""
    return f"W{int(gray)}"


def get_capture_params(gray: int) -> Tuple[float, float]:
    """获取指定灰阶拍照所需的曝光与增益参数，若无配置则使用默认单灰阶曝光。"""
    value = int(gray)
    return GrayConfig.CAPTURE_PARAMS.get(
        value,
        GrayConfig.CAPTURE_PARAMS[GrayConfig.DEFAULT_SINGLE_GRAY],
    )


def get_pattern_index(gray: int) -> int:
    """获取指定灰阶在 C# 客户端的 Pattern Index，若无配置抛出异常。"""
    value = int(gray)
    if value not in GrayConfig.DISPLAY_PATTERN_INDEX:
        raise KeyError(f"No pattern index configured for gray {value}.")
    return GrayConfig.DISPLAY_PATTERN_INDEX[value]


def get_capture_mim_path(gray: int) -> str:
    """获取指定灰阶抓拍的原始图像保存路径。"""
    return os.path.join(PathConfig.CAMERA_IMAGE_DIR, f"{gray_to_capture_name(gray)}_0.MIM")


def get_result_tiff_candidates(gray: int) -> List[str]:
    """获取外部算法处理后，可能生成的 TIFF 结果文件名候选列表（兼容两种命名习惯）。"""
    value = int(gray)
    return [
        os.path.join(PathConfig.RESULT_DIR, f"{gray_to_result_name(value)}.tiff"),
        os.path.join(PathConfig.RESULT_DIR, f"{gray_to_capture_name(value)}.tiff"),
    ]
