
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# 模型与窗口配置：分类模型是本项目训练好的 9 类模型，proposal 模型用于可选的通用目标预检测
MODEL_PATH = Path("runs/classify/target_item_classes/weights/best.pt")  # 项目训练好的分类模型权重路径
PROPOSAL_MODEL_PATH = Path("yolov8n.pt")  # 通用 YOLO 检测模型路径，仅在不强制卡片检测时使用
WINDOW_NAME = "YOLO 9 Item Detection"  # OpenCV 显示窗口标题
CHINESE_FONT_SIZE = 24  # 画面中文标签字号

# 中文字体候选路径。Windows 本机优先，Ubuntu/Raspberry Pi 可安装 Noto CJK 或文泉驿字体。
CHINESE_FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
]

# 摄像头与显示配置
CAMERA_INDEX = 0  # 摄像头编号；如果打不开摄像头，可改为 1、2 等其他设备号
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_WIDTH = 960
DISPLAY_WIDTH = 960  # 显示画面的最大宽度；检测仍使用高分辨率原帧

# 候选区域、置信度、NMS 等检测阈值。调参时优先关注这些常量
MAX_CANDIDATES = 14  # 普通轮廓兜底检测最多保留的候选框数量
MAX_CARD_CANDIDATES = 2  # 单张卡片外再留少量候选余量
MAX_DETECTIONS = 1  # 比赛每次只输出当前展示的一种物品
PROPOSAL_CONF_THRESHOLD = 0.28  # 通用 YOLO proposal 模型的基础置信度阈值
PROPOSAL_IOU_THRESHOLD = 0.45  # 通用 YOLO proposal 模型内部 NMS 的 IoU 阈值
CONF_THRESHOLD = 0.78  # 分类模型默认置信度阈值
MARGIN_THRESHOLD = 0.12  # top1 与 top2 分类分数最小差值，越大越保守
NMS_IOU_THRESHOLD = 0.35  # 候选框去重的 IoU 阈值，低于该值才认为不是重复框
DETECTION_INTERVAL = 6  # 每隔多少帧运行一次完整检测；中间帧使用追踪结果
TRACK_MIN_HITS = 3  # 轨迹至少连续命中多少次后才显示，防止一帧误检
TRACK_MAX_MISSED = 8  # 轨迹最多允许连续丢失多少帧，超过后清除
BOX_SMOOTHING = 0.90  # 检测框位置平滑系数；越接近 1，框移动越慢越稳
CONFIDENCE_SMOOTHING = 0.82  # 置信度平滑系数；越接近 1，分数变化越慢
MOTION_THRESHOLD = 18  # 帧差二值化阈值，用于生成运动区域掩码
NEW_TRACK_MIN_MOTION = 0.006  # 非可信检测创建新轨迹所需的最小运动比例
ENABLE_CONTOUR_FALLBACK = True  # 是否启用基于轮廓的候选区域兜底方案
FALLBACK_CONF_THRESHOLD = 0.90  # 兜底分类结果需要达到的最低置信度
FALLBACK_MIN_MOTION = 0.015  # 兜底分类结果需要达到的最小运动比例
FACE_OVERLAP_THRESHOLD = 0.12  # 候选框与人脸区域重叠超过该比例时过滤
ENABLE_FACE_FILTER = False  # 比赛场景只检测卡片，默认关闭人脸过滤以降低卡顿
CARD_CONF_THRESHOLD = 0.78  # 卡片裁剪图分类的最低置信度
TRACK_VISIBLE_CONFIDENCE_RATIO = 0.90  # 追踪显示阈值相对类别阈值的比例
SINGLE_TARGET_MODE = True  # 比赛按顺序逐张识别，每次只维护一个主目标
LABEL_SWITCH_HITS = 2  # 标签连续变化多少次后才真正切换轨迹标签

# 单目标追踪参数：用于抑制框突然跳动、短暂丢帧和标签抖动
PRIMARY_JUMP_IOU = 0.06  # 主目标新旧框 IoU 低于该值时，可能被判定为跳变
PRIMARY_JUMP_CENTER_RATIO = 0.55  # 主目标中心距离超过该比例时，可能被判定为跳变
PRIMARY_CONF_DECAY = 0.97  # 主目标丢帧时置信度每帧衰减系数
TERMINAL_REPEAT_INTERVAL = 5.0  # 终端相同识别结果的重复打印间隔，单位秒

# 卡片/包裹区域检测参数：先找疑似卡片，再对卡片裁剪图做分类
REQUIRE_CARD_FOR_DETECTION = True  # 是否必须先检测到卡片/包裹区域才允许输出结果
CARD_MISSING_CLEAR_FRAMES = 2  # 连续多少个检测周期找不到卡片后清空已有轨迹
DEBUG_PRINT_INTERVAL = 1.0  # 调试输出间隔，单位秒；当前逻辑预留该参数
CARD_MIN_BRIGHT_RATIO = 0.46  # 卡片内部亮色像素最低比例，用于过滤暗色区域
CARD_MIN_EDGE_DENSITY = 0.002  # 卡片内部最低边缘密度，太低可能是空白背景
CARD_MAX_EDGE_DENSITY = 0.18  # 卡片内部最高边缘密度，太高可能是杂乱背景
CARD_MIN_DARK_RATIO = 0.001  # 卡片内部暗色内容最低比例，用于确认有图案/文字
CARD_MAX_DARK_RATIO = 0.55  # 卡片内部暗色内容最高比例，过高可能不是白色卡片
CARD_MIN_RECTANGULARITY = 0.62  # 亮色候选轮廓的最低矩形度
ENABLE_ORIENTATION_TTA = False  # 是否对裁剪图进行旋转测试增强，提升方向鲁棒性
ORIENTATION_TTA_MARGIN_WEIGHT = 0.25  # 旋转增强结果排序时，分类边际分数的权重
CARD_MIN_BORDER_SIDES = 3  # 至少多少条边需要有明显边框特征
CARD_BORDER_STRIP_RATIO = 0.06  # 检查边框时，四边取样条带占边长的比例
CARD_BORDER_DARK_THRESHOLD = 0.025  # 单侧边框暗色像素比例阈值
CARD_BORDER_EDGE_THRESHOLD = 0.020  # 单侧边框边缘密度阈值
CARD_MIN_BORDER_SCORE = 1.20  # 卡片边框综合评分最低要求
CARD_MIN_AREA_RATIO = 0.002  # 适配 20 cm 距离下的小卡片
CARD_MAX_AREA_RATIO = 0.30
CARD_MIN_SIDE = 36
CARD_WARP_PADDING = 0.06
CARD_CONTAINER_MIN_CHILDREN = 4
CARD_CONTAINER_MAX_CHILD_AREA_RATIO = 0.35
CARD_NESTED_MIN_PARENT_RATIO = 1.60
CARD_NESTED_MAX_PARENT_RATIO = 8.00
EDGE_CARD_MIN_AREA_RATIO = 0.002  # 边缘卡片候选占整帧面积的最小比例
EDGE_CARD_MAX_AREA_RATIO = 0.30  # 过滤整张试题纸等超大矩形
EDGE_CARD_MIN_SIDE = 36  # 20 cm 场景仍需保留的小卡片最短边
EDGE_CARD_MAX_ASPECT = 1.45  # 边缘卡片候选最大长宽比，越小越接近正方形
EDGE_CARD_MIN_RECTANGULARITY = 0.45  # 边缘卡片候选最低矩形度

# 本项目训练集支持的目标标签
TARGET_LABELS = {
    "air_conditioner",
    "apple_fruit",
    "banana_fruit",
    "clothes",
    "orange_fruit",
    "refrigerator_appliance",
    "television_set",
    "tissue_paper",
    "toothbrush",
}

# 将 COCO 通用检测类别映射到本项目的分类标签
COCO_TO_TARGET = {
    "apple": "apple_fruit",
    "banana": "banana_fruit",
    "orange": "orange_fruit",
    "refrigerator": "refrigerator_appliance",
    "toothbrush": "toothbrush",
    "tv": "television_set",
}

# 通用检测模型各类别的独立阈值，避免大家电等类别过度触发
PROPOSAL_CLASS_THRESHOLDS = {
    "apple": 0.35,  # 通用模型检测苹果的最低置信度
    "banana": 0.35,  # 通用模型检测香蕉的最低置信度
    "orange": 0.35,  # 通用模型检测橙子的最低置信度
    "refrigerator": 0.65,  # 通用模型检测冰箱的最低置信度；设高一些减少误检
    "toothbrush": 0.35,  # 通用模型检测牙刷的最低置信度
    "tv": 0.55,  # 通用模型检测电视机的最低置信度
}

FALLBACK_ONLY_LABELS = TARGET_LABELS  # 兜底分类流程允许识别的标签集合

# 分类模型各类别的额外阈值；容易误检的类别可以设得更高
CLASS_CONF_THRESHOLDS = {
    "air_conditioner": 0.78,  # 空调分类最低置信度
    "clothes": 0.78,  # 衣服分类最低置信度
    "refrigerator_appliance": 0.84,  # 冰箱分类最低置信度；较易混淆时提高
    "television_set": 0.84,  # 电视机分类最低置信度；较易混淆时提高
    "tissue_paper": 0.82,  # 卫生纸分类最低置信度
}

# 终端输出使用的中文类别名称
CATEGORY_NAMES_CN = {
    "daily_items": "日用品",
    "fruits": "水果",
    "home_appliances": "家电",
}

# 物品到大类的映射，用于输出“属于哪一类”
ITEM_CATEGORIES = {
    "toothbrush": "daily_items",
    "tissue_paper": "daily_items",
    "clothes": "daily_items",
    "banana_fruit": "fruits",
    "apple_fruit": "fruits",
    "orange_fruit": "fruits",
    "television_set": "home_appliances",
    "refrigerator_appliance": "home_appliances",
    "air_conditioner": "home_appliances",
}

# 终端输出使用的中文物品名称
ITEM_NAMES_CN = {
    "toothbrush": "牙刷",
    "tissue_paper": "卫生纸",
    "clothes": "衣服",
    "banana_fruit": "香蕉",
    "apple_fruit": "苹果",
    "orange_fruit": "橙子",
    "television_set": "电视机",
    "refrigerator_appliance": "冰箱",
    "air_conditioner": "空调",
}

# 不同物品在画面中的框线颜色，OpenCV 使用 BGR 顺序
BOX_COLORS = {
    "toothbrush": (255, 170, 0),
    "tissue_paper": (0, 255, 255),
    "clothes": (180, 130, 255),
    "banana_fruit": (0, 220, 255),
    "apple_fruit": (0, 0, 255),
    "orange_fruit": (0, 150, 255),
    "television_set": (255, 80, 0),
    "refrigerator_appliance": (180, 255, 80),
    "air_conditioner": (255, 220, 120),
}


@dataclass
class Detection:
    # 单帧检测结果，box 坐标格式为 (x1, y1, x2, y2)
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    margin: float = 0.0
    motion: float = 1.0
    trusted: bool = False


@dataclass
class Track:
    # 跨帧追踪状态，用于平滑框位置并减少识别结果闪烁
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    hits: int = 1
    missed: int = 0
    candidate_label: str = ""
    candidate_hits: int = 0


def load_face_detectors() -> list[cv2.CascadeClassifier]:
    # 加载 OpenCV 自带的人脸检测器，后续用来排除人脸区域
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    detectors: list[cv2.CascadeClassifier] = []
    for cascade_name in cascade_names:
        detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / cascade_name))
        if not detector.empty():
            detectors.append(detector)
    return detectors


def resize_for_display(frame: np.ndarray, width: int) -> np.ndarray:
    # 保持宽高比缩放画面，避免窗口过大影响显示和处理速度
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    scale = width / current_width
    return cv2.resize(frame, (width, int(height * scale)), interpolation=cv2.INTER_AREA)


def configure_camera(camera: cv2.VideoCapture) -> None:
    # 高分辨率采集是 20 cm 小目标识别的前提。
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}", flush=True)
    if actual_width < 1280 or actual_height < 720:
        print(
            "Warning: use at least 1280x720; 1920x1080 is recommended at 20 cm.",
            flush=True,
        )


def expand_box(box: tuple[int, int, int, int], shape: tuple[int, int], ratio: float = 0.12) -> tuple[int, int, int, int]:
    # 适当扩大候选框，给分类模型保留物体边缘上下文
    x, y, w, h = box
    frame_h, frame_w = shape
    pad_x = int(w * ratio)
    pad_y = int(h * ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w - 1, x + w + pad_x)
    y2 = min(frame_h - 1, y + h + pad_y)
    return x1, y1, x2, y2


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    # 计算两个框的 IoU，用于 NMS 和追踪匹配
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    union_area = box_area(a) + box_area(b) - inter_area
    return inter_area / union_area if union_area else 0.0


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    # 用框对角线归一化中心点距离，便于判断目标是否突然跳动
    ax, ay = box_center(a)
    bx, by = box_center(b)
    distance = float(np.hypot(ax - bx, ay - by))
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    reference = max(
        float(np.hypot(ax2 - ax1, ay2 - ay1)),
        float(np.hypot(bx2 - bx1, by2 - by1)),
        1.0,
    )
    return distance / reference


def overlap_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    # 计算 inner 有多少比例落在 outer 内，用于过滤人脸重叠区域
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    inter_area = box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    area = box_area(inner)
    return inter_area / area if area else 0.0


def detect_faces(
    frame: np.ndarray,
    detectors: list[cv2.CascadeClassifier],
) -> list[tuple[int, int, int, int]]:
    # 多个人脸检测器的结果合并后做一次 NMS，减少重复人脸框
    if not detectors:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_boxes: list[tuple[int, int, int, int]] = []
    for detector in detectors:
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_boxes.extend((int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces)
    return nms_boxes(face_boxes, 0.30)


def overlaps_face(box: tuple[int, int, int, int], face_boxes: list[tuple[int, int, int, int]]) -> bool:
    # 只要目标框和任一人脸框明显重叠，就认为该候选不可靠
    return any(
        overlap_ratio(face_box, box) > FACE_OVERLAP_THRESHOLD
        or overlap_ratio(box, face_box) > FACE_OVERLAP_THRESHOLD
        for face_box in face_boxes
    )


def nms_boxes(boxes: list[tuple[int, int, int, int]], threshold: float) -> list[tuple[int, int, int, int]]:
    # 非极大值抑制：保留面积更大的框，去掉高度重叠的重复框
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=box_area, reverse=True):
        if all(box_iou(box, old_box) < threshold for old_box in kept):
            kept.append(box)
    return kept


def remove_container_candidates(
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]],
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    # 整张纸可能也是一个矩形；若内部包含多张小卡片，则不送入分类模型。
    filtered: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    for candidate in candidates:
        _, box, _ = candidate
        area = box_area(box)
        child_count = sum(
            1
            for _, other_box, _ in candidates
            if other_box != box
            and box_area(other_box) < area * CARD_CONTAINER_MAX_CHILD_AREA_RATIO
            and overlap_ratio(other_box, box) > 0.88
        )
        if child_count < CARD_CONTAINER_MIN_CHILDREN:
            filtered.append(candidate)

    outer_candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    for candidate in filtered:
        _, box, _ = candidate
        area = box_area(box)
        is_nested = any(
            box != parent_box
            and area * CARD_NESTED_MIN_PARENT_RATIO < box_area(parent_box)
            <= area * CARD_NESTED_MAX_PARENT_RATIO
            and overlap_ratio(box, parent_box) > 0.88
            for _, parent_box, _ in filtered
        )
        if not is_nested:
            outer_candidates.append(candidate)
    return outer_candidates


def order_points(points: np.ndarray) -> np.ndarray:
    # 将四个点固定排序为左上、右上、右下、左下，便于透视矫正
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def pad_card_points(points: np.ndarray, ratio: float = CARD_WARP_PADDING) -> np.ndarray:
    center = points.mean(axis=0)
    return (points - center) * (1.0 + ratio) + center


def warp_card(frame: np.ndarray, points: np.ndarray, size: int = 320) -> np.ndarray:
    # 把倾斜的卡片区域拉正为正方形裁剪图，再送入分类模型
    source = order_points(points)
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 242),
    )


def card_crop_score(crop: np.ndarray) -> float:
    # 根据亮度、暗色内容和边缘密度评估裁剪图是否像一张有效卡片
    height, width = crop.shape[:2]
    inset_x = max(2, int(width * 0.06))
    inset_y = max(2, int(height * 0.06))
    inner = crop[inset_y : height - inset_y, inset_x : width - inset_x]
    if inner.size == 0:
        return 0.0

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)

    bright_ratio = float(np.mean((hsv[:, :, 2] > 135) & (hsv[:, :, 1] < 135)))
    dark_ratio = float(np.mean(gray < 218))
    edges = cv2.Canny(gray, 35, 115)
    edge_density = float(cv2.countNonZero(edges)) / float(edges.size)

    if bright_ratio < CARD_MIN_BRIGHT_RATIO:
        return 0.0
    if dark_ratio < CARD_MIN_DARK_RATIO or dark_ratio > CARD_MAX_DARK_RATIO:
        return 0.0
    if edge_density < CARD_MIN_EDGE_DENSITY or edge_density > CARD_MAX_EDGE_DENSITY:
        return 0.0

    return bright_ratio * 1.4 + min(dark_ratio * 1.2, 0.45) + min(edge_density * 9.0, 0.45)


def card_border_score(crop: np.ndarray) -> float:
    # 检查卡片四边是否有足够的边缘/暗色信息，过滤纯白背景误检
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    strip = max(6, int(min(height, width) * CARD_BORDER_STRIP_RATIO))
    side_regions = [
        gray[:strip, :],
        gray[height - strip :, :],
        gray[:, :strip],
        gray[:, width - strip :],
    ]

    strong_sides = 0
    dark_total = 0.0
    edge_total = 0.0
    for region in side_regions:
        dark_ratio = float(np.mean(region < 130))
        edges = cv2.Canny(region, 35, 110)
        edge_density = float(cv2.countNonZero(edges)) / float(edges.size)
        dark_total += dark_ratio
        edge_total += edge_density
        if dark_ratio >= CARD_BORDER_DARK_THRESHOLD or edge_density >= CARD_BORDER_EDGE_THRESHOLD:
            strong_sides += 1

    if strong_sides < CARD_MIN_BORDER_SIDES:
        return 0.0
    score = strong_sides / 4.0 + dark_total / 4.0 + min(edge_total, 0.45)
    return score if score >= CARD_MIN_BORDER_SCORE else 0.0


def edge_card_candidates(
    frame: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    # 从边缘轮廓中寻找卡片候选，适合卡片颜色不够亮但边框清楚的场景
    height, width = frame.shape[:2]
    frame_area = height * width

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []

    for contour in contours:
        # 面积、边长、长宽比和矩形度共同过滤掉细碎边缘和非卡片物体
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / frame_area
        if area_ratio < EDGE_CARD_MIN_AREA_RATIO or area_ratio > EDGE_CARD_MAX_AREA_RATIO:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w < EDGE_CARD_MIN_SIDE or rect_h < EDGE_CARD_MIN_SIDE:
            continue

        aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)
        if aspect > EDGE_CARD_MAX_ASPECT:
            continue

        rect_area = rect_w * rect_h
        rectangularity = contour_area / rect_area if rect_area > 0 else 0.0
        if rectangularity < EDGE_CARD_MIN_RECTANGULARITY:
            continue

        points = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        box = expand_box((x, y, w, h), (height, width), ratio=0.02)
        if face_boxes is not None and overlaps_face(box, face_boxes):
            continue
        crop = warp_card(frame, pad_card_points(points))
        border_score = card_border_score(crop)
        if border_score <= 0.0:
            continue

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edge_density = float(cv2.countNonZero(cv2.Canny(crop_gray, 35, 110))) / float(crop_gray.size)
        score = (
            rectangularity * 0.55
            + min(area_ratio * 1.2, 0.75)
            + min(edge_density * 4.0, 0.35)
            + border_score
            - abs(1.0 - aspect) * 0.15
        )
        candidates.append((score, box, crop))

    return candidates


def card_candidate_crops(
    frame: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    # 先用“亮且低饱和”的颜色特征找白色卡片，再融合边缘候选
    height, width = frame.shape[:2]
    frame_area = height * width
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bright = hsv[:, :, 2] > 180
    low_saturation = hsv[:, :, 1] < 135
    mask = np.where(bright & low_saturation, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []

    for contour in contours:
        # 逐个轮廓评估是否像卡片，并把通过的区域透视矫正为裁剪图
        contour_area = cv2.contourArea(contour)
        if contour_area < frame_area * CARD_MIN_AREA_RATIO or contour_area > frame_area * CARD_MAX_AREA_RATIO:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w < CARD_MIN_SIDE or rect_h < CARD_MIN_SIDE:
            continue
        aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)
        if aspect > 1.85:
            continue

        rect_area = rect_w * rect_h
        rectangularity = contour_area / rect_area if rect_area > 0 else 0.0
        if rectangularity < CARD_MIN_RECTANGULARITY:
            continue

        points = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        box = expand_box((x, y, w, h), (height, width), ratio=0.04)
        if face_boxes is not None and overlaps_face(box, face_boxes):
            continue
        crop = warp_card(frame, points)
        score = card_crop_score(crop)
        if score <= 0.0:
            continue
        border_score = card_border_score(crop)
        if border_score <= 0.0:
            continue
        candidates.append((score + rectangularity * 0.35 + border_score, box, crop))

    candidates.extend(edge_card_candidates(frame, face_boxes=face_boxes))
    candidates = remove_container_candidates(candidates)
    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for _, box, crop in candidates:
        # 对卡片候选做 NMS，只保留最可靠的几个裁剪图
        if all(box_iou(box, old_box) < NMS_IOU_THRESHOLD for old_box, _ in kept):
            kept.append((box, crop))
        if len(kept) >= MAX_CARD_CANDIDATES:
            break
    return kept


def candidate_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    # 兜底候选框生成：当不要求卡片时，基于边缘轮廓找可能的物体区域
    height, width = frame.shape[:2]
    frame_area = height * width

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 120)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < frame_area * 0.006 or area > frame_area * 0.75:
            continue
        if w < 32 or h < 32:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.18 or aspect > 5.5:
            continue

        crop = gray[y : y + h, x : x + w]
        if crop.size and float(np.std(crop)) < 10.0:
            continue
        boxes.append(expand_box((x, y, w, h), (height, width)))

    boxes = nms_boxes(boxes, NMS_IOU_THRESHOLD)
    boxes = sorted(boxes, key=box_area, reverse=True)[:MAX_CANDIDATES]

    if not boxes:
        # 完全找不到轮廓时，用画面中央区域作为保底候选，避免流程空转
        side = int(min(height, width) * 0.58)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        boxes.append((x1, y1, x1 + side, y1 + side))

    return boxes


def build_motion_mask(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray]:
    # 用相邻帧差分生成运动掩码，帮助过滤静止背景中的误检
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if previous_gray is None or previous_gray.shape != gray.shape:
        return None, gray

    diff = cv2.absdiff(gray, previous_gray)
    _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask, gray


def motion_score(mask: np.ndarray | None, box: tuple[int, int, int, int]) -> float:
    # 统计候选框内运动像素比例，返回值越高说明区域越可能是新出现目标
    if mask is None:
        return 0.0
    x1, y1, x2, y2 = box
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(cv2.countNonZero(region)) / float(region.size)


def proposal_detections(
    proposal_model: YOLO,
    frame: np.ndarray,
    motion_mask: np.ndarray | None,
    face_boxes: list[tuple[int, int, int, int]],
) -> list[Detection]:
    # 使用通用 YOLO 模型先定位 COCO 类别，再映射到本项目标签
    result = proposal_model.predict(
        frame,
        conf=PROPOSAL_CONF_THRESHOLD,
        iou=PROPOSAL_IOU_THRESHOLD,
        verbose=False,
    )[0]
    if result.boxes is None:
        return []

    detections: list[Detection] = []
    height, width = frame.shape[:2]

    for raw_box in result.boxes:
        # 只保留能映射到目标类别且置信度达标的通用检测结果
        coco_name = result.names[int(raw_box.cls[0])]
        label = COCO_TO_TARGET.get(coco_name)
        if label is None:
            continue
        confidence = float(raw_box.conf[0])
        if confidence < PROPOSAL_CLASS_THRESHOLDS.get(coco_name, PROPOSAL_CONF_THRESHOLD):
            continue

        x1, y1, x2, y2 = raw_box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        box = (
            max(0, x1),
            max(0, y1),
            min(width - 1, x2),
            min(height - 1, y2),
        )
        if overlaps_face(box, face_boxes):
            continue

        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=box,
                margin=1.0,
                motion=motion_score(motion_mask, box),
                trusted=True,
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def orientation_variants(crop: np.ndarray) -> list[np.ndarray]:
    # 测试时增强：同一张裁剪图旋转四个方向，取最可信的分类结果
    if not ENABLE_ORIENTATION_TTA:
        return [crop]
    return [
        crop,
        cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(crop, cv2.ROTATE_180),
        cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def classify_candidate_crops(
    model: YOLO,
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]],
    motion_mask: np.ndarray | None,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
    allowed_labels: set[str] | None = None,
    min_confidence: float | None = None,
    min_motion: float = 0.0,
    trusted: bool = False,
) -> list[Detection]:
    # 对候选裁剪图批量分类，并按置信度、边际分数、运动和人脸重叠过滤
    valid = [(box, crop) for box, crop in candidates if crop.size]
    if not valid:
        return []

    variant_images: list[np.ndarray] = []
    variant_candidate_indices: list[int] = []
    for candidate_index, (_, crop) in enumerate(valid):
        # 记录旋转图对应的原始候选索引，方便后面按候选聚合最佳结果
        for variant in orientation_variants(crop):
            variant_images.append(variant)
            variant_candidate_indices.append(candidate_index)

    results = model.predict(variant_images, verbose=False)
    best_by_candidate: dict[int, tuple[str, float, float]] = {}
    detections: list[Detection] = []

    for candidate_index, result in zip(variant_candidate_indices, results):
        # top1 置信度和 top1-top2 差值同时使用，降低“相似类别”误判
        scores = result.probs.data.detach().cpu()
        values, indices = scores.topk(2)
        confidence = float(values[0])
        margin = float(values[0] - values[1]) if len(values) > 1 else confidence
        label = result.names[int(indices[0])]
        if allowed_labels is not None and label not in allowed_labels:
            continue

        score = confidence + margin * ORIENTATION_TTA_MARGIN_WEIGHT
        old = best_by_candidate.get(candidate_index)
        if old is None or score > old[1] + old[2] * ORIENTATION_TTA_MARGIN_WEIGHT:
            best_by_candidate[candidate_index] = (label, confidence, margin)

    for candidate_index, (label, confidence, margin) in best_by_candidate.items():
        box, _ = valid[candidate_index]

        # 类别专属阈值和调用方传入阈值取更严格者
        class_confidence = CLASS_CONF_THRESHOLDS.get(label, CONF_THRESHOLD)
        required_confidence = (
            max(min_confidence, class_confidence)
            if min_confidence is not None
            else class_confidence
        )
        if confidence < required_confidence or margin < MARGIN_THRESHOLD:
            continue

        motion = motion_score(motion_mask, box)
        if motion < min_motion:
            continue
        if face_boxes is not None and overlaps_face(box, face_boxes):
            continue

        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=box,
                margin=margin,
                motion=motion,
                trusted=trusted,
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    kept_labels: set[str] = set()
    for detection in detections:
        # 对分类后的目标框再次去重，只输出数量上限内的最高分结果
        if detection.label in kept_labels:
            continue
        if all(box_iou(detection.box, old.box) < NMS_IOU_THRESHOLD for old in kept):
            kept.append(detection)
            kept_labels.add(detection.label)
        if len(kept) >= MAX_DETECTIONS:
            break
    return kept


def classify_candidates(
    model: YOLO,
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    motion_mask: np.ndarray | None,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
    allowed_labels: set[str] | None = None,
    min_confidence: float | None = None,
    min_motion: float = 0.0,
) -> list[Detection]:
    # 将普通框裁剪成图像候选，再复用统一的候选分类逻辑
    candidates = [
        (box, frame[y1:y2, x1:x2])
        for box in boxes
        for x1, y1, x2, y2 in [box]
    ]
    return classify_candidate_crops(
        model,
        candidates,
        motion_mask,
        face_boxes=face_boxes,
        allowed_labels=allowed_labels,
        min_confidence=min_confidence,
        min_motion=min_motion,
    )


def smooth_box(
    current: tuple[int, int, int, int],
    previous: tuple[int, int, int, int] | None,
    smoothing: float = BOX_SMOOTHING,
) -> tuple[int, int, int, int]:
    # 指数平滑框坐标，让显示框移动更稳定
    if previous is None:
        return current
    return tuple(
        int(previous_value * smoothing + current_value * (1.0 - smoothing))
        for previous_value, current_value in zip(previous, current)
    )


def visible_confidence_threshold(label: str) -> float:
    # 追踪中的可见阈值略低于新检测阈值，避免短暂波动造成框闪烁
    return max(
        CONF_THRESHOLD,
        CLASS_CONF_THRESHOLDS.get(label, CONF_THRESHOLD) * TRACK_VISIBLE_CONFIDENCE_RATIO,
    )


def visible_tracks(tracks: dict[str, Track]) -> list[Detection]:
    # 只显示命中次数、丢失帧数和置信度都满足条件的追踪目标
    detections = [
        Detection(track.label, track.confidence, track.box)
        for track in tracks.values()
        if track.hits >= TRACK_MIN_HITS and track.missed <= TRACK_MAX_MISSED
        and track.confidence >= visible_confidence_threshold(track.label)
    ]
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections[:MAX_DETECTIONS]


def primary_detection_score(detection: Detection, track: Track | None) -> float:
    # 单目标模式下的排序分数：偏向可信检测、大框和与当前轨迹更接近的目标
    score = detection.confidence
    if detection.trusted:
        score += 0.25
    score += min(box_area(detection.box) / 120000.0, 0.20)

    if track is not None:
        score += box_iou(detection.box, track.box) * 1.6
        score -= center_distance_ratio(detection.box, track.box) * 0.35
        if detection.label == track.label:
            score += 0.20
    return score


def pick_primary_detection(detections: list[Detection], track: Track | None) -> Detection | None:
    # 从当前帧候选中选出最适合作为主目标的检测
    if not detections:
        return None
    return max(detections, key=lambda detection: primary_detection_score(detection, track))


def visible_primary_track(track: Track | None) -> list[Detection]:
    # 将主目标轨迹转换成可绘制的检测结果；轨迹不稳定时不显示
    if track is None:
        return []
    if track.hits < TRACK_MIN_HITS or track.missed > TRACK_MAX_MISSED:
        return []
    if track.confidence < visible_confidence_threshold(track.label):
        return []
    return [Detection(track.label, track.confidence, track.box)]


def is_jump_detection(detection: Detection, track: Track | None) -> bool:
    # 判断检测框是否离当前轨迹太远，防止目标突然跳到画面另一处
    if track is None or track.missed > TRACK_MAX_MISSED:
        return False
    if detection.trusted and detection.label == track.label and detection.confidence >= 0.985:
        return False
    return (
        box_iou(detection.box, track.box) < PRIMARY_JUMP_IOU
        and center_distance_ratio(detection.box, track.box) > PRIMARY_JUMP_CENTER_RATIO
    )


def should_snap_to_trusted_detection(detection: Detection, track: Track | None) -> bool:
    # 对极高置信度的可信检测允许快速吸附，避免旧轨迹拖住新位置
    if track is None or not detection.trusted or detection.confidence < 0.985:
        return False
    return (
        box_iou(detection.box, track.box) < 0.15
        and center_distance_ratio(detection.box, track.box) > 0.35
    )


def update_primary_track(
    detections: list[Detection],
    track: Track | None,
) -> tuple[list[Detection], Track | None]:
    # 单目标模式：每帧只维护一个主轨迹，适合比赛场景中一次识别一个包裹
    detection = pick_primary_detection(detections, track)

    if detection is None or is_jump_detection(detection, track):
        # 当前帧没有可靠目标时，不立即清空；先让旧轨迹按置信度衰减几帧
        if track is None:
            return [], None
        track.missed += 1
        track.confidence *= PRIMARY_CONF_DECAY
        if (
            track.missed > TRACK_MAX_MISSED
            or track.confidence < visible_confidence_threshold(track.label) * 0.82
        ):
            return [], None
        return visible_primary_track(track), track

    if track is None or track.missed > TRACK_MAX_MISSED:
        # 新建主轨迹；可信检测可直接满足最小命中次数，减少启动延迟
        initial_hits = TRACK_MIN_HITS if detection.trusted else 1
        track = Track(
            label=detection.label,
            confidence=detection.confidence,
            box=detection.box,
            hits=initial_hits,
        )
        return visible_primary_track(track), track

    snap_to_detection = should_snap_to_trusted_detection(detection, track)
    had_missed = track.missed > 0
    # 丢帧恢复时降低平滑系数，让框更快追上实际目标位置
    box_smoothing = 0.0 if snap_to_detection else (0.82 if had_missed else BOX_SMOOTHING)
    confidence_smoothing = 0.45 if had_missed else CONFIDENCE_SMOOTHING

    track.box = smooth_box(detection.box, track.box, smoothing=box_smoothing)
    track.confidence = (
        track.confidence * confidence_smoothing
        + detection.confidence * (1.0 - confidence_smoothing)
    )
    track.hits = min(track.hits + 1, TRACK_MIN_HITS + TRACK_MAX_MISSED)
    track.missed = 0

    if detection.label == track.label:
        track.candidate_label = ""
        track.candidate_hits = 0
    else:
        # 标签变化需要连续命中多次才切换，避免一两帧误判导致输出抖动
        if detection.label == track.candidate_label:
            track.candidate_hits += 1
        else:
            track.candidate_label = detection.label
            track.candidate_hits = 1

        if (
            track.candidate_hits >= LABEL_SWITCH_HITS
            and detection.confidence >= visible_confidence_threshold(detection.label)
        ):
            track.label = detection.label
            track.candidate_label = ""
            track.candidate_hits = 0

    return visible_primary_track(track), track


def update_tracks(
    detections: list[Detection],
    tracks: dict[str, Track],
) -> tuple[list[Detection], dict[str, Track]]:
    # 多目标模式：按标签各保留一个最佳轨迹
    best_by_label: dict[str, Detection] = {}
    for detection in detections:
        old = best_by_label.get(detection.label)
        if old is None or detection.confidence > old.confidence:
            best_by_label[detection.label] = detection

    for track in tracks.values():
        # 所有旧轨迹先衰减；本帧匹配到检测后再恢复
        track.missed += 1
        track.confidence *= 0.96

    for label, detection in best_by_label.items():
        track = tracks.get(label)
        if track is None:
            # 非可信检测必须有足够运动，才允许创建新轨迹
            if not detection.trusted and detection.motion < NEW_TRACK_MIN_MOTION:
                continue
            tracks[label] = Track(label=label, confidence=detection.confidence, box=detection.box)
            continue

        track.box = smooth_box(detection.box, track.box)
        track.confidence = (
            track.confidence * CONFIDENCE_SMOOTHING
            + detection.confidence * (1.0 - CONFIDENCE_SMOOTHING)
        )
        track.hits = min(track.hits + 1, TRACK_MIN_HITS + TRACK_MAX_MISSED)
        track.missed = 0

    tracks = {
        label: track
        for label, track in tracks.items()
        if track.missed <= TRACK_MAX_MISSED
        and track.confidence >= visible_confidence_threshold(label) * 0.82
    }
    # 清理丢失太久或置信度过低的轨迹后，再返回可见目标
    return visible_tracks(tracks), tracks


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    # PIL 负责绘制中文，OpenCV 默认字体无法稳定显示中文字符。
    font = load_chinese_font()
    text_bbox = measure_text(text, font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    padding_x = 8
    padding_y = 5
    label_w = text_w + padding_x * 2
    label_h = text_h + padding_y * 2
    x = max(0, min(x, frame.shape[1] - label_w - 1))
    y = max(label_h + 2, min(y, frame.shape[0] - 2))
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if brightness > 155 else (255, 255, 255)

    background_top = y - label_h
    background_bottom = y
    cv2.rectangle(frame, (x - 2, background_top - 2), (x + label_w + 2, background_bottom + 2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, background_top), (x + label_w, background_bottom), color, -1)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text(
        (x + padding_x, background_top + padding_y - text_bbox[1]),
        text,
        font=font,
        fill=bgr_to_rgb(text_color),
    )
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=1)
def load_chinese_font() -> ImageFont.ImageFont:
    for font_path in CHINESE_FONT_PATHS:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), CHINESE_FONT_SIZE)
    return ImageFont.load_default()


def measure_text(text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(image)
    return draw.textbbox((0, 0), text, font=font)


def bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


def detection_display_text(detection: Detection) -> str:
    item_name = ITEM_NAMES_CN.get(detection.label, detection.label)
    category_key = ITEM_CATEGORIES.get(detection.label, "unknown")
    category_name = CATEGORY_NAMES_CN.get(category_key, "未知类别")
    return f"类别：{category_name}  物品：{item_name}  {detection.confidence:.2f}"


def draw_detections(frame: np.ndarray, detections: list[Detection], fps: float) -> None:
    # 在画面上绘制检测框、标签和 FPS
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = BOX_COLORS.get(detection.label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, detection_display_text(detection), x1, y1 - 6, color)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)


def detection_signature(detections: list[Detection]) -> tuple[str, ...]:
    # 用标签集合表示当前输出结果，便于判断终端是否需要重复打印
    return tuple(sorted(detection.label for detection in detections))


def should_print_detection(
    current_signature: tuple[str, ...],
    last_signature: tuple[str, ...],
    last_printed_at: float,
    now: float,
) -> bool:
    # 结果变化时立即打印；结果不变时按固定间隔重复提醒
    if not current_signature:
        return False
    if current_signature != last_signature:
        return True
    return now - last_printed_at >= TERMINAL_REPEAT_INTERVAL


def print_detection_results(detections: list[Detection]) -> None:
    # 将识别结果转换成中文句子输出到终端
    if not detections:
        return

    for detection in detections:
        item_name = ITEM_NAMES_CN.get(detection.label, detection.label)
        category_key = ITEM_CATEGORIES.get(detection.label, "unknown")
        category_name = CATEGORY_NAMES_CN.get(category_key, "未知类别")
        print(f"图中的包裹是{item_name}，类别为{category_name}", flush=True)


def main() -> None:
    # 启动前确认训练好的分类模型存在
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run python train.py first.")

    # 根据配置加载分类模型、可选通用检测模型、人脸检测器和摄像头
    model = YOLO(str(MODEL_PATH))
    proposal_model = None if REQUIRE_CARD_FOR_DETECTION else YOLO(str(PROPOSAL_MODEL_PATH))
    face_detectors = load_face_detectors() if ENABLE_FACE_FILTER else []
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Unable to open camera.")
    configure_camera(camera)

    previous_time = time.time()
    tracks: dict[str, Track] = {}
    primary_track: Track | None = None
    visible_detections: list[Detection] = []
    frame_index = 0
    previous_gray: np.ndarray | None = None
    last_printed_signature: tuple[str, ...] = ()
    last_printed_at = 0.0
    card_missing_frames = 0

    print("Started item detection. Press q or Esc to quit.")

    try:
        while True:
            # 逐帧读取摄像头画面
            ok, frame = camera.read()
            if not ok:
                print("Failed to read camera frame.")
                break

            frame = resize_for_display(frame, PROCESS_WIDTH)
            motion_mask, previous_gray = build_motion_mask(frame, previous_gray)
            if frame_index % DETECTION_INTERVAL == 0:
                # 每隔若干帧运行一次完整检测，其余帧复用追踪结果以提高速度
                face_boxes = detect_faces(frame, face_detectors) if ENABLE_FACE_FILTER else []
                detections: list[Detection] = []

                if ENABLE_CONTOUR_FALLBACK:
                    # 优先找卡片/包裹区域，再对裁剪图做分类
                    card_candidates = card_candidate_crops(frame, face_boxes=face_boxes)
                    card_detections = classify_candidate_crops(
                        model,
                        card_candidates,
                        motion_mask,
                        face_boxes=face_boxes,
                        allowed_labels=TARGET_LABELS,
                        min_confidence=CARD_CONF_THRESHOLD,
                        min_motion=0.0,
                        trusted=True,
                    )
                    if card_detections:
                        detections = card_detections

                if not detections:
                    card_missing_frames += 1
                    if REQUIRE_CARD_FOR_DETECTION:
                        # 强制要求卡片时，连续缺失会清空轨迹，避免旧结果一直停留
                        if card_missing_frames >= CARD_MISSING_CLEAR_FRAMES:
                            primary_track = None
                            tracks.clear()
                        detections = []
                    else:
                        # 不强制卡片时，退回到通用检测模型和轮廓候选分类
                        detections = (
                            proposal_detections(proposal_model, frame, motion_mask, face_boxes)
                            if proposal_model is not None
                            else []
                        )

                        if ENABLE_CONTOUR_FALLBACK:
                            boxes = candidate_boxes(frame)
                            detections.extend(
                                classify_candidates(
                                    model,
                                    frame,
                                    boxes,
                                    motion_mask,
                                    face_boxes=face_boxes,
                                    allowed_labels=FALLBACK_ONLY_LABELS,
                                    min_confidence=FALLBACK_CONF_THRESHOLD,
                                    min_motion=FALLBACK_MIN_MOTION,
                                )
                            )
                else:
                    card_missing_frames = 0

                if SINGLE_TARGET_MODE:
                    # 比赛模式默认只输出一个主目标
                    visible_detections, primary_track = update_primary_track(detections, primary_track)
                else:
                    visible_detections, tracks = update_tracks(detections, tracks)
            else:
                # 非完整检测帧只刷新已有轨迹，减少模型推理次数
                visible_detections = (
                    visible_primary_track(primary_track)
                    if SINGLE_TARGET_MODE
                    else visible_tracks(tracks)
                )
            frame_index += 1

            now = time.time()
            current_signature = detection_signature(visible_detections)
            if should_print_detection(current_signature, last_printed_signature, last_printed_at, now):
                # 终端打印会限频，避免同一结果刷屏
                print_detection_results(visible_detections)
                last_printed_signature = current_signature
                last_printed_at = now

            fps = 1 / max(now - previous_time, 1e-6)
            previous_time = now

            draw_detections(frame, visible_detections, fps)
            display_frame = resize_for_display(frame, DISPLAY_WIDTH)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        # 无论正常退出还是异常退出，都释放摄像头和窗口资源
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
