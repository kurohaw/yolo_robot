# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path("runs/classify/target_item_classes/weights/best.pt")
PROPOSAL_MODEL_PATH = Path("yolov8n.pt")
WINDOW_NAME = "YOLO 9 Item Detection"

CAMERA_INDEX = 0
DISPLAY_WIDTH = 960

MAX_CANDIDATES = 14
MAX_CARD_CANDIDATES = 6
MAX_DETECTIONS = 1
PROPOSAL_CONF_THRESHOLD = 0.28
PROPOSAL_IOU_THRESHOLD = 0.45
CONF_THRESHOLD = 0.78
MARGIN_THRESHOLD = 0.12
NMS_IOU_THRESHOLD = 0.35
DETECTION_INTERVAL = 2
TRACK_MIN_HITS = 3
TRACK_MAX_MISSED = 8
BOX_SMOOTHING = 0.90
CONFIDENCE_SMOOTHING = 0.82
MOTION_THRESHOLD = 18
NEW_TRACK_MIN_MOTION = 0.006
ENABLE_CONTOUR_FALLBACK = True
FALLBACK_CONF_THRESHOLD = 0.90
FALLBACK_MIN_MOTION = 0.015
FACE_OVERLAP_THRESHOLD = 0.12
CARD_CONF_THRESHOLD = 0.78
TRACK_VISIBLE_CONFIDENCE_RATIO = 0.90
SINGLE_TARGET_MODE = True
LABEL_SWITCH_HITS = 3
PRIMARY_JUMP_IOU = 0.06
PRIMARY_JUMP_CENTER_RATIO = 0.55
PRIMARY_CONF_DECAY = 0.97
TERMINAL_REPEAT_INTERVAL = 5.0
REQUIRE_CARD_FOR_DETECTION = True
CARD_MISSING_CLEAR_FRAMES = 2
DEBUG_PRINT_INTERVAL = 1.0
CARD_MIN_BRIGHT_RATIO = 0.46
CARD_MIN_EDGE_DENSITY = 0.002
CARD_MAX_EDGE_DENSITY = 0.18
CARD_MIN_DARK_RATIO = 0.001
CARD_MAX_DARK_RATIO = 0.55
CARD_MIN_RECTANGULARITY = 0.62
ENABLE_ORIENTATION_TTA = True
ORIENTATION_TTA_MARGIN_WEIGHT = 0.25
CARD_MIN_BORDER_SIDES = 3
CARD_BORDER_STRIP_RATIO = 0.06
CARD_BORDER_DARK_THRESHOLD = 0.025
CARD_BORDER_EDGE_THRESHOLD = 0.020
CARD_MIN_BORDER_SCORE = 1.20
EDGE_CARD_MIN_AREA_RATIO = 0.05
EDGE_CARD_MAX_AREA_RATIO = 0.75
EDGE_CARD_MIN_SIDE = 120
EDGE_CARD_MAX_ASPECT = 1.45
EDGE_CARD_MIN_RECTANGULARITY = 0.45

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

COCO_TO_TARGET = {
    "apple": "apple_fruit",
    "banana": "banana_fruit",
    "orange": "orange_fruit",
    "refrigerator": "refrigerator_appliance",
    "toothbrush": "toothbrush",
    "tv": "television_set",
}

PROPOSAL_CLASS_THRESHOLDS = {
    "apple": 0.35,
    "banana": 0.35,
    "orange": 0.35,
    "refrigerator": 0.65,
    "toothbrush": 0.35,
    "tv": 0.55,
}

FALLBACK_ONLY_LABELS = TARGET_LABELS

CLASS_CONF_THRESHOLDS = {
    "air_conditioner": 0.78,
    "clothes": 0.78,
    "refrigerator_appliance": 0.84,
    "television_set": 0.84,
    "tissue_paper": 0.82,
}

DISPLAY_NAMES = {
    "toothbrush": "Toothbrush",
    "tissue_paper": "Tissue Paper",
    "clothes": "Clothes",
    "banana_fruit": "Banana",
    "apple_fruit": "Apple",
    "orange_fruit": "Orange",
    "television_set": "Television",
    "refrigerator_appliance": "Refrigerator",
    "air_conditioner": "Air Conditioner",
}

CATEGORY_NAMES = {
    "daily_items": "Daily Items",
    "fruits": "Fruits",
    "home_appliances": "Home Appliances",
}

CATEGORY_NAMES_CN = {
    "daily_items": "日用品",
    "fruits": "水果",
    "home_appliances": "家电",
}

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
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    margin: float = 0.0
    motion: float = 1.0
    trusted: bool = False


@dataclass
class Track:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    hits: int = 1
    missed: int = 0
    candidate_label: str = ""
    candidate_hits: int = 0


def load_face_detectors() -> list[cv2.CascadeClassifier]:
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
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    scale = width / current_width
    return cv2.resize(frame, (width, int(height * scale)), interpolation=cv2.INTER_AREA)


def expand_box(box: tuple[int, int, int, int], shape: tuple[int, int], ratio: float = 0.12) -> tuple[int, int, int, int]:
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
    if not detectors:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_boxes: list[tuple[int, int, int, int]] = []
    for detector in detectors:
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_boxes.extend((int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces)
    return nms_boxes(face_boxes, 0.30)


def overlaps_face(box: tuple[int, int, int, int], face_boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(
        overlap_ratio(face_box, box) > FACE_OVERLAP_THRESHOLD
        or overlap_ratio(box, face_box) > FACE_OVERLAP_THRESHOLD
        for face_box in face_boxes
    )


def nms_boxes(boxes: list[tuple[int, int, int, int]], threshold: float) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=box_area, reverse=True):
        if all(box_iou(box, old_box) < threshold for old_box in kept):
            kept.append(box)
    return kept


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def warp_card(frame: np.ndarray, points: np.ndarray, size: int = 320) -> np.ndarray:
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
    height, width = frame.shape[:2]
    frame_area = height * width

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []

    for contour in contours:
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
        crop = warp_card(frame, points)
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
        contour_area = cv2.contourArea(contour)
        if contour_area < frame_area * 0.018 or contour_area > frame_area * 0.70:
            continue

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w < 70 or rect_h < 70:
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
    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for _, box, crop in candidates:
        if all(box_iou(box, old_box) < NMS_IOU_THRESHOLD for old_box, _ in kept):
            kept.append((box, crop))
        if len(kept) >= MAX_CARD_CANDIDATES:
            break
    return kept


def candidate_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
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
        side = int(min(height, width) * 0.58)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        boxes.append((x1, y1, x1 + side, y1 + side))

    return boxes


def build_motion_mask(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray]:
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
    valid = [(box, crop) for box, crop in candidates if crop.size]
    if not valid:
        return []

    variant_images: list[np.ndarray] = []
    variant_candidate_indices: list[int] = []
    for candidate_index, (_, crop) in enumerate(valid):
        for variant in orientation_variants(crop):
            variant_images.append(variant)
            variant_candidate_indices.append(candidate_index)

    results = model.predict(variant_images, verbose=False)
    best_by_candidate: dict[int, tuple[str, float, float]] = {}
    detections: list[Detection] = []

    for candidate_index, result in zip(variant_candidate_indices, results):
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
    for detection in detections:
        if all(box_iou(detection.box, old.box) < NMS_IOU_THRESHOLD for old in kept):
            kept.append(detection)
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
    if previous is None:
        return current
    return tuple(
        int(previous_value * smoothing + current_value * (1.0 - smoothing))
        for previous_value, current_value in zip(previous, current)
    )


def visible_confidence_threshold(label: str) -> float:
    return max(
        CONF_THRESHOLD,
        CLASS_CONF_THRESHOLDS.get(label, CONF_THRESHOLD) * TRACK_VISIBLE_CONFIDENCE_RATIO,
    )


def visible_tracks(tracks: dict[str, Track]) -> list[Detection]:
    detections = [
        Detection(track.label, track.confidence, track.box)
        for track in tracks.values()
        if track.hits >= TRACK_MIN_HITS and track.missed <= TRACK_MAX_MISSED
        and track.confidence >= visible_confidence_threshold(track.label)
    ]
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections[:MAX_DETECTIONS]


def primary_detection_score(detection: Detection, track: Track | None) -> float:
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
    if not detections:
        return None
    return max(detections, key=lambda detection: primary_detection_score(detection, track))


def visible_primary_track(track: Track | None) -> list[Detection]:
    if track is None:
        return []
    if track.hits < TRACK_MIN_HITS or track.missed > TRACK_MAX_MISSED:
        return []
    if track.confidence < visible_confidence_threshold(track.label):
        return []
    return [Detection(track.label, track.confidence, track.box)]


def is_jump_detection(detection: Detection, track: Track | None) -> bool:
    if track is None or track.missed > TRACK_MAX_MISSED:
        return False
    if detection.trusted and detection.label == track.label and detection.confidence >= 0.985:
        return False
    return (
        box_iou(detection.box, track.box) < PRIMARY_JUMP_IOU
        and center_distance_ratio(detection.box, track.box) > PRIMARY_JUMP_CENTER_RATIO
    )


def should_snap_to_trusted_detection(detection: Detection, track: Track | None) -> bool:
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
    detection = pick_primary_detection(detections, track)

    if detection is None or is_jump_detection(detection, track):
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
    best_by_label: dict[str, Detection] = {}
    for detection in detections:
        old = best_by_label.get(detection.label)
        if old is None or detection.confidence > old.confidence:
            best_by_label[detection.label] = detection

    for track in tracks.values():
        track.missed += 1
        track.confidence *= 0.96

    for label, detection in best_by_label.items():
        track = tracks.get(label)
        if track is None:
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
    return visible_tracks(tracks), tracks


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(text_h + baseline + 4, y)
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if brightness > 155 else (255, 255, 255)

    cv2.rectangle(frame, (x - 2, y - text_h - baseline - 8), (x + text_w + 10, y + 6), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y - text_h - baseline - 6), (x + text_w + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - baseline), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: list[Detection], fps: float) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = BOX_COLORS.get(detection.label, (0, 255, 0))
        label = DISPLAY_NAMES.get(detection.label, detection.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, f"{label} {detection.confidence:.2f}", x1, y1 - 6, color)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)


def detection_signature(detections: list[Detection]) -> tuple[str, ...]:
    return tuple(sorted(detection.label for detection in detections))


def should_print_detection(
    current_signature: tuple[str, ...],
    last_signature: tuple[str, ...],
    last_printed_at: float,
    now: float,
) -> bool:
    if not current_signature:
        return False
    if current_signature != last_signature:
        return True
    return now - last_printed_at >= TERMINAL_REPEAT_INTERVAL


def print_detection_results(detections: list[Detection]) -> None:
    if not detections:
        return

    for detection in detections:
        item_name = ITEM_NAMES_CN.get(detection.label, detection.label)
        category_key = ITEM_CATEGORIES.get(detection.label, "unknown")
        category_name = CATEGORY_NAMES_CN.get(category_key, "未知类别")
        print(f"图中的包裹是{item_name}，类别为{category_name}", flush=True)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run python train.py first.")

    model = YOLO(str(MODEL_PATH))
    proposal_model = None if REQUIRE_CARD_FOR_DETECTION else YOLO(str(PROPOSAL_MODEL_PATH))
    face_detectors = load_face_detectors()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Unable to open camera.")

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
            ok, frame = camera.read()
            if not ok:
                print("Failed to read camera frame.")
                break

            frame = resize_for_display(frame, DISPLAY_WIDTH)
            motion_mask, previous_gray = build_motion_mask(frame, previous_gray)
            if frame_index % DETECTION_INTERVAL == 0:
                face_boxes = detect_faces(frame, face_detectors)
                detections: list[Detection] = []

                if ENABLE_CONTOUR_FALLBACK:
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
                        if card_missing_frames >= CARD_MISSING_CLEAR_FRAMES:
                            primary_track = None
                            tracks.clear()
                        detections = []
                    else:
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
                    visible_detections, primary_track = update_primary_track(detections, primary_track)
                else:
                    visible_detections, tracks = update_tracks(detections, tracks)
            else:
                visible_detections = (
                    visible_primary_track(primary_track)
                    if SINGLE_TARGET_MODE
                    else visible_tracks(tracks)
                )
            frame_index += 1

            now = time.time()
            current_signature = detection_signature(visible_detections)
            if should_print_detection(current_signature, last_printed_signature, last_printed_at, now):
                print_detection_results(visible_detections)
                last_printed_signature = current_signature
                last_printed_at = now

            fps = 1 / max(now - previous_time, 1e-6)
            previous_time = now

            draw_detections(frame, visible_detections, fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
