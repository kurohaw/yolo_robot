from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import detect  # noqa: E402


ROW_MAJOR_LABELS = (
    "television_set",
    "air_conditioner",
    "tissue_paper",
    "clothes",
    "banana_fruit",
    "orange_fruit",
    "apple_fruit",
    "toothbrush",
    "refrigerator_appliance",
)
FRUIT_LABELS = {"apple_fruit", "banana_fruit", "orange_fruit"}
APPLIANCE_CONFUSION_LABELS = {"air_conditioner", "refrigerator_appliance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic classifier dataset from a 3x3 source sheet.")
    parser.add_argument("--source", type=Path, required=True, help="Photo containing the 3x3 competition sheet.")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "target_item_classes")
    parser.add_argument("--train-count", type=int, default=180, help="Images per non-fruit class.")
    parser.add_argument("--val-count", type=int, default=45, help="Validation images per non-fruit class.")
    parser.add_argument("--fruit-multiplier", type=int, default=2, help="Extra samples for apple/banana/orange.")
    parser.add_argument("--appliance-multiplier", type=int, default=2, help="Extra samples for air-conditioner/refrigerator.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Delete the output dataset before generating.")
    return parser.parse_args()


def sort_grid(candidates: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    if len(candidates) < 9:
        raise RuntimeError(f"Expected at least 9 card candidates, found {len(candidates)}.")

    largest = sorted(candidates, key=lambda item: detect.box_area(item[0]), reverse=True)[:9]
    largest.sort(key=lambda item: ((item[0][1] + item[0][3]) / 2.0, (item[0][0] + item[0][2]) / 2.0))

    rows = [largest[index : index + 3] for index in range(0, 9, 3)]
    ordered: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: (item[0][0] + item[0][2]) / 2.0))
    return ordered


def load_base_crops(source: Path) -> dict[str, np.ndarray]:
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Unable to read source image: {source}")

    old_max_card_candidates = detect.MAX_CARD_CANDIDATES
    detect.MAX_CARD_CANDIDATES = 14
    try:
        candidates = detect.card_candidate_crops(image, face_boxes=[])
    finally:
        detect.MAX_CARD_CANDIDATES = old_max_card_candidates
    ordered = sort_grid(candidates)
    crops: dict[str, np.ndarray] = {}
    for label, (_, crop) in zip(ROW_MAJOR_LABELS, ordered):
        crops[label] = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
    return crops


def random_perspective(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    height, width = image.shape[:2]
    jitter = min(height, width) * strength
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 242),
    )


def random_affine(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.86, 1.12))
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += float(rng.uniform(-12.0, 12.0))
    matrix[1, 2] += float(rng.uniform(-12.0, 12.0))
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 242),
    )


def random_color(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + int(rng.integers(-4, 5)), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(rng.uniform(0.75, 1.35)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(rng.uniform(0.72, 1.28)) + int(rng.integers(-18, 19)), 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(image, alpha=float(rng.uniform(0.88, 1.15)), beta=int(rng.integers(-12, 13)))


def random_shadow(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > 0.55:
        return image
    height, width = image.shape[:2]
    x_gradient = np.linspace(float(rng.uniform(0.72, 1.0)), float(rng.uniform(0.72, 1.0)), width)
    y_gradient = np.linspace(float(rng.uniform(0.82, 1.0)), float(rng.uniform(0.82, 1.0)), height)
    mask = np.outer(y_gradient, x_gradient)
    shaded = image.astype(np.float32) * mask[:, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def random_noise_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.35:
        kernel = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    if rng.random() < 0.25:
        noise = rng.normal(0.0, float(rng.uniform(2.0, 7.0)), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def random_low_contrast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray_bias = np.full_like(image, int(rng.integers(212, 242)))
    alpha = float(rng.uniform(0.55, 0.85))
    image = cv2.addWeighted(image, alpha, gray_bias, 1.0 - alpha, 0)
    return cv2.convertScaleAbs(image, alpha=float(rng.uniform(0.82, 1.05)), beta=int(rng.integers(-10, 11)))


def random_resolution_loss(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > 0.75:
        return image
    side = int(rng.integers(72, 145))
    small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    interpolation = cv2.INTER_LINEAR if rng.random() < 0.65 else cv2.INTER_NEAREST
    return cv2.resize(small, image.shape[:2][::-1], interpolation=interpolation)


def random_edge_crop(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > 0.45:
        return image
    height, width = image.shape[:2]
    margin = int(rng.integers(0, 15))
    x1 = int(rng.integers(0, margin + 1))
    y1 = int(rng.integers(0, margin + 1))
    x2 = width - int(rng.integers(0, margin + 1))
    y2 = height - int(rng.integers(0, margin + 1))
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def augment(image: np.ndarray, rng: np.random.Generator, label: str, split: str) -> np.ndarray:
    augmented = image.copy()
    strength = 0.035 if split == "train" else 0.025
    augmented = random_perspective(augmented, rng, strength)
    augmented = random_affine(augmented, rng)
    augmented = random_color(augmented, rng)
    augmented = random_shadow(augmented, rng)
    augmented = random_noise_blur(augmented, rng)

    if label in FRUIT_LABELS and split == "train":
        # Fruit classes get stronger color and blur variation to reduce apple/orange/banana confusion.
        augmented = random_color(augmented, rng)
        if rng.random() < 0.25:
            augmented = cv2.GaussianBlur(augmented, (5, 5), 0)

    if label in APPLIANCE_CONFUSION_LABELS:
        # Air-conditioner and refrigerator are both pale appliances; stress thin lines and low contrast.
        augmented = random_low_contrast(augmented, rng)
        augmented = random_resolution_loss(augmented, rng)
        augmented = random_edge_crop(augmented, rng)
        if rng.random() < (0.35 if split == "train" else 0.20):
            augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return cv2.resize(augmented, (320, 320), interpolation=cv2.INTER_AREA)


def save_dataset(
    crops: dict[str, np.ndarray],
    output: Path,
    train_count: int,
    val_count: int,
    fruit_multiplier: int,
    appliance_multiplier: int,
    rng: np.random.Generator,
) -> None:
    for split, base_count in (("train", train_count), ("val", val_count)):
        for label, crop in crops.items():
            multiplier = 1
            if label in FRUIT_LABELS:
                multiplier = fruit_multiplier
            if label in APPLIANCE_CONFUSION_LABELS:
                multiplier = max(multiplier, appliance_multiplier)
            count = base_count * multiplier
            folder = output / split / label
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{label}_base.jpg"), crop)
            for index in range(count):
                image = augment(crop, rng, label, split)
                cv2.imwrite(str(folder / f"{label}_{index:04d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)

    crops = load_base_crops(args.source)
    rng = np.random.default_rng(args.seed)
    save_dataset(crops, output, args.train_count, args.val_count, args.fruit_multiplier, args.appliance_multiplier, rng)

    print(f"Generated dataset at {output}")
    for split in ("train", "val"):
        for label in ROW_MAJOR_LABELS:
            count = len(list((output / split / label).glob("*.jpg")))
            print(f"{split:5} {label:24} {count:4}")


if __name__ == "__main__":
    main()
