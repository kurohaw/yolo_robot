from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from datetime import datetime

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "air_conditioner",
    "apple_fruit",
    "banana_fruit",
    "clothes",
    "orange_fruit",
    "refrigerator_appliance",
    "television_set",
    "tissue_paper",
    "toothbrush",
)
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
RUNTIME_WEIGHTS = ROOT / "runs" / "classify" / "target_item_classes" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the 9-item classification model.")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "target_item_classes")
    parser.add_argument("--model", type=Path, default=RUNTIME_WEIGHTS, help="Starting weights.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224, help="Use 224 for speed, 320 for maximum detail.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="target_item_classes_retrain")
    parser.add_argument("--device", default=None, help="Examples: cpu, 0")
    parser.add_argument("--install", action="store_true", help="Replace runtime best.pt after training.")
    return parser.parse_args()


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)


def validate_dataset(data_dir: Path) -> None:
    missing: list[str] = []
    for split in ("train", "val"):
        for label in LABELS:
            folder = data_dir / split / label
            if count_images(folder) == 0:
                missing.append(str(folder))
    if missing:
        print("Dataset is incomplete. Missing images in:")
        for folder in missing:
            print(f"  - {folder}")
        print("\nExpected structure: datasets/target_item_classes/{train,val}/{class_name}/*.jpg")
        raise SystemExit(2)

    print("Dataset image counts:")
    for label in LABELS:
        train_count = count_images(data_dir / "train" / label)
        val_count = count_images(data_dir / "val" / label)
        print(f"  {label:24} train={train_count:4} val={val_count:4}")
        if train_count < 50 or val_count < 10:
            print("    warning: collect more samples for stable competition performance")


def install_weights(best_path: Path) -> None:
    RUNTIME_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if RUNTIME_WEIGHTS.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = RUNTIME_WEIGHTS.with_name(f"best_backup_{stamp}.pt")
        shutil.copy2(RUNTIME_WEIGHTS, backup)
        print(f"Backed up old weights to {backup}")
    shutil.copy2(best_path, RUNTIME_WEIGHTS)
    print(f"Installed new runtime weights to {RUNTIME_WEIGHTS}")


def main() -> None:
    args = parse_args()
    data_dir = args.data.resolve()
    validate_dataset(data_dir)

    model_path = args.model if args.model.exists() else "yolov8n-cls.pt"
    model = YOLO(str(model_path))
    result = model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(ROOT / "runs" / "classify"),
        name=args.name,
        exist_ok=True,
        device=args.device,
        seed=42,
        fliplr=0.0,
        degrees=8.0,
        translate=0.08,
        scale=0.20,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,
    )

    save_dir = Path(result.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    print(f"Training complete: {best_path}")
    print("Run validation before installing:")
    print(f"python tools/validate_classifier.py --model {best_path} --data {data_dir / 'val'} --imgsz {args.imgsz}")
    if args.install:
        install_weights(best_path)


if __name__ == "__main__":
    main()
