from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate classifier accuracy and confusion pairs.")
    parser.add_argument("--model", type=Path, default=ROOT / "runs" / "classify" / "target_item_classes" / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "target_item_classes" / "val")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None, help="Examples: cpu, 0")
    return parser.parse_args()


def iter_images(data_dir: Path):
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image in sorted(class_dir.rglob("*")):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                yield class_dir.name, image


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Validation folder not found: {args.data}")

    model = YOLO(str(args.model))
    total = 0
    correct = 0
    by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: Counter[tuple[str, str]] = Counter()
    elapsed = 0.0

    for truth, image in iter_images(args.data):
        start = time.perf_counter()
        result = model.predict(str(image), imgsz=args.imgsz, device=args.device, verbose=False)[0]
        elapsed += time.perf_counter() - start

        scores = result.probs.data.detach().cpu()
        pred = result.names[int(scores.argmax())]
        total += 1
        by_class[truth][1] += 1
        if pred == truth:
            correct += 1
            by_class[truth][0] += 1
        else:
            confusion[(truth, pred)] += 1

    if total == 0:
        raise RuntimeError(f"No validation images found under {args.data}")

    print(f"Overall accuracy: {correct}/{total} = {correct / total:.3f}")
    print(f"Average predict time: {elapsed / total * 1000:.1f} ms/image")
    print("\nPer-class accuracy:")
    for label in sorted(by_class):
        ok, count = by_class[label]
        print(f"  {label:24} {ok:4}/{count:<4} {ok / count:.3f}")

    if confusion:
        print("\nTop confusions:")
        for (truth, pred), count in confusion.most_common(20):
            print(f"  {truth:24} -> {pred:24} {count}")
    else:
        print("\nNo confusions.")


if __name__ == "__main__":
    main()
