from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import detect  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rectified card crops for classifier retraining.")
    parser.add_argument("--label", choices=LABELS, required=True, help="Class folder to save samples into.")
    parser.add_argument("--split", choices=("train", "val"), default="train", help="Dataset split.")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "target_item_classes")
    parser.add_argument("--camera", type=int, default=detect.CAMERA_INDEX)
    parser.add_argument("--display-width", type=int, default=960)
    return parser.parse_args()


def best_candidate(frame):
    candidates = detect.card_candidate_crops(frame, face_boxes=[])
    if not candidates:
        return None
    return max(candidates, key=lambda item: detect.box_area(item[0]))


def main() -> None:
    args = parse_args()
    output_dir = args.data / args.split / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera}.")
    detect.configure_camera(camera)

    saved = len(list(output_dir.glob("*.jpg")))
    print(f"Saving {args.label} samples to {output_dir}")
    print("Press SPACE or s to save the highlighted crop. Press q or Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read camera frame.")
                break

            frame = detect.resize_for_display(frame, detect.PROCESS_WIDTH)
            candidate = best_candidate(frame)
            preview = frame.copy()
            if candidate is not None:
                box, _ = candidate
                x1, y1, x2, y2 = box
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
                status = f"{args.label} {args.split} saved={saved} - SPACE/s save"
            else:
                status = f"{args.label} {args.split} saved={saved} - no card candidate"

            cv2.putText(preview, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
            cv2.imshow("Collect training samples", detect.resize_for_display(preview, args.display_width))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord(" "), ord("s")):
                if candidate is None:
                    print("No card candidate, not saved.")
                    continue
                _, crop = candidate
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = output_dir / f"{args.label}_{stamp}.jpg"
                cv2.imwrite(str(path), crop)
                saved += 1
                print(f"Saved {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
