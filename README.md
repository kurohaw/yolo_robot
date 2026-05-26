# YOLO Robot Competition Runtime

Ubuntu deployment package for the YOLO 9-item recognition program.

## Files

```text
yolo_robot/
├── detect.py
├── requirements.txt
└── runs/classify/target_item_classes/weights/best.pt
```

## Ubuntu setup

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## Run

```bash
python detect.py
```

If the camera does not open, check the camera device:

```bash
ls /dev/video*
```

Then change `CAMERA_INDEX` in `detect.py` if needed.
