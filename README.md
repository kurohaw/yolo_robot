# YOLO 机器人比赛识别程序

这是用于 Ubuntu 部署的 YOLO 九种物品识别程序。

## 项目文件

```text
yolo_robot/
├── detect.py
├── requirements.txt
└── runs/classify/target_item_classes/weights/best.pt
```

## Ubuntu 环境配置

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## 运行程序

```bash
python detect.py
```

## 比赛使用说明

- 摄像头与识别卡片之间保持约 20 cm 的距离。
- 摄像头应使用真实的 1920×1080 分辨率，程序启动时会输出实际分辨率。
- 每次只展示一张方形物品卡，并尽量放在画面中央。
- 保持光线均匀，避免反光，等待摄像头完成自动对焦后再判断识别结果。
- 等待画面中的类别标签稳定后，再更换下一张物品卡。

程序会先检测当前方形卡片并进行透视矫正，然后将其识别为以下九种物品之一：

- 电视
- 空调
- 卫生纸
- 衣服
- 香蕉
- 橙子
- 苹果
- 牙刷
- 冰箱

## 摄像头故障排查

如果程序无法打开摄像头，请先检查摄像头设备：

```bash
ls /dev/video*
```

然后根据实际设备编号修改 `detect.py` 中的 `CAMERA_INDEX`，例如改为 `1` 或 `2`。
