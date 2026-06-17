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

## 重新训练模型

如果出现以下问题，优先补充训练数据并重新训练模型，不建议继续在 `detect.py` 中硬编码规则：

- 香蕉识别不到。
- 橙子容易识别成苹果。
- 某些图需要很久才稳定显示结果。

比赛时仍然是每次只识别 1 张图片。下面的数据集也是“单张卡片分类数据集”，九宫格照片只是为了方便一次性提取 9 张固定图片的基准样本。

训练数据目录结构如下：

```text
datasets/target_item_classes/
├── train/
│   ├── air_conditioner/
│   ├── apple_fruit/
│   ├── banana_fruit/
│   ├── clothes/
│   ├── orange_fruit/
│   ├── refrigerator_appliance/
│   ├── television_set/
│   ├── tissue_paper/
│   └── toothbrush/
└── val/
    ├── air_conditioner/
    ├── apple_fruit/
    ├── banana_fruit/
    ├── clothes/
    ├── orange_fruit/
    ├── refrigerator_appliance/
    ├── television_set/
    ├── tissue_paper/
    └── toothbrush/
```

每类建议至少采集：

- `train`：50 张以上，最好 100 张以上。
- `val`：10 张以上，用来验证模型是否真的变好。
- 香蕉、橙子、苹果要多采一些比赛现场的困难样本，包括 20 cm 距离、不同角度、轻微模糊、偏暗、偏亮、反光场景。
- 冰箱和空调也要多采集低对比度、轻微模糊和远距离样本，因为它们都是白色家电，远距离下容易混淆。
- 对于“识别很慢”的图片，也要加入对应类别的训练集和验证集，因为这通常说明模型置信度不够稳定。

### 采集样本

每次采集一个类别。把对应卡片放到画面中央，按空格或 `s` 保存裁剪后的卡片图，按 `q` 退出。

```bash
python tools/collect_card_samples.py --label banana_fruit --split train
python tools/collect_card_samples.py --label banana_fruit --split val

python tools/collect_card_samples.py --label orange_fruit --split train
python tools/collect_card_samples.py --label orange_fruit --split val

python tools/collect_card_samples.py --label apple_fruit --split train
python tools/collect_card_samples.py --label apple_fruit --split val
```

其他类别也按相同方式采集。类别目录名必须使用上面目录结构中的英文名。

如果已经有一张包含 9 张图片的九宫格照片，也可以直接生成合成训练集：

```bash
python tools/generate_synthetic_dataset.py --source path/to/九宫格照片.png --fruit-multiplier 2 --appliance-multiplier 2 --clean
```

这个脚本会从九宫格中裁出 9 张基准卡片，再生成旋转、亮度、模糊、透视和轻微偏移后的单张卡片样本。水果类会生成更多样本，用于缓解香蕉识别不到、橙子和苹果混淆的问题；空调和冰箱也会生成更多低对比度样本，用于减少二者混淆。

### 训练

默认从当前 `best.pt` 继续微调，输入尺寸使用 `224`，速度更快；如果识别仍不稳，可以改成 `320`。

```bash
python tools/train_classifier.py --data datasets/target_item_classes --epochs 80 --imgsz 224 --batch 16
```

训练完成后先验证：

```bash
python tools/validate_classifier.py --model runs/classify/target_item_classes_retrain/weights/best.pt --data datasets/target_item_classes/val --imgsz 224
```

重点看输出中的混淆项：

- `orange_fruit -> apple_fruit` 越少越好。
- `banana_fruit -> ...` 越少越好。
- `Average predict time` 越低，单张分类越快。

确认验证结果满意后，再替换运行模型：

```bash
cp runs/classify/target_item_classes/weights/best.pt runs/classify/target_item_classes/weights/best_backup.pt
cp runs/classify/target_item_classes_retrain/weights/best.pt runs/classify/target_item_classes/weights/best.pt
```

如果你已经确定本次训练结果可以直接使用，也可以在训练命令后加 `--install`，脚本会自动备份旧权重并替换为新权重。

## 摄像头故障排查

如果程序无法打开摄像头，请先检查摄像头设备：

```bash
ls /dev/video*
```

然后根据实际设备编号修改 `detect.py` 中的 `CAMERA_INDEX`，例如改为 `1` 或 `2`。
