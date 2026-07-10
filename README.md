# 🏓 AI Pickleball Referee — 匹克球 AI 裁判系统

基于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 的匹克球（Pickleball）智能检测、跟踪与自动计分系统。

## ✨ 项目亮点

- 🎯 **匹克球检测**：针对空中高速小球优化，支持远距离、小目标检测
- 🏃 **实时跟踪**：集成 ByteTrack / BoTSORT 多目标跟踪器，解决球断续消失问题
- 📊 **自动计分**：基于球场边界与球轨迹的智能计分系统，支持手动微调
- 🗺️ **小地图**：实时显示球场缩略图、球轨迹与得分区域
- 🎥 **视频推理**：支持视频文件与摄像头实时检测
- 🔧 **完整训练流程**：从数据标注到模型微调的完整 Pipeline

## 🏗️ 项目结构

```
ultralytics-8.3.163/
├── PickleBallTrain.py        # 匹克球模型训练（合并数据集微调）
├── 150Train.py               # 150 轮长周期训练（抗过拟合优化）
├── Shibie.py                 # 球检测 + ByteTrack 跟踪 + 轨迹绘制
├── TEST1.py                  # 自动计分系统（卡尔曼滤波 + 透视变换）
├── TEST7.py                  # 增强版计分系统（小地图 + 交互控制）
├── mypredict.py              # 视频推理脚本
├── mycam.py                  # 摄像头实时检测
├── tiaoModel.py              # 超参数自动搜索（Ray Tune）
├── YOUHUA.py                 # 基础训练脚本
├── pkTrainTest.py            # 训练 + 多数据集交叉验证
├── trainagain.py             # 从预训练模型继续训练
├── dele.py                   # 数据集清理工具（删除无标签图片）
├── court_calibration.txt     # 球场标定坐标
├── TESTVIDEO/                # 测试视频目录
│   ├── pkvideo.mp4
│   ├── pkvideo2.mp4
│   ├── pkvideo3.mp4
│   └── pkvideo4.mp4
├── datasets/                 # 数据集目录
│   ├── Ai-pickleball.yolov11-NEW2/   # 主训练数据集（含自定义锚框）
│   ├── Ai-pickleball-referee.v5-all-train.yolov11/      # 裁判辅助数据集
│   ├── Ai-pickleball-referee.v5-all-train.yolov11-NEW/  # 增强裁判数据集
│   ├── pickleball.v4i.yolov11/       # Roboflow 基础数据集
│   ├── coco8/                        # COCO8 验证数据集
│   └── diy_dataset/                  # 自定义数据集
├── runs/                     # 训练输出目录
├── ultralytics/              # Ultralytics 库源码（含自定义修改）
├── yolo11n.pt / yolo11m.pt   # YOLO11 预训练权重
└── pyproject.toml            # 项目配置
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA（推荐，GPU 加速训练与推理）
- OpenCV

### 安装

```bash
# 克隆项目
git clone https://github.com/JK-PCCC/ai-pickleball-referee.git
cd ai-pickleball-referee

# 安装依赖（以开发模式安装 ultralytics）
pip install -e .
```

### 主要依赖

```
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.6.0
numpy>=1.23.0
matplotlib>=3.3.0
pillow>=7.1.2
scipy>=1.4.1
pandas>=1.1.4
ultralytics-thop>=2.0.0
```

## 📖 使用方法

### 1. 模型训练

#### 基础训练（从 YOLO11 预训练开始）

```bash
python trainagain.py
```

#### 匹克球专项训练（合并数据集微调）

```bash
python PickleBallTrain.py
```

#### 150 轮长周期训练（抗过拟合）

```bash
python 150Train.py
```

> 核心优化：AdamW 优化器 + 余弦学习率调度 + 冻结前 2 层 Backbone + 自定义小目标锚框

#### 超参数自动搜索

```bash
python tiaoModel.py
```

### 2. 视频检测与跟踪

```bash
# 基础视频推理
python mypredict.py

# 匹克球跟踪 + 轨迹绘制
python Shibie.py
```

### 3. 自动计分系统

#### 基础计分（卡尔曼滤波 + 透视变换）

```bash
python TEST1.py
```

#### 增强版计分（推荐）

```bash
python TEST7.py
```

**交互控制键位：**

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `r` | 重置比分与轨迹 |
| `t` | 仅重置轨迹 |
| `+` / `-` | 调整得分区域高度 |
| `m` | 切换小地图显示 |
| `e` | 暂停/恢复自动计分 |
| `a` / `z` | A 队加/减 1 分 |
| `w` / `s` | B 队加/减 1 分 |

### 4. 摄像头实时检测

```bash
python mycam.py
```

## 🏋️ 训练策略详解

### 针对空中小球的核心优化

本项目针对匹克球在空中高速飞行时检测困难的问题，采用了以下策略：

| 优化方向 | 具体措施 |
|----------|----------|
| **锚框优化** | 自定义 3 组锚框，专为 720p 下空中超小球设计（最小 6×8 像素） |
| **多尺度推理** | 使用 `imgsz=[640, 720, 800]` 覆盖不同距离的球 |
| **数据增强** | 高强度 HSV / 旋转 / 透视 / Copy-Paste 增强，模拟空中球运动特征 |
| **冻结策略** | 冻结 Backbone 前 2 层，保护预训练特征，仅训练头部适配 |
| **损失权重** | `box=9.0` 强化小目标定位，`dfl=2.0` 优化边界框回归 |
| **跟踪补全** | ByteTrack / BoTSORT 跟踪器 + `persist=True`，球短暂消失后仍保留 ID |

### 训练参数参考

```python
model.train(
    data="datasets/Ai-pickleball.yolov11-NEW2/data.yaml",
    epochs=150,
    imgsz=720,
    batch=4,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.001,
    cos_lr=True,
    freeze=[0, 1],  # 冻结前 2 层
    box=9.0,  # 强化小目标框定位
    cls=1.5,  # 分类损失权重
    dfl=2.0,  # 分布式焦点损失
    single_cls=True,  # 单类别（匹克球）
    mosaic=0.8,
    mixup=0.15,
    copy_paste=0.2,
    close_mosaic=5,  # 最后 5 轮关闭马赛克
    patience=25,
    amp=True,  # 混合精度训练
)
```

## 📊 数据集

本项目使用多个匹克球数据集进行训练与验证，均来自 Roboflow：

| 数据集 | 说明 | 来源 |
|--------|------|------|
| `Ai-pickleball.yolov11-NEW2` | 主训练集，含自定义锚框标注 | Roboflow |
| `Ai-pickleball-referee.v5-all-train.yolov11` | 裁判辅助标注数据集 | Roboflow |
| `Ai-pickleball-referee.v5-all-train.yolov11-NEW` | 增强版裁判数据集 | Roboflow |
| `pickleball.v4i.yolov11` | 基础匹克球检测数据集 | Roboflow |

数据集格式遵循 Ultralytics YOLO 标准格式，`data.yaml` 配置示例：

```yaml
path: datasets/Ai-pickleball.yolov11-NEW2
train: train/images
val: val/images
nc: 1
names:
  - pickleball
anchors:
  - [0.0083, 0.0111, 0.0139, 0.0208, 0.0222, 0.0306]  # 小尺度（空中球）
  - [0.0389, 0.0500, 0.0625, 0.0833, 0.0972, 0.1319]  # 中尺度
  - [0.1528, 0.2083, 0.2500, 0.3056, 0.3472, 0.4167]  # 大尺度
```

## 🔬 计分算法

自动计分流程：

```
1. 手动标注球场四角 → 透视变换对齐
2. YOLO 检测 + ByteTrack/BoTSORT 跟踪 → 获取球位置
3. 卡尔曼滤波 / 移动平均平滑 → 消除抖动
4. 判断球穿越中线方向 → 识别击球方
5. 球落入对方得分区域 → 自动计分
```

## 🛠️ 工具脚本

| 脚本 | 功能 |
|------|------|
| `dele.py` | 清理无标签的图片文件 |
| `court_calibration.txt` | 球场标定坐标存储 |
| `pkTrainTest.py` | 训练后多数据集交叉验证 |

## 📌 注意事项

- 训练时 `workers=0` 适配 Windows 环境，Linux 可设为 8
- 显存 ≥ 8GB 建议 `batch=4~6`，不足时设 `batch=2` + `accumulate=2`
- 远距离球检测需要降低置信度阈值（`conf=0.2`），但会增加误检
- 首次使用计分系统需手动标注球场四角，支持撤销与重新选择

## 📄 许可证

本项目基于 Ultralytics 框架，遵循 **AGPL-3.0** 许可证。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO11 目标检测框架
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — 多目标跟踪算法
- [Roboflow](https://roboflow.com) — 数据集标注与管理平台