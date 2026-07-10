# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os
import yaml
from pathlib import Path

if __name__ == "__main__":
    # 1. 数据集配置（核心：确保路径+锚框正确，避免训练报错）
    data_yaml_path = r"datasets/Ai-pickleball.yolov11-NEW2/data.yaml"
    dataset_dir = Path("datasets/Ai-pickleball.yolov11-NEW2")  # 数据集根目录

    # 修复yaml：关键添加【自定义锚框（归一化）】+ 确保路径绝对化
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        # 1.1 路径绝对化（避免相对路径歧义）
        data_config['path'] = str(dataset_dir.resolve())  # 绝对路径，确保YOLO能找到
        data_config['train'] = str((dataset_dir / 'train/images').resolve())
        data_config['val'] = str((dataset_dir / 'val/images').resolve()) if (dataset_dir / 'val/images').exists() else \
        data_config['train']

        # 1.2 添加自定义锚框（适配空中超小球，基于720分辨率归一化）
        # 原像素锚框[[6,8,10,15,16,22], [28,36,45,60,70,95], [110,150,180,220,250,300]] → 归一化÷720
        data_config['anchors'] = [
            [0.0083, 0.0111, 0.0139, 0.0208, 0.0222, 0.0306],  # 小尺度（空中球）
            [0.0389, 0.0500, 0.0625, 0.0833, 0.0972, 0.1319],  # 中尺度
            [0.1528, 0.2083, 0.2500, 0.3056, 0.3472, 0.4167]  # 大尺度
        ]

        # 1.3 确保类别配置正确（单类别匹克球）
        data_config['nc'] = 1
        data_config['names'] = ['pickleball']

        # 保存修复后的yaml
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ 修复data.yaml：已添加锚框+绝对路径，路径：{data_yaml_path}")
    except Exception as e:
        print(f"⚠️  修复yaml失败：{e}，将使用临时配置文件")
        # 备用：创建临时yaml（确保锚框+路径正确）
        temp_yaml_path = "temp_data.yaml"
        temp_config = {
            'path': str(dataset_dir.resolve()),
            'train': str((dataset_dir / 'train/images').resolve()),
            'val': str((dataset_dir / 'val/images').resolve()) if (dataset_dir / 'val/images').exists() else str(
                (dataset_dir / 'train/images').resolve()),
            'nc': 1,
            'names': ['pickleball'],
            'anchors': [
                [0.0083, 0.0111, 0.0139, 0.0208, 0.0222, 0.0306],
                [0.0389, 0.0500, 0.0625, 0.0833, 0.0972, 0.1319],
                [0.1528, 0.2083, 0.2500, 0.3056, 0.3472, 0.4167]
            ]
        }
        with open(temp_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f, default_flow_style=False, sort_keys=False)
        data_yaml_path = temp_yaml_path
        print(f"✅ 创建临时yaml：{temp_yaml_path}")

    # 2. 加载模型（确保路径存在）
    model_path = r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train86_pk_far_small_final43\weights\best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11s.pt"  # 备用：加载YOLO11s预训练模型（小目标检测更优）
    print(f"✅ 加载模型：{model_path}")
    model = YOLO(model_path, task="detect")

    # 3. 150轮训练参数（核心优化：抗过拟合+后期收敛）
    try:
        model.train(
            # 核心数据配置
            data=data_yaml_path,

            # 1. 训练周期与早停（避免无效训练）
            epochs=150,  # 长轮次目标：充分拟合+后期细调
            patience=25,  # 早停阈值：25轮val无提升则停止（长轮次需延长，避免误停）
            save_period=10,  # 每10轮保存一次模型：方便后期选择最优轮次（避免最后一轮过拟合）
            save=True,  # 保存最佳模型（基于val/mAP50-95）

            # 2. 输入与批量（平衡效率与稳定性）
            imgsz=720,  # 与锚框分辨率匹配，适配空中球
            batch=4,  # 若显存≥8G可设为6（提升效率），不足则设2+accumulate=2
            accumulate=1,  # 批量累积：batch=4时无需累积，显存不足时调整
            workers=0,  # 适配Windows，避免多线程报错
            cache=True,  # 启用缓存（16G内存足够，减少重复加载）
            device=0,  # GPU加速（CPU设为"cpu"）
            amp=True,  # 混合精度训练：减少显存占用+提升速度（长轮次必开）
            deterministic=False,  # 关闭确定性：提升训练速度（长轮次可牺牲少量确定性换效率）

            # 3. 优化器与学习率（长轮次收敛关键）
            optimizer="AdamW",  # 对小目标+长轮次更友好，梯度更平滑
            lr0=0.001,  # 初始学习率：避免前期震荡
            lrf=0.001,  # 最终学习率：降至0.001（原0.003过高，长轮次易后期震荡）
            cos_lr=True,  # 余弦学习率：前期快速拟合，后期缓慢下降（长轮次最优调度）
            warmup_epochs=5,  # 热身轮次：前5轮缓慢提升学习率，避免前期过拟合
            warmup_bias_lr=0.1,  # 热身阶段偏置学习率：稳定前期分类损失
            weight_decay=0.0008,  # 权重衰减：强化正则化，抑制过拟合（长轮次核心抗过拟合手段）

            # 4. 冻结策略（保护预训练特征，避免前期破坏）
            freeze=[0, 1],  # 前10轮冻结backbone前2层：先训头部适配数据集，再解冻训深层
            unfreeze_at=10,  # 第10轮开始解冻：逐步学习深层特征，避免过拟合

            # 5. 数据增强（强化泛化，对抗长轮次过拟合）
            augment=True,
            # 5.1 色彩增强：适配空中球光影变化
            hsv_h=0.04,  # 色调波动：模拟不同光线
            hsv_s=0.9,  # 饱和度：强化球与背景区分
            hsv_v=0.6,  # 亮度：应对空中球强光/阴影
            # 5.2 几何增强：模拟空中球运动特征
            degrees=15.0,  # 旋转：覆盖飞行姿态
            translate=0.2,  # 平移：模拟飞行轨迹
            scale=0.9,  # 尺度：0.1~0.9×720，覆盖空中超小球
            shear=2.0,  # 剪切：模拟运动模糊（空中球关键）
            perspective=0.005,  # 透视：模拟摄像头角度变化
            flipud=0.3,  # 垂直翻转：补充上下飞行姿态
            fliplr=0.5,  # 水平翻转：增加对称性
            # 5.3 正则化增强：抗过拟合
            mosaic=0.8,  # 降低马赛克强度：避免小目标被分割
            close_mosaic=5,  # 最后5轮关闭马赛克：细调空中球细节（原2轮不足）
            mixup=0.15,  # 混合样本：增强抗背景干扰
            copy_paste=0.2,  # 复制粘贴：增加空中球样本数量
            erasing=0.2,  # 擦除：抑制过拟合（原0.4过高，易擦除小目标）
            dropout=0.08,  # 启用dropout：随机关闭8%神经元，强化泛化（长轮次必开）

            # 6. 损失权重（优化空中球检测）
            box=9.0,  # 框损失：强化小目标定位
            cls=1.5,  # 分类损失：区分空中球与背景（如天空、灯光）
            dfl=2.0,  # 分布损失：优化边界框回归（空中球定位关键）
            single_cls=True,  # 单类别优化：提升匹克球检测效率

            # 7. 其他配置（监控+输出）
            name="train86_pk_far_small_150ep",  # 区分长轮次训练
            exist_ok=False,  # 不覆盖现有文件夹，避免误删
            val=True,  # 保留验证：实时监控泛化能力（长轮次必开，防过拟合）
            plots=True,  # 生成指标图：便于分析过拟合/收敛情况
            verbose=True,  # 打印日志：实时查看损失+指标变化
        )
        print("🎉 150轮训练完成！最优模型已保存至 runs/detect/train86_pk_far_small_150ep/weights/best.pt")
    except Exception as e:
        print(f"❌ 训练出错：{e}")
        # 紧急备用：若长轮次报错，自动降级为30轮基础训练
        print("🔄 自动降级为30轮训练...")
        model.train(
            data=data_yaml_path,
            epochs=30,
            imgsz=720,
            batch=4,
            workers=0,
            val=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.001,
            cos_lr=True,
            weight_decay=0.0008,
            freeze=[0],
            augment=True,
            name="train86_pk_far_small_30ep_backup",
            save=True,
            amp=True
        )