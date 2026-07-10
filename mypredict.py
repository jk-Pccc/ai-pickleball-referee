from ultralytics import YOLO

if __name__ == "__main__":
    # 加载final4最优模型，延续已有能力
    model = YOLO(r"runs/detect/train84_pk_merged_final4/weights/best.pt")

    # 合并后数据集路径（原数据集+视频数据集混合）
    merged_data_path = r"datasets/Ai-pickleball.yolov11-NEW2/data.yaml"

    # 核心训练参数（适配合并数据集，平衡静态+动态场景）
    model.train(
        data=merged_data_path,
        epochs=12,  # 微调12轮：合并数据集后微调适当轮次
        imgsz=720,  # 保持与final4一致，适配视频帧
        batch=4,
        cache=False,
        workers=0,
        val=True,
        # 1. 学习率与优化器：低速率微调，保护原能力
        optimizer="AdamW",
        lr0=0.0015,  # 初始学习率：仅微调表层参数，不破坏final4已有特征
        lrf=0.004,  # 最终学习率降低，强化收敛稳定性
        cos_lr=True,
        weight_decay=0.0012,  # 小幅提升权重衰减：抑制合并数据集的过拟合
        # 2. 数据增强：平衡静态与动态场景
        augment=True,
        # 降低增强强度，避免过度偏向动态场景
        hsv_h=0.025,  # 比final4原参数低：原数据集静态图无需极端色调增强
        hsv_s=0.85,
        hsv_v=0.55,
        degrees=10.0,  # 降低旋转：原数据集静态球姿态稳定，避免过度增强
        translate=0.22,  # 降低平移：平衡静态图的位置稳定性与动态图的移动性
        scale=0.75,  # 降低尺度波动：避免原数据集静态球的尺度适配受影响
        shear=1.0,  # 降低剪切：平衡静态图的清晰度与动态图的模糊性
        # 保留针对性增强（解决视频场景短板）
        perspective=0.003,
        flipud=0.25,
        mixup=0.12,  # 降低混合样本比例：避免静态图特征被过度混合
        cutmix=0.08,  # 降低剪切混合：平衡静态无遮挡与动态遮挡场景
        # 3. 模型保护与泛化优化
        freeze=[0, 1],  # 冻结backbone前2层：仅训练头部和浅层，避免原能力退化
        patience=10,  # 早停阈值：10轮val无提升则停止，避免无效训练
        multi_scale=True,  # 保持多尺度：适配合并数据集中的静态大球与动态小球
        close_mosaic=3,  # 最后3轮关闭马赛克：精细化静态+动态球的细节特征
        # 4. 损失权重：针对性优化分类（合并后分类干扰增加）
        box=7.5,
        cls=1.1,  # 小幅提升分类权重：合并数据集后背景多样性增加，需强化分类区分
        dfl=1.5,
        # 5. 输出与验证：聚焦视频场景效果
        name="train84_pk_merged_final",  # 合并数据集最终版
        save=True,
        save_period=-1,
        verbose=True,
    )

    # 训练后验证：测试不同数据集的精度，确保无退化
    print("\n训练完成！开始验证不同场景精度...")

    # 测试合并数据集的val集
    merged_val_result = model.val(data=merged_data_path, imgsz=720, batch=4)

    # 使用原数据集进行额外验证
    orig_data_path = r"datasets/Ai-pickleball-referee.v5-all-train.yolov11-NEW/data.yaml"
    orig_val_result = model.val(data=orig_data_path, imgsz=720, batch=4)

    print(f"合并数据集val集mAP50: {merged_val_result.box.map50:.4f}")
    print(f"原数据集val集mAP50: {orig_val_result.box.map50:.4f}")
