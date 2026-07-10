import os

# 定义图片和标签文件的路径（注意：您输入的"labels"可能是"labels"的拼写错误，若实际文件夹名不同请修改）
images_dir = "DATABASES/diy_database2/train/images"
labels_dir = "DATABASES/diy_database2/train/labels"  # 若实际是"labels"请修改此处


def delete_unmatched_images():
    # 检查文件夹是否存在
    if not os.path.exists(images_dir):
        print(f"图片文件夹不存在: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"标签文件夹不存在: {labels_dir}")
        return

    # 遍历图片文件夹中的所有文件
    for image_filename in os.listdir(images_dir):
        # 构建完整的图片路径
        image_path = os.path.join(images_dir, image_filename)

        # 只处理文件（跳过文件夹）
        if not os.path.isfile(image_path):
            continue

        # 获取图片文件名（不含后缀）
        image_name = os.path.splitext(image_filename)[0]

        # 构建对应的标签文件路径
        label_filename = f"{image_name}.txt"
        label_path = os.path.join(labels_dir, label_filename)

        # 检查标签文件是否存在，不存在则删除图片
        if not os.path.exists(label_path):
            try:
                os.remove(image_path)
                print(f"已删除无对应标签的图片: {image_path}")
            except Exception as e:
                print(f"删除图片失败 {image_path}: {e}")


if __name__ == "__main__":
    # 执行前提醒用户确认
    print("注意：此操作将删除所有没有对应txt标签的图片，建议先备份数据！")
    confirm = input("是否继续？(y/n): ").strip().lower()
    if confirm == "y":
        delete_unmatched_images()
        print("处理完成")
    else:
        print("已取消操作")
