import os
import shutil

# 分离一部分训练集为验证集
def split2val(train_dir, val_dir):

    # 训练集图片、标签目录
    train_img_dir = os.path.join(train_dir, 'data')
    train_label_dir = os.path.join(train_dir, 'label')

    # 训练集图片、标签列表
    img_list = os.listdir(train_img_dir)
    label_list = os.listdir(train_label_dir)

    # 验证集图片、标签目录
    val_img_dir = os.path.join(val_dir, 'data')
    val_label_dir = os.path.join(val_dir, 'label')

    # 遍历训练集，每50个转移1个到训练集
    i = 0
    for img_name in img_list:
        label_name = img_name[:-3] + 'txt'

        if i % 50 == 0:
            if os.path.exists(os.path.join(train_label_dir, label_name)):
                print(img_name)

                shutil.move(os.path.join(train_img_dir, img_name), val_img_dir)
                shutil.move(os.path.join(train_label_dir, label_name), val_label_dir)

        i += 1

split2val('../datasets/train', '../datasets/val')
