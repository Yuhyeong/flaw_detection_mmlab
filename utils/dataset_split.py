import os
import shutil


def split2val(train_dir, val_dir):
    img_list = os.listdir(os.path.join(train_dir, 'data'))
    label_list = os.listdir(os.path.join(train_dir, 'label'))

    train_img_dir = os.path.join(train_dir, 'data')
    train_label_dir = os.path.join(train_dir, 'label')


    val_img_dir = os.path.join(val_dir, 'data')
    val_label_dir = os.path.join(val_dir, 'label')

    i = 0
    for img_name in img_list:
        label_name = img_name[:-3] + 'txt'

        if i % 50 == 0:
            if os.path.exists(os.path.join(train_label_dir, label_name)):
                shutil.move(os.path.join(train_img_dir, img_name), val_img_dir)
                shutil.move(os.path.join(train_label_dir, label_name), val_label_dir)

        i += 1

split2val('../datasets/train', '../datasets/val')
