import os
import mmcv
from pprint import pprint
import numpy as np


# 待办：检查每个图片的ann里面是不是只能包含一个标签


# INPUT:
# imgs_path：所有图片所在文件夹
# labels_path：所有标签所在文件夹（类别，瑕疵中心绝对坐标X,瑕疵中心绝对坐标Y,瑕疵框宽W,瑕疵框高H）
# annotation_path:中间格式数据注释所在路径

# OUTPUT:
# 将中间格式标签存入annotation_path

def convert_label_to_midlle(imgs_path, labels_path, annotation_path):
    CLASSES = ('1', '2', '3')
    # 类别反查表
    cat2label = {k: i for i, k in enumerate(CLASSES)}

    # 所有图像和标注的信息存储在一个列表中
    data_infos = []

    # 依照图片名处理标签文件
    img_list = os.listdir(imgs_path)
    for img_name in img_list:

        img_path = os.path.join(imgs_path, img_name)  # 获取图片路径
        label_path = os.path.join(labels_path, img_name.split('.')[0] + '.txt')  # 获取图片对应的标签路径
        if not os.path.exists(label_path):
            continue

        img = mmcv.imread(img_path)  # 读取图片
        height, width = img.shape[:2]  # 读取图片宽高

        label_file = open(label_path, "r", encoding='utf-8')  # 按行当前图片读取label.txt
        label_lines = label_file.readlines()  # 获取所有标签行

        if label_lines[0].split(",")[0] == 'Perfect':  # 若为Perfect则跳过本图片
            print(img_name)
            continue  # 考虑要不要删除本图片

        # 单张图像的信息存储在字典中
        # w,h默认1280*800
        # 单图片anno增加filename、width、height
        data_info = dict(filename=img_name, width=width, height=height)

        # anno种的ann部分
        gt_labels = []
        gt_bboxes = []

        # 对当前图片的每一标签行进行处理
        for line in label_lines:
            all_factors = line.split(",")  # 对单一标签的内容进行分解
            bbox = []  # 单一检测框的坐标信息

            gt_labels.append(cat2label[all_factors[0]])  # 读取单一检测类名字

            # 处理[中心点绝对坐标x,中心点绝对坐标y,宽H,高W]---->[左上角绝对坐标x,左上角绝对坐标y，右下角绝对坐标x,右下角绝对坐标y]
            centerX, centerY = float(all_factors[1]), float(all_factors[2])
            boxW, boxH = int(all_factors[3].split('.')[0]), int(all_factors[4].split('.')[0])
            x1, y1 = centerX - float(boxW / 2), centerY - float(boxH / 2)  # 左上角
            x2, y2 = centerX + float(boxW / 2), centerY + float(boxH / 2)  # 右下角
            bbox.append(x1)
            bbox.append(y1)
            bbox.append(x2)
            bbox.append(y2)

            gt_bboxes.append(bbox)
            bbox = []

        # 将标注信息（坐标和标签）转换为nparray
        data_anno = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=np.int64))

        data_info.update(ann=data_anno)
        # 所有图像和标注的信息存储在一个列表中
        data_infos.append(data_info)

    pprint(data_infos)
    mmcv.dump(data_infos, annotation_path)


convert_label_to_midlle('../datasets/train/data', '../datasets/train/label', '../datasets/train.pkl')
convert_label_to_midlle('../datasets/val/data', '../datasets/val/label', '../datasets/val.pkl')