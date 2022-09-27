import os
import cv2
import json
import numpy as np
from mmdet.apis import init_detector, inference_detector
from pprint import pprint


# 对一个（x,y,w,h,score）的np数组 列表使用nms算法
def nms(cls_detect_array, threshold=0.5):
    # INPUT:
    # cls_detect_array: 模型检测出的“一个检测类”框体的np.array:（x,y,w,h,score）
    # threshold: IoU交并比筛选阈值，高于threshold的删除
    # OUTPUT:
    # 返回一个图片的“一个检测类”检测框体的，经过nms处理的np.array:（x,y,w,h,score）

    # OUTPUT： 得到对所有框体分别进行nms后的筛选过的框体np数组

    # 若分类检测框数量为空则返回空数np数组
    if len(cls_detect_array) == 0:
        return np.empty(shape=(0, 0))

    # 遍历所有框体进行nms合并
    for i in range(len(cls_detect_array)):

        if cls_detect_array[i][4] < 0.3:  # 置信度小于0.3的都舍弃

            if i == 0:
                cls_detect_array = np.empty(shape=(0, 0))
            else:
                # cls_detect_array[i][4] = 0
                return cls_detect_array[:i, :]

            break

        # 从最高分开始获取框体坐标，并计算框体体积
        # x1, y1, x2, y2 = cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[i][2], cls_detect_array[i][3]
        x1, y1, w1, h1 = cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[i][2], cls_detect_array[i][3]
        high_area = w1 * h1

        # 将高分框与其他低分框进行交并比计算,若合并则置信度设为0，遍历其余，最后重新按置信度从大到小排序，保持原数组长度不变，最后最后提取非0置信度的元素
        j = i + 1
        while j < len(cls_detect_array):

            if cls_detect_array[j][4] == 0:
                break

            # 获取其他框体坐标
            # x3, y3, x4, y4 = cls_detect_array[j][0], cls_detect_array[j][1], cls_detect_array[j][2], \
            #                  cls_detect_array[j][3]
            x2, y2, w2, h2 = cls_detect_array[j][0], cls_detect_array[j][1], cls_detect_array[j][2], \
                             cls_detect_array[j][3]

            low_area = w2 * h2

            # 计算相交部分框体坐标，若无则返回0
            # 感觉有点问题，
            and_x1, and_y1, and_x2, and_y2 = np.maximum(x1, x2), np.maximum(y1, y2), np.minimum(x1 + w1,
                                                                                                x2 + w2), np.minimum(
                y1 + h1, y2 + h2)

            and_w, and_h = np.maximum(0, and_x2 - and_x1), np.maximum(0, and_y2 - and_y1 + 1)
            and_area = and_w * and_h

            # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框的置信度设为0，并重排
            IoU = and_area / (high_area + low_area - and_area)
            if IoU > threshold:
                cls_detect_array[j][4] = 0

            j += 1

        # 按置信度排序，获得使用nms算法处理后的检测框列表
        cls_detect_array = cls_detect_array[np.argsort(-cls_detect_array[:, 4], ), :]

    if cls_detect_array.size != 0:
        cond = np.where(cls_detect_array[:, 4] > 0.001)
        cls_detect_array = cls_detect_array[cond]  # 剔除0分

    return cls_detect_array


# 对一个图片进行推理，得到三类检测列表
def img_inference(img_path, model=None, checkpoint_file=''):
    # INPUT:
    # img_path:单一图片所在目录
    # model:读取好的模型
    # checkpoint_file：若不传入读取好的模型，则应传入模型路径
    # OUTPUT:
    # 返回一个图片的检测框体List

    # 若不传入模型，则检查是否有模型路径
    if model == None:

        config_file = '../work_dir_custom/customformat.py'
        device = 'cuda'

        # 若无模型且无模型路径
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_custom/epoch_3.pth'
        # 读取配置
        checkpoint_file = '../work_dir_custom/epoch_3.pth'
        # 初始化检测器
        model = init_detector(config_file, checkpoint_file, device=device)

    img_result = [[], [], []]

    single_result = inference_detector(model, img_path)

    # 对本图片每个检测类使用nms算法
    cls1 = nms(single_result[0], threshold=0.2)
    cls2 = nms(single_result[1], threshold=0.2)
    cls3 = nms(single_result[2], threshold=0.2)

    # 输入到本图片检测结果
    img_result[0] = cls1.tolist()
    img_result[1] = cls2.tolist()
    img_result[2] = cls3.tolist()

    return img_result


# 对一个文件夹内（一个事件流）的所有图片进行推理，并使用nms，得到本文件夹所有图片的综合检测结果
def single_events_inference(events_dir_path, out_label_path, model=None, checkpoint_file=''):
    # INPUT:
    # events_dir_path:单一事件流所在目录
    # out_label_path:单一事件流标签综合结果输出路径（用了nms）
    # model:读取好的模型
    # checkpoint_file：若不传入读取好的模型，则应传入模型路径
    # OUTPUT:
    # 输出一个事件流的推理标签

    # 若不传入模型，则检查是否有模型路径
    if model == None:

        config_file = '../work_dir_custom/customformat.py'
        device = 'cuda'

        # 若无模型且无模型路径
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_custom/batch2_9.pth'
        # 读取配置
        checkpoint_file = '../work_dir_custom/batch2_9.pth'
        # 初始化检测器
        model = init_detector(config_file, checkpoint_file, device=device)

    # 本图片所有分类的瑕疵框体
    all_cls = [[], [], []]

    # 读取所有图片并提取信息，遍历图片
    img_list = os.listdir(events_dir_path)
    for img in img_list:
        print(img)
        # 设置单张图片输入输出路径
        img_path = os.path.join(events_dir_path, img)
        imr_result = img_inference(img_path, model)

        # 输入到本事件流全检测中
        all_cls[0] += imr_result[0]
        all_cls[1] += imr_result[1]
        all_cls[2] += imr_result[2]

    cls1 = nms(np.array(all_cls[0]))
    cls2 = nms(np.array(all_cls[1]))
    cls3 = nms(np.array(all_cls[2]))

    cls1 = np.insert(cls1, 0, 1, axis=1)
    cls2 = np.insert(cls2, 0, 2, axis=1)
    cls3 = np.insert(cls3, 0, 3, axis=1)

    # 覆盖写入本地文件，且文件名与图片相对应
    lines = np.array(cls1.tolist() + cls2.tolist() + cls3.tolist())

    # 分割后处理坐标为相对坐标
    cls = lines[:, 0]
    pos = np.array(lines)[:, 1:3] - 356
    w = lines[:, 3]
    h = lines[:, 4]
    score = lines[:, 5]

    # 重新拼接
    temp = np.insert(pos, 0, cls, axis=1)
    temp = np.insert(temp, 3, w, axis=1)
    temp = np.insert(temp, 4, h, axis=1)
    temp = np.insert(temp, 5, score, axis=1)
    lines = temp

    # 筛选置信度大于0.4的
    if lines.size != 0:
        cond = np.where(lines[:, 5] > 0.4)
        lines = lines[cond]  # 剔除0分

    # txt每一行内容确定
    ans = []
    for line in lines:
        cls_name = int(line[0])
        centerX = int(line[1])
        centerY = int(line[2])
        boxW = int(line[3])
        boxH = int(line[4])
        score = line[5]

        ans.append(
            str(cls_name) + ',' + str(centerX) + ',' + str(centerY) + ',' + str(boxW) + ',' + str(boxH) + ',' + str(
                score))

    # ans中内容写入文件
    out = ''
    if len(ans) == 0:
        out += 'Perfect'
    else:
        for line in ans:
            out += (line + "\n")
    with open(out_label_path, 'w', encoding="utf-8") as f:
        f.write(out)
        f.close()


# 对一个文件夹内的所有文件夹中的所有图片（所有事件流）进行推理，并使用nms，得到所有事件流的检测结果
def all_events_inference(all_events_dir_path, out_labels_dir_path):
    if not os.path.exists(out_labels_dir_path):
        os.mkdir(out_labels_dir_path)

    # 读取配置
    config_file = '../work_dir_custom/customformat.py'
    checkpoint_file = '../work_dir_custom/batch2_9.pth'
    device = 'cuda'

    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)

    # 读取事件流列表，对所有事件流进行处理
    events_list = os.listdir(all_events_dir_path)
    for events_name in events_list:
        # 单事件流文件夹
        single_events_path = os.path.join(all_events_dir_path, events_name)
        out_label_path = os.path.join(out_labels_dir_path, events_name + '.txt')
        single_events_inference(single_events_path, out_label_path, model)


single_events_inference('../datasets/test/data/027', '../result/label/027.txt')
# all_events_inference('datasets/test/data', 'result/label')
