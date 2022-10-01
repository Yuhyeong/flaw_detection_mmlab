import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pprint import pprint


########################################################
########################################################
#   用于推理坐标为（x,y,x,y,score）格式的模型
#   如：faster_rcnn_r50_fpn_1x_coco.py
#      cascade_rcnn_r101_fpn_1x_coco.py
########################################################
########################################################


# 剔除框体在图像外的
def result_fliter_start(cls_detect_array):
    if len(cls_detect_array) == 0:
        return np.empty(shape=(0, 0))

    for i in range(len(cls_detect_array)):
        x1, y1, x2, y2 = cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[i][2], cls_detect_array[i][3]

        if ((x2 > 706) or (y2 > 706)) or (((x1 < 10) or (y1 < 10))):
            cls_detect_array[i][4] = 0

    cls_detect_array = cls_detect_array[np.argsort(-cls_detect_array[:, 4], ), :]

    if cls_detect_array.size != 0:
        cond = np.where(cls_detect_array[:, 4] > 0.001)
        cls_detect_array = cls_detect_array[cond]  # 剔除0分

    return cls_detect_array


# 对一个（x1,y1,x2,y2,score）的np数组 列表使用nms算法
def nms(cls_detect_array, iou_thresh=0.5, score_thresh=0.2):
    # OUTPUT： 得到对所有框体分别进行nms后的筛选过的框体np数组

    # 若分类检测框数量为空则返回空数np数组
    if len(cls_detect_array) == 0:
        return np.empty(shape=(0, 0))

    # 遍历所有框体进行nms合并
    for i in range(len(cls_detect_array)):

        # 置信度小于0.3的都舍弃
        if cls_detect_array[i][4] < score_thresh:

            if i == 0:
                cls_detect_array = np.empty(shape=(0, 0))
            else:
                return cls_detect_array[:i, :]

            break

        # 从最高分开始获取框体坐标，并计算框体体积
        x1, y1, x2, y2 = cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[i][2], cls_detect_array[i][3]
        high_area = (y2 - y1) * (x2 - x1)

        # 将高分框与其他低分框进行交并比计算,若合并则置信度设为0，遍历其余，最后重新按置信度从大到小排序，保持原数组长度不变，最后最后提取非0置信度的元素
        j = i + 1
        while j < len(cls_detect_array):

            if cls_detect_array[j][4] == 0:
                break

            # 获取其他框体坐标
            x3, y3, x4, y4 = cls_detect_array[j][0], cls_detect_array[j][1], cls_detect_array[j][2], \
                             cls_detect_array[j][3]

            low_area = (y4 - y3) * (x4 - x3)

            # 计算相交部分框体坐标，若无则返回0
            and_x1, and_y1, and_x2, and_y2 = np.maximum(x1, x3), np.maximum(y1, y3), np.minimum(x2, x4), np.minimum(y2,
                                                                                                                    y4)
            and_w, and_h = np.maximum(0, and_x2 - and_x1), np.maximum(0, and_y2 - and_y1)
            and_area = and_w * and_h

            # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框的置信度设为0，并重排
            IoU = and_area / (high_area + low_area - and_area)
            if IoU > iou_thresh:
                cls_detect_array[j][4] = 0

            j += 1

        # 按置信度排序，获得使用nms算法处理后的检测框列表
        cls_detect_array = cls_detect_array[np.argsort(-cls_detect_array[:, 4], ), :]

    if cls_detect_array.size != 0:
        cond = np.where(cls_detect_array[:, 4] > 0.001)
        cls_detect_array = cls_detect_array[cond]  # 剔除0分

    return cls_detect_array


# 左上角点若在周围20个像素里则认为是邻居
def is_neighbor(xi, yi, xj, yj, dist=20):
    if abs(xj - xi) < dist and abs(yj - yi) < dist:
        return True
    else:
        return False


# 筛选结果
def result_fliter_final(cls_detect_array):
    if len(cls_detect_array) == 0:
        return np.empty((0, 5))

    # 每个框体有几个邻居
    neighbors = np.zeros(len(cls_detect_array))
    cls_detect_array = np.array(cls_detect_array)
    # 遍历所有框体
    for i in range(len(cls_detect_array)):

        # 判定与其他框体是否为邻居
        for j in range(len(cls_detect_array)):

            # 若是则邻居加一
            if is_neighbor(cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[j][0],
                           cls_detect_array[j][1]):
                neighbors[i] = neighbors[i] + 1

    # 若不为空的结果
    if len(cls_detect_array) != 0:
        # 小于十个邻居的删除
        cond = np.where(neighbors > 1)
        cls_detect_array = cls_detect_array[cond]  # 小于4个邻居的删除

    return cls_detect_array


# 对一个图片进行推理，得到三类检测列表
def img_inference(img_path, model=None, config_file='', checkpoint_file=''):
    if model == None:
        device = 'cuda'
        model = init_detector(config_file, checkpoint_file, device=device)

    img_result = [[], [], []]

    single_result = inference_detector(model, img_path)

    cls1 = result_fliter_start(np.array(single_result[0]))
    cls2 = result_fliter_start(np.array(single_result[1]))
    cls3 = result_fliter_start(np.array(single_result[2]))

    # 对本图片每个检测类使用nms算法
    cls1 = nms(cls1, iou_thresh=0.05, score_thresh=0.2)
    cls2 = nms(cls2, iou_thresh=0.05, score_thresh=0.2)
    cls3 = nms(cls3, iou_thresh=0.05, score_thresh=0.2)

    # 输入到本图片检测结果
    img_result[0] = cls1.tolist()
    img_result[1] = cls2.tolist()
    img_result[2] = cls3.tolist()

    return img_result


def single_events_inference(events_dir_path, out_label_path, visulized_dir, model=None, config_file='',
                            checkpoint_file='',
                            score_thresh=0.2, ):
    # OUTPUT:
    # 输出一个事件流的推理标签

    # 若不传入模型，则检查是否有模型路径
    if model == None:
        device = 'cuda'
        model = init_detector(config_file, checkpoint_file, device=device)

    box_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
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

    # 同一事件流下每一类的大团圆
    cls1 = nms(np.array(all_cls[0]), iou_thresh=0.05, score_thresh=0.2)
    cls2 = nms(np.array(all_cls[1]), iou_thresh=0.05, score_thresh=0.2)
    cls3 = nms(np.array(all_cls[2]), iou_thresh=0.05, score_thresh=0.2)

    # cls1 = result_fliter_final(cls1)
    # cls2 = result_fliter_final(cls2)
    # cls3 = result_fliter_final(cls3)

    # 添加标签名准备合并输出txt
    cls1 = np.insert(cls1, 0, 1, axis=1)
    cls2 = np.insert(cls2, 0, 2, axis=1)
    cls3 = np.insert(cls3, 0, 3, axis=1)
    lines = np.array(cls1.tolist() + cls2.tolist() + cls3.tolist())

    # 画框画图
    img_path = os.path.join('../datasets/test/data_only30', os.path.basename(events_dir_path) + '.jpg')
    pictured_img = cv2.imread(img_path)
    for line in lines:
        cls, x1, y1, x2, y2 = int(line[0]), line[1], line[2], line[3], line[4]
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        cv2.rectangle(pictured_img, top_left, bottom_right, box_colors[cls - 1], 2)
        cv2.circle(pictured_img, top_left, 1, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(visulized_dir, os.path.basename(events_dir_path) + '.jpg'), pictured_img)

    # (cls,左上绝对x,左上绝对y,右下绝对x,右下绝对y,score)
    # --->(cls,左上相对x,左上相对y,右下相对x,右下相对y,score)
    if len(lines) != 0:
        # 分割后处理坐标为相对坐标
        cls = lines[:, 0]
        pos = np.array(lines)[:, 1:5] - 356
        score = lines[:, 5]
        # 重新拼接
        temp = np.insert(pos, 4, score, axis=1)
        temp = np.insert(temp, 0, cls, axis=1)
        lines = temp

    # 筛选置信度大于0.2的
    if lines.size != 0:
        cond = np.where(lines[:, 5] > score_thresh)
        lines = lines[cond]  # 剔除0分

    # txt每一行内容确定
    ans = []
    for line in lines:
        cls_name = int(line[0])
        centerX = int((line[1] + line[3]) / 2)
        centerY = int((line[2] + line[4]) / 2)
        boxW = int(line[3] - line[1])
        boxH = int(line[4] - line[2])
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
def all_events_inference(all_events_dir_path, out_labels_dir_path, config_file, checkpoint_file, visulized_dir):
    if not os.path.exists(out_labels_dir_path):
        os.mkdir(out_labels_dir_path)
    if not os.path.exists(visulized_dir):
        os.mkdir(visulized_dir)
    device = 'cuda'
    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)

    # 读取事件流列表，对所有事件流进行处理
    events_list = os.listdir(all_events_dir_path)
    for events_name in events_list:
        # 单事件流文件夹
        single_events_path = os.path.join(all_events_dir_path, events_name)
        out_label_path = os.path.join(out_labels_dir_path, events_name + '.txt')
        single_events_inference(single_events_path, out_label_path, visulized_dir, model=model)


# single_events_inference('../result/only_circles', '../result/only_circles.txt',
#                         config_file='../work_dir_cascade_r101/cascade_r101.py', checkpoint_file='../work_dir_cascade_r101/epoch_10.pth')

# all_events_inference('../datasets/test/data', '../result/label', config_file='../work_dir_cascade_r101/cascade_r101.py', checkpoint_file='../work_dir_cascade_r101/epoch_10.pth')
all_events_inference('../datasets/test/data', '../result/label', config_file='../work_dir_faster/faster.py',
                     checkpoint_file='../work_dir_faster/epoch_12.pth', visulized_dir='../result/imgs/faster')