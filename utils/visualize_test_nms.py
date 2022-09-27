import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from pprint import pprint


# INPUT:
# single_events_dir_path:单一事件流所在目录
# single_events_label_path:单一事件流输出路径

# OUTPUT:
# 输出一个事件流的推理标签

# 对一个（x1,y1,x2,y2,score）的np数组 列表使用nms算法
def nms(cls_detect_array, threshold=0.5):
    # INPUT:
    # cls_detect_array: 模型检测出的“一个检测类”框体的np.array:（x1,y1,x2,y2,score）
    # threshold: IoU交并比筛选阈值，高于threshold的删除
    # OUTPUT:
    # 返回一个图片的“一个检测类”检测框体的，经过nms处理的np.array:（x1,y1,x2,y2,score）

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
        x1, y1, x2, y2 = cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[i][2], cls_detect_array[i][3]
        high_area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 将高分框与其他低分框进行交并比计算,若合并则置信度设为0，遍历其余，最后重新按置信度从大到小排序，保持原数组长度不变，最后最后提取非0置信度的元素
        j = i + 1
        while j < len(cls_detect_array):

            if cls_detect_array[j][4] == 0:
                break

            # 获取其他框体坐标
            x3, y3, x4, y4 = cls_detect_array[j][0], cls_detect_array[j][1], cls_detect_array[j][2], \
                             cls_detect_array[j][3]
            low_area = (x4 - x3 + 1) * (y4 - y3 + 1)

            # 计算相交部分框体坐标，若无则返回0
            # 感觉有点问题，
            and_x1, and_y1, and_x2, and_y2 = np.maximum(x1, x3), np.maximum(y1, y3), np.minimum(x2, x4), np.minimum(y2,
                                                                                                                    y4)
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


# 对一个文件夹内的所有图片进行相对坐标提取，并使用nms
def batch_inference(imgs_dir_path, out_labels_dir_path, visuallized_dir_path):
    # 读取配置
    config_file = '../work_dir_custom/customformat.py'
    checkpoint_file = '../work_dir_custom/epoch_3.pth'
    device = 'cuda'
    if not os.path.exists(out_labels_dir_path):
        os.mkdir(out_labels_dir_path)

    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    template = cv2.imread('../test_data/template.png')
    h, w = template.shape[:2]
    method = eval('cv2.TM_CCOEFF')

    # 读取所有图片并提取信息，遍历图片
    img_list = os.listdir(imgs_dir_path)
    for img in img_list:

        # 设置单张图片输入输出路径
        img_path = os.path.join(imgs_dir_path, img)
        out_label_path = os.path.join(out_labels_dir_path, img[:-3] + 'txt')
        result = inference_detector(model, img_path)
        box_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        # 匹配圆心
        img_cv = cv2.imread(img_path)
        pictured_img = img_cv.copy()
        res = cv2.matchTemplate(img_cv, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 提取圆心绝对坐标
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角坐标元组
        cv2.rectangle(pictured_img, top_left, bottom_right, (255, 255, 0), 2)

        # 画圆心
        circle_centerX = (top_left[0] + bottom_right[0]) / 2
        circle_centerY = (top_left[1] + bottom_right[1]) / 2
        circle_center = (int(circle_centerX), int(circle_centerY))  # 圆心坐标
        cv2.circle(pictured_img, circle_center, 1, (255, 255, 0), 2)

        # 处理本图片所有瑕疵分类
        lines = []  # 每张图片的标签读取为lines列表
        # 对单个图片进行推理结果分析
        for i in range(len(result)):

            boxes_info = nms(result[i], threshold=0.5)  # 单个图片的每个分类先使用nms筛选

            # 第i类别，若无检测则退出。用j遍历所有瑕疵框
            # 下面的循环结束后，得到的是本图片第i个分类下的置信度较高，且使用过nms算法的检测框列表
            for j in range(len(result[i])):

                # 筛选置信度大于0.2的；
                # if result[i][j][4] < 0.1 or j > 6: 若同一类瑕疵超过6个也停止
                if boxes_info[j][4] < 0.2 or j > 3:
                    break  # 当前类停止统计

                # 第j个检测框的所有信息（左上角绝对坐标x1,左上角绝对坐标y1,左上角绝对坐标x2,左上角绝对坐标y2,置信度）
                cls_info = boxes_info[j]
                x1, y1, x2, y2, score = cls_info[0], cls_info[1], cls_info[2], cls_info[3], cls_info[4]

                # 画瑕疵框
                top_left = (int(x1), int(y1))
                bottom_right = (int(x2), int(y2))
                cv2.rectangle(pictured_img, top_left, bottom_right, box_colors[i], 2)

                # 计算得到举办方要求的相对坐标格式（瑕疵类别，瑕疵中心相对坐标x,瑕疵中心相对坐标,瑕疵框W,瑕疵框H,置信度）
                print('框体坐标:({},{}),({},{})'.format(int(x1), int(y1), int(x2), int(y2)))
                cls_X = int((x1 + x2) / 2)
                cls_Y = int((y1 + y2) / 2)
                cls_W = int(abs(x1 - x2) + 0.5)
                cls_H = int(abs(y1 - y2) + 0.5)
                relativeX = int(cls_X - circle_centerX)
                relativeY = int(cls_Y - circle_centerY)
                lines.append(
                    str(i + 1) + "," + str(relativeX) + ',' + str(relativeY) + ',' + str(cls_W) + ',' + str(
                        cls_H) + ',' + str(score))

        # 对单个图片提取出来的标注信息进行打印，便于观察
        pprint(lines)
        print()

        # 覆盖写入本地文件，且文件名与图片相对应
        out = ''
        if len(lines) == 0:
            out += 'Perfect'
        else:
            for line in lines:
                out += (line + "\n")
        with open(out_label_path, 'w', encoding="utf-8") as f:
            f.write(out)
            f.close()

        # 保存筛选过的检测框图片
        if not os.path.exists(visuallized_dir_path):
            os.mkdir(visuallized_dir_path)
        cv2.imwrite(os.path.join(visuallized_dir_path, img), pictured_img)


batch_inference('../result/only_circles', '../result/label', '../result/visualized')
