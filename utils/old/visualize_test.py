import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pprint import pprint


# INPUT:
# imgs_dir_path:图片集所在目录
# out_labels_dir_path:输出标签所在目录
# visuallized_dir_path:瑕疵可视化后输出目录

# OUTPUT:
# 输出推理标签
# 输出带检测框的图像

# 对一个文件夹内的所有图片进行相对坐标提取，并可视化
def batch_inference(imgs_dir_path, out_labels_dir_path, visuallized_dir_path, mode='', checkpoint_file=''):
    # 读取配置
    if mode == 'faster':
        config_file = '../work_dir_faster/faster.py'
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_faster/epoch_12.pth'
    elif mode == 'cascade':
        config_file = '../work_dir_cascade/cascade.py'
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_cascade/epoch_12.pth'
    else:
        config_file = '../work_dir_custom/customformat.py'
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_custom/epoch_12.pth'

    device = 'cuda'

    if not os.path.exists(out_labels_dir_path):
        os.mkdir(out_labels_dir_path)

    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    template = cv2.imread('../test_data/template_new.png')
    template_h, template_w = template.shape[:2]
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

        # 画小圆方框
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (int(top_left[0] + template_w), int(top_left[1] + template_h))  # 右下角坐标元组
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

            # 第i类别，若无检测则退出。
            for j in range(len(result[i])):

                # 筛选置信度大于0.2的；
                if result[i][j][4] < 0.2 or j > 3:
                    break  # 当前类停止统计，筛选置信度大于0.5的；若同一类瑕疵超过两个也停止

                # 第j个检测框的所有信息（左上角绝对坐标x1,左上角绝对坐标y1,左上角绝对坐标x2,左上角绝对坐标y2,置信度）
                box_info = result[i][j]
                x, y, w, h, score = box_info[0], box_info[1], box_info[2], box_info[3], box_info[4]

                # 画瑕疵框
                top_left = (int(x), int(y))
                bottom_right = (int(x + w), int(y + h))
                cv2.rectangle(pictured_img, top_left, bottom_right, box_colors[i], 2)

                # 计算得到举办方要求的相对坐标格式（瑕疵类别，瑕疵中心相对坐标x,瑕疵中心相对坐标,瑕疵框W,瑕疵框H,置信度）
                print('框体信息:左上角({},{}),宽高({},{})'.format(int(x), int(y), int(w), int(h)))

                # 求框体中心点绝对坐标
                cls_X = int(x + w / 2)
                cls_Y = int(y + h / 2)
                cls_W = int(w)
                cls_H = int(h)

                # 求框体中心点相对坐标
                relativeX = int(cls_X - circle_centerX)
                relativeY = int(cls_Y - circle_centerY)
                lines.append(
                    str(i + 1) + "," + str(relativeX) + ',' + str(relativeY) + ',' + str(cls_W) + ',' + str(
                        cls_H) + ',' + str(score))

        # 对单个图片提取出来的标注信息进行打印，便于观察
        pprint(lines)
        print()

        # 覆盖写入本地文件，且文件名与图片相对应，若无瑕疵则保存Perfect
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


def result_fliter(cls_detect_array):
    if len(cls_detect_array) == 0:
        return np.empty(shape=(0, 0))

    for i in range(len(cls_detect_array)):
        x, y, w, h = cls_detect_array[i][0], cls_detect_array[i][1], cls_detect_array[i][2], cls_detect_array[i][3]

        if (x + w > 716) or (y + h > 716):
            cls_detect_array[i][4] = 0

        cls_detect_array = cls_detect_array[np.argsort(-cls_detect_array[:, 4], ), :]

    if cls_detect_array.size != 0:
        cond = np.where(cls_detect_array[:, 4] > 0.001)
        cls_detect_array = cls_detect_array[cond]  # 剔除0分

    return cls_detect_array


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


# 对一个文件夹内的所有图片进行相对坐标提取，并使用nms，并可视化
def batch_inference_nms(imgs_dir_path, out_labels_dir_path, visuallized_dir_path, mode='', checkpoint_file=''):
    # 读取配置
    if mode == 'faster':
        config_file = '../work_dir_faster/faster.py'
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_faster/epoch_12.pth'
    elif mode == 'cascade':
        config_file = '../work_dir_cascade/cascade.py'
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_cascade/epoch_12.pth'
    else:
        config_file = '../work_dir_custom/customformat.py'
        if checkpoint_file == '':
            checkpoint_file = '../work_dir_custom/epoch_12.pth'

    device = 'cuda'

    if not os.path.exists(out_labels_dir_path):
        os.mkdir(out_labels_dir_path)

    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    template = cv2.imread('../test_data/template_new.png')
    template_h, template_w = template.shape[:2]
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

        # 画小圆方框
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (int(top_left[0] + template_w), int(top_left[1] + template_h))  # 右下角坐标元组
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

            boxes_info = result_fliter(result[i]).tolist()

            boxes_info = nms(boxes_info, threshold=0.2)  # 单个图片的每个分类先使用nms筛选

            # 第i类别，若无检测则退出。用j遍历所有瑕疵框
            # 下面的循环结束后，得到的是本图片第i个分类下的置信度较高，且使用过nms算法的检测框列表
            for j in range(len(boxes_info)):

                # 筛选置信度大于0.2的；
                if boxes_info[j][4] < 0.2 or j > 3:
                    break  # 当前类停止统计

                # 第j个检测框的所有信息（左上角绝对坐标x1,左上角绝对坐标y1,左上角绝对坐标x2,左上角绝对坐标y2,置信度）
                box_info = boxes_info[j]
                x, y, w, h, score = box_info[0], box_info[1], box_info[2], box_info[3], box_info[4]

                # 画瑕疵框
                top_left = (int(x), int(y))
                bottom_right = (int(x + w), int(y + h))
                cv2.rectangle(pictured_img, top_left, bottom_right, box_colors[i], 2)

                # 计算得到举办方要求的相对坐标格式（瑕疵类别，瑕疵中心相对坐标x,瑕疵中心相对坐标,瑕疵框W,瑕疵框H,置信度）
                print('框体信息:左上角({},{}),宽高({},{})'.format(int(x), int(y), int(w), int(h)))

                # 求框体中心点绝对坐标
                cls_X = int(x + w / 2)
                cls_Y = int(y + h / 2)
                cls_W = int(w)
                cls_H = int(h)

                # 求框体中心点相对坐标
                relativeX = int(cls_X - circle_centerX)
                relativeY = int(cls_Y - circle_centerY)
                lines.append(
                    str(i + 1) + "," + str(relativeX) + ',' + str(relativeY) + ',' + str(cls_W) + ',' + str(
                        cls_H) + ',' + str(score))

        # 对单个图片提取出来的标注信息进行打印，便于观察
        pprint(lines)
        print()

        # 覆盖写入本地文件，且文件名与图片相对应，若无瑕疵则保存Perfect
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


batch_inference_nms('../datasets/val/data', '../result/label_val',
                    '../result/visualized_testOnVal', 'faster', '../work_dir_faster/1.5e-9/epoch_12.pth')
# batch_inference_nms('../datasets/val/data', '../result/label',
#                     '../result/visualized_test_fliter', 'cascade', '../work_dir_cascade/1.5e-9/epoch_12.pth')
# batch_inference('../datasets/test/data_only30', '../result/data_only30_label',
#                 '../result/visualized_test')
