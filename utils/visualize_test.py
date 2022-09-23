import os
import cv2
from mmdet.apis import init_detector, inference_detector
from pprint import pprint


# INPUT:
# imgs_dir_path:图片集所在目录
# out_labels_dir_path:输出标签所在目录
# visuallized_dir_path:瑕疵可视化后输出目录

# OUTPUT:
# 输出推理标签
# 输出带检测框的图像

def batch_inference_and_visualize(imgs_dir_path, out_labels_dir_path, visuallized_dir_path):
    # 读取配置
    config_file = '../work_dir_custom/customformat.py'
    checkpoint_file = '../work_dir_custom/batch2_8.pth'
    device = 'cpu'

    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    template = cv2.imread('../test_data/template_new.png')
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

        # 画小圆
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

            # 第i类别，若无检测则退出。
            for j in range(len(result[i])):

                if result[i][j][4] < 0.5 or j > 3:
                    # if result[i][j][4] < 0.1:
                    break  # 当前类停止统计，筛选置信度大于0.5的；若同一类瑕疵超过两个也停止

                # 第j个检测框的所有信息（左上角绝对坐标x1,左上角绝对坐标y1,左上角绝对坐标x2,左上角绝对坐标y2,置信度）
                cls_info = result[i][j]
                x1, y1, x2, y2, score = cls_info[0], cls_info[1], cls_info[2], cls_info[3], cls_info[4]

                # 画瑕疵框
                top_left = (int(x1), int(y1))
                bottom_right = (int(x2), int(y2))
                cv2.rectangle(pictured_img, top_left, bottom_right, box_colors[i], 2)

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


batch_inference_and_visualize('../datasets/test/data', '../result/label', '../result/visualized_test')
