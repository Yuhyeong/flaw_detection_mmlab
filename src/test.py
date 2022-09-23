import os
import cv2
from mmdet.apis import init_detector, inference_detector
from pprint import pprint


# INPUT:
# imgs_dir_path:图片集所在目录
# out_labels_dir_path:输出标签所在目录

# OUTPUT:
# 输出一堆图片的推理标签


# 对一个文件夹内的所有图片进行相对坐标提取
def batch_inference(imgs_dir_path, out_labels_dir_path):
    # 读取配置
    config_file = '../work_dir_custom/customformat.py'
    checkpoint_file = '../work_dir_custom/batch1_12.pth'
    device = 'cpu'

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

        # 匹配圆心
        img_cv = cv2.imread(img_path)
        res = cv2.matchTemplate(img_cv, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 提取圆心
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角坐标元组
        circle_centerX = (top_left[0] + bottom_right[0]) / 2
        circle_centerY = (top_left[1] + bottom_right[1]) / 2

        # 处理本图片所有瑕疵分类
        lines = []  # 每张图片的标签读取为lines列表
        # 对单个图片进行推理结果分析
        for i in range(len(result)):

            # 第i类别，若无检测则退出。
            for j in range(len(result[i])):

                if result[i][j][4] < 0.95 or j > 1:
                    break  # 当前类停止统计，筛选置信度大于0.5的；若同一类瑕疵超过两个也停止

                cls_info = result[i][j]
                x1, y1, x2, y2, score = cls_info[0], cls_info[1], cls_info[2], cls_info[3], cls_info[4]

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


batch_inference('../datasets/test/data', '../result/label')
