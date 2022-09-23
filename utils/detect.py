import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from pprint import pprint

config_file = '../work_dir_custom/customformat.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../work_dir_custom/epoch_3.pth'
device = 'cpu'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
img = '../datasets/test/data/011.jpg'
result = inference_detector(model, img)
class0 = result[0]
class1 = result[1]
class2 = result[2]
lines = []

if len(result[0]) != 0:
    class1_max_index = np.argmax(result[0], axis=0)[0]
    class1_info = result[0][class1_max_index]
    class1_X = int((class1_info[1] + class1_info[3]) / 2)
    class1_Y = int((class1_info[2] + class1_info[4]) / 2)
    class1_W = int(abs(class1_info[1] - class1_info[3]))
    class1_H = int(abs(class1_info[2] - class1_info[4]))
    lines.append("1," + str(class1_X) + ',' + str(class1_Y) + ',' + str(class1_W) + ',' + str(class1_H))
if len(result[1]) != 0:
    class2_max_index = np.argmax(result[1], axis=0)[0]
    class2_info = result[1][class2_max_index]
    class2_X = int((class2_info[1] + class2_info[3]) / 2)
    class2_Y = int((class2_info[2] + class2_info[4]) / 2)
    class2_W = int(abs(class2_info[1] - class2_info[3]))
    class2_H = int(abs(class2_info[2] - class2_info[4]))
    lines.append("2," + str(class2_X) + ',' + str(class2_Y) + ',' + str(class2_W) + ',' + str(class2_H))
if len(result[2]) != 0:
    class3_max_index = np.argmax(result[2], axis=0)[0]
    class3_info = result[2][class3_max_index]
    class3_X = int((class3_info[1] + class3_info[3]) / 2)
    class3_Y = int((class3_info[2] + class3_info[4]) / 2)
    class3_W = int(abs(class3_info[1] - class3_info[3]))
    class3_H = int(abs(class3_info[2] - class3_info[4]))
    lines.append("3," + str(class3_X) + ',' + str(class3_Y) + ',' + str(class3_W) + ',' + str(class3_H))

out = ''
for line in lines:
    out += line + "\r\n"

print(out)

model.show_result(img, result)
model.show_result(img, result, out_file='../result/011.jpg')
