import cv2
import os
import math


# INPUT:
# ori_imgs_dir：所有图片所在文件夹
# circles_dir：圆片关键内容输出文件夹


# OUTPUT:
# 将中间格式标签存入annotation_path（pkl文件）

# 输入事件流所在文件夹ori_imgs_dir，输出到out_circles_dir
def extract_circle(ori_imgs_dir, out_circles_dir):
    # 若目录不存在则创建文件夹
    if not os.path.exists(out_circles_dir):
        os.mkdir(out_circles_dir)

    # 设置圆心提取使用的变量
    template = cv2.imread('../test_data/template_new.png')
    h, w = template.shape[:2]
    large_radium = 356
    small_radium = 95

    # 遍历原图
    img_list = os.listdir(ori_imgs_dir)
    for img_name in img_list:
        print(img_name)

        img_path = os.path.join(ori_imgs_dir, img_name)
        out_cirle_path = os.path.join(out_circles_dir, img_name)
        img = cv2.imread(img_path)

        # 提取圆心
        res = cv2.matchTemplate(img, template, eval('cv2.TM_CCOEFF'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角坐标元组
        circle_centerX = (top_left[0] + bottom_right[0]) / 2
        circle_centerY = (top_left[1] + bottom_right[1]) / 2
        circle_center = (int(circle_centerX), int(circle_centerY))  # 圆心坐标

        # 提取圆片box
        box_left = max(0, int(circle_centerX - large_radium))
        box_right = min(1280, int(circle_centerX + large_radium))
        box_top = max(0, int(circle_centerY - large_radium))
        box_bottom = min(800, int(circle_centerY + large_radium))
        circle_box_img = img[box_top:box_bottom, box_left:box_right, :]
        circle_h, circle_w = circle_box_img.shape[:2]

        if circle_h!=circle_w:
            continue

        # 在圆片box上提取圆心
        res = cv2.matchTemplate(circle_box_img, template, eval('cv2.TM_CCOEFF'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角坐标元组
        circle_centerX = (top_left[0] + bottom_right[0]) / 2
        circle_centerY = (top_left[1] + bottom_right[1]) / 2
        center = (int(circle_centerX), int(circle_centerY))

        # 遮盖大圆，（大圆半径-21，大圆半径加201 ）
        # 等价于在大圆半径
        # 大圆粗线画法
        paint_radium = large_radium + 90
        cv2.circle(circle_box_img, center, paint_radium, 0, 222)

        # 遮盖小圆
        # 小圆粗线画法
        cv2.circle(circle_box_img, center, int(small_radium / 2), 0, int(small_radium + 10))

        # # 提取圆心
        # cv2.circle(circle_box_img, center, large_radium, (100,200,100), 2)

        cv2.imwrite(out_cirle_path, circle_box_img)

def extract_circle_and_modify_label(ori_imgs_dir, out_circles_dir, out_label_dir):
    # 若目录不存在则创建文件夹
    if not os.path.exists(out_circles_dir):
        os.mkdir(out_circles_dir)

    if not  os.path.join(out_label_dir):
        os.mkdir(out_label_dir)

    # 设置圆心提取使用的变量
    template = cv2.imread('../test_data/template_new.png')
    h, w = template.shape[:2]
    large_radium = 356
    small_radium = 95

    # 遍历原图
    img_list = os.listdir(ori_imgs_dir)
    for img_name in img_list:
        print(img_name)

        img_path = os.path.join(ori_imgs_dir, img_name)
        out_cirle_path = os.path.join(out_circles_dir, img_name)
        img = cv2.imread(img_path)

        # 提取圆心
        res = cv2.matchTemplate(img, template, eval('cv2.TM_CCOEFF'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角坐标元组
        circle_centerX = (top_left[0] + bottom_right[0]) / 2
        circle_centerY = (top_left[1] + bottom_right[1]) / 2
        circle_center = (int(circle_centerX), int(circle_centerY))  # 圆心坐标

        # 提取圆片box
        box_left = max(0, int(circle_centerX - large_radium))
        box_right = min(1280, int(circle_centerX + large_radium))
        box_top = max(0, int(circle_centerY - large_radium))
        box_bottom = min(800, int(circle_centerY + large_radium))
        circle_box_img = img[box_top:box_bottom, box_left:box_right, :]
        circle_h, circle_w = circle_box_img.shape[:2]

        if circle_h!=circle_w:
            continue

        # 在圆片box上提取圆心
        res = cv2.matchTemplate(circle_box_img, template, eval('cv2.TM_CCOEFF'))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc  # 左上角坐标元组
        bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角坐标元组
        circle_centerX = (top_left[0] + bottom_right[0]) / 2
        circle_centerY = (top_left[1] + bottom_right[1]) / 2
        center = (int(circle_centerX), int(circle_centerY))

        # 遮盖大圆，（大圆半径-21，大圆半径加201 ）
        # 等价于在大圆半径
        # 大圆粗线画法
        paint_radium = large_radium + 90
        cv2.circle(circle_box_img, center, paint_radium, 0, 222)

        # 遮盖小圆
        # 小圆粗线画法
        cv2.circle(circle_box_img, center, int(small_radium / 2), 0, int(small_radium + 10))

        # # 提取圆心
        # cv2.circle(circle_box_img, center, large_radium, (100,200,100), 2)

        cv2.imwrite(out_cirle_path, circle_box_img)

def extract_all_events_denoise(all_events_dir_path, all_new_events_dir_path):

    if not os.path.exists(all_new_events_dir_path):
        os.mkdir(all_new_events_dir_path)

    # 遍历所有事件流
    events_list = os.listdir(all_events_dir_path)
    for events_name in events_list:

        # 事件流所在文件夹
        events_dir_path = os.path.join(all_events_dir_path,events_name)
        # 输出新事件流所在文件夹
        new_events_dir_path = os.path.join(all_new_events_dir_path,events_name)

        if not os.path.exists(new_events_dir_path):
            os.mkdir(new_events_dir_path)

        extract_circle(events_dir_path, new_events_dir_path)



extract_all_events_denoise('../test_data/denoise_output_gaussian','../result/denoise_circle')
# extract_circle('../datasets/test/data', '../result/only_circles')
