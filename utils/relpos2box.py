import cv2


def rel2box(img_path, label_path):
    img = cv2.imread(img_path)
    pictured_img = img.copy()
    f = open(label_path, 'r', encoding='utf-8')
    lines = f.readlines()
    box_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for line in lines:
        info = line.strip().split(',')
        cls, centerX, centerY, w, h, score = int(info[0]), float(info[1]), float(info[2]), float(info[3]), float(
            info[4]), float(info[5])

        x1, y1, x2, y2 = centerX - w / 2 + 356, centerY - h / 2 + 356, centerX + w / 2 + 356, centerY + h / 2 + 356
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))

        cv2.rectangle(pictured_img, top_left, bottom_right, box_colors[cls - 1], 2)
        cv2.circle(pictured_img, center, 1, (255, 255, 0), 2)

    cv2.imwrite('../result/tes.jpg', pictured_img)


rel2box('../datasets/test/data_only30/001.jpg', '../result/label/001.txt')
