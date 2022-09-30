import os


def del_wrong(txt_path, ori_dir):
    # 提取所有应当删除的图像名字
    del_names = []
    f = open(txt_path, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        info = line.strip().split('\\')

        basename = info[-1][5:]
        eve_name = info[-2]
        realname = eve_name + basename
        del_names.append(realname)

    img_dir = os.path.join(ori_dir, 'data')
    label_dir = os.path.join(ori_dir, 'label')

    img_list = os.listdir(img_dir)

    i = 0
    for img_name in img_list:
        if img_name in del_names:
            i += 1
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name[:-3] + 'txt')
            os.remove(img_path)
            os.remove(label_path)
            print(str(i) + '号:', img_name)


# del_wrong('wrong_label.txt', 'val')
del_wrong('wrong_label.txt', 'train')
