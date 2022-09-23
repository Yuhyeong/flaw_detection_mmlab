import cv2 as cv

img = cv.imread('../test_data/21.png')
template = cv.imread('../test_data/template.png')
h, w = template.shape[:2]
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED']

# 分别使用以上三种方法进行检测
for meth in methods:
    img2 = img.copy()

    # 匹配方法在methods数组中的的序号
    method = eval(meth)
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)


    # 左上角坐标元组
    top_left = max_loc
    # 右下角坐标元组
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv.rectangle(img2, top_left, bottom_right, 255, 2)
    # 显示出来
    cv.imwrite(meth + str('_rec.png'), img2)

