import cv2

img = cv2.imread('../test_data/1.bmp')
img2 = img.copy()

top_left = (int(237.23), int(173.70))
bottom_right = (int(312.33), int(365.33))

cv2.rectangle(img2, top_left, bottom_right, 255, 2)
cv2.imwrite('../result/1.bmp',img2)

