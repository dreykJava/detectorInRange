import cv2 as cv
import numpy as np

blue = ((149, 126, 90), (165, 255, 255))
yel = ((34, 83, 50), (48, 255, 255))
red = ((240, 110, 47), (360, 255, 255), (20))

img_rgb = cv.imread("etalon_krug.jpg")
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV_FULL)
mask = cv.inRange(img, red[0], red[1])
mask_down = cv.inRange(img, (0, red[0][1], red[0][2]), (red[2], red[1][1], red[1][2]))
mask = cv.bitwise_or(mask, mask_down)

img_rgb1 = cv.imread("etalon_stop.jpg")
img1 = cv.cvtColor(img_rgb1, cv.COLOR_BGR2HSV_FULL)
mask1 = cv.inRange(img1, red[0], red[1])
mask_down1 = cv.inRange(img1, (0, red[0][1], red[0][2]), (red[2], red[1][1], red[1][2]))
mask1 = cv.bitwise_or(mask1, mask_down1)

cv.imshow("img", img_rgb)
cv.imshow("img2hsv", img)
cv.imshow("mask", mask)

cv.imshow("img1", img_rgb1)
cv.imshow("img2hsv1", img1)
cv.imshow("mask1", mask1)

comp = np.zeros((64, 64))
mask = cv.resize(mask, (64, 64))
mask1 = cv.resize(mask1, (64, 64))
comp[mask == mask1] = 1
print(np.sum(comp))

cv.waitKey()