import re
import cv2 as cv
import dlib as db
import numpy as np
import os
import time

blue = ((149, 126, 90), (165, 255, 255))
yel = ((34, 83, 50), (48, 255, 255))
red = ((240, 110, 47), (360, 255, 255), (20))

etalon_park = cv.imread('etalon_park.jpg')
# etalon_park = cv.cvtColor(etalon_park_rgb, cv.COLOR_BGR2HSV_FULL)

etalon_straight = cv.imread('etalon_straight.jpg')
# etalon_straight = cv.cvtColor(etalon_straight_rgb, cv.COLOR_BGR2HSV_FULL)
    
etalon_stop = cv.imread('etalon_stop.jpg')
# etalon_stop = cv.cvtColor(etalon_stop_rgb, cv.COLOR_BGR2HSV_FULL)

etalon_not_enter = cv.imread('etalon_not_enter.jpg')
# etalon_not_enter = cv.cvtColor(etalon_not_enter_rgb, cv.COLOR_BGR2HSV_FULL)

etalon_road_works = cv.imread('etalon_road_works.jpg')
# etalon_road_works = cv.cvtColor(etalon_road_works_rgb, cv.COLOR_BGR2HSV_FULL)

etalon_main_road = cv.imread('etalon_main_road.jpg')
# etalon_main_road = cv.cvtColor(etalon_main_road_rgb, cv.COLOR_BGR2HSV_FULL)

etalon_krug = cv.imread('etalon_krug.jpg')
# etalon_krug = cv.cvtColor(etalon_krug_rgb, cv.COLOR_BGR2HSV_FULL)

etalon_zebra = cv.imread('etalon_zebra.jpg')
# etalon_zebra = cv.cvtColor(etalon_zebra_rgb, cv.COLOR_BGR2HSV_FULL)

park_detector = db.simple_object_detector("parking_sign.svm")
stop_detector = db.simple_object_detector("fullStopDetector.svm")
no_drive_detector = db.simple_object_detector("fullNoDriveDetector.svm")

def getSignCoordinates(img) -> list:
    contours = stop_detector(img)
    contours2 = no_drive_detector(img)

    imgesList = []

    for contour in contours:
        # print(cv.contourArea(contour))
        x, y, x2, y2 = [contour.left(), contour.top(), contour.right(), contour.bottom()]
        cv.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 3)
        imgesList.append(img[y : y2, x : x2])
    
    for contour in contours2:
        x, y, x2, y2 = [contour.left(), contour.top(), contour.right(), contour.bottom()]
        cv.rectangle(img, (x, y), (x2, y2), (0, 150, 0), 3)
        imgesList.append(img[y : y2, x : x2])

    return imgesList

def getMask(img_rgb, color, red = 0):
    try:
        img = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV_FULL)
        mask = cv.inRange(img, color[0], color[1])    

        if red == 1:
            mask_down = cv.inRange(img, (0, color[0][1], color[0][2]), (color[2], color[1][1], color[1][2]))
            mask = cv.bitwise_or(mask, mask_down)

        # cv.imshow('test', mask)

        mask = cv.dilate(mask, np.ones((3, 3)))
        mask = cv.erode(mask, np.ones((3, 3)))

        # cv.imshow('test1', mask)

        return cv.resize(mask, (64, 64))
    except:
        # print(type(img_rgb))
        return np.zeros((64, 64))

etalon_park_mask = getMask(etalon_park, blue)
etalon_krug_mask = getMask(etalon_krug, red, 1)
etalon_stop_mask = getMask(etalon_stop, red, 1)

def compare(mask, color):
    if (color == blue):
        park_comp = np.zeros((64, 64))
        park_comp[mask == etalon_park_mask] = 1

        if np.sum(park_comp) >= 2700:
            print(np.sum(park_comp))
            time.sleep(1)
            print('Знак парковки')
    elif (color == red):
        stop_comp = np.zeros((64, 64))
        krug_comp = np.zeros((64, 64))

        # print("123", mask.shape)
        # print("098", etalon_stop_mask.shape)
        # print()

        # if mask.shape:
        #     print("YES")
        # else:
        #     print("NONONONONONONONNONON")

        stop_comp[mask == etalon_stop_mask] = 1 # много матриц это не хорошо, цените память, молодой человек!
        krug_comp[mask == etalon_krug_mask] = 1

        # теперь будет выводиться не первый похожий, а наиболее похожий
        comp = [('Знак проезда нет', np.sum(krug_comp)), ('Знак стоп', np.sum(stop_comp))]
        comp = sorted(comp, key = lambda x: x[1], reverse=True)
        if comp[0][1] > 2700:
            print(comp[0][0])

        

        # if np.sum(krug_comp) >= 2700 and np.sum(krug_comp) != 3552:
        #     print(np.sum(krug_comp))
        #     time.sleep(1)
        #     print('Знак проезда нет')
        # elif np.sum(stop_comp) >= 2700:
        #     print(np.sum(stop_comp))
        #     time.sleep(1)
        #     print('Знак стоп')
            

capture = cv.VideoCapture(0)

while True:
    ret, frame_rgb = capture.read()
    frame_rgb = cv.resize(frame_rgb, (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2))
    if frame_rgb.any():
        frame_list = getSignCoordinates(frame_rgb)

        for frame in frame_list:
            frameBlue = getMask(frame, blue)
            # cv.imshow("blue", cv.resize(frameBlue, (128, 128)))

            frameRed = getMask(frame, red, 1)
            # cv.imshow("red", cv.resize(frameRed, (128, 128)))

            compare(frameBlue, blue)
            compare(frameRed, red)

        cv.imshow("frame", frame_rgb)
        
    if (cv.waitKey(30)) == 27:
        break

capture.release()
cv.destroyAllWindows()