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
etalon_straight = cv.imread('etalon_straight.jpg')
etalon_stop = cv.imread('etalon_stop.jpg')
etalon_not_enter = cv.imread('etalon_not_enter.jpg')
etalon_road_works = cv.imread('etalon_road_works.jpg')
etalon_main_road = cv.imread('etalon_main_road.jpg')
etalon_krug = cv.imread('etalon_krug.jpg')
etalon_zebra = cv.imread('etalon_zebra.jpg')

park_detector = db.simple_object_detector("parking_sign.svm")
stop_detector = db.simple_object_detector("fullStopDetector.svm")
no_drive_detector = db.simple_object_detector("fullNoDriveDetector.svm")