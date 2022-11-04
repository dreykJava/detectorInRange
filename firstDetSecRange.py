import cv2 as cv
import dlib as db
import numpy as np
import os
import time

key_color = {
    'blue' : ((149, 126, 90), (165, 255, 255)),
    'yellow' : ((34, 83, 50), (48, 255, 255)),
    'red' : ((240, 110, 47), (360, 255, 255), (20))
}

def create_etalon_dict(dir_name: str) -> dict:
    etalon_dit = {}

    for color_name in os.listdir(dir_name):
        for file_name in os.listdir(f'{dir_name}/{color_name}'):
            etalon_img = cv.imread(f'{dir_name}/{color_name}/{file_name}')
            etalon_mask = getMask(etalon_img, key_color[color_name])

            file_name = str(file_name.split('.')[0])

            if not color_name in etalon_dit.keys():
                etalon_dit[color_name] = {}
            etalon_dit[color_name][file_name] = etalon_mask

    return etalon_dit


park_detector = db.simple_object_detector("parking_sign.svm")
stop_detector = db.simple_object_detector("fullStopDetector.svm")
no_drive_detector = db.simple_object_detector("fullNoDriveDetector.svm")


def get_crop_sign(img: np.ndarray) -> list:
    detect = [
        stop_detector(img),
        no_drive_detector(img),
        park_detector(img)
    ]

    img_copy = img.copy()
    imgesList = []

    for contours in detect:
        for contour in contours:
            x, y, x2, y2 = [contour.left(), contour.top(), contour.right(), contour.bottom()]
            cv.rectangle(img, (x-10, y-10), (x2+10, y2+10), (0, 255, 0), 3)
            imgesList.append(img_copy[y-10 : y2+10, x-10 : x2+10])
    
    # возможно есть смысл добавить проверку на контур в контуре

    return imgesList

def getMask(img_rgb, color):
    try:
        img = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV_FULL)
        mask = cv.inRange(img, color[0], color[1])    

        if len(color) == 3:
            mask_down = cv.inRange(img, (0, color[0][1], color[0][2]), (color[2], color[1][1], color[1][2]))
            mask = cv.bitwise_or(mask, mask_down)

        mask = cv.dilate(mask, np.ones((2, 2)))
        mask = cv.erode(mask, np.ones((2, 2)))

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if contours:
            contours = sorted(contours, key=cv.contourArea, reverse=True)
            if cv.contourArea(contours[0]) > 100:
                x, y, w, h = cv.boundingRect(contours[0])
                mask = mask[y : y+h, x : x+w]
                return cv.resize(mask, (64, 64))
    except:
        return None

def compare(mask: np.ndarray, color: str) -> str:
   
    comp_summ_list = []

    for sign_name, etalon_mask in etalon_dit[color].items():
        comp = np.zeros((64, 64))
        comp[etalon_mask == mask] = 1
        comp_sum = np.sum(comp)
        if 2700 < comp_sum < 3550:
            comp_summ_list.append((sign_name, np.sum(comp)))


    comp_summ_list.sort(key = lambda x: x[1], reverse=True)
    
    if comp_summ_list:
        print(comp_summ_list)
        return comp_summ_list[0][0]
            

capture = cv.VideoCapture(0)

etalon_dit = create_etalon_dict('etalon list')

while True:
    ret, frame_rgb = capture.read()
    frame_rgb = cv.resize(frame_rgb, (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2))
    if frame_rgb.any():
        frame_list = get_crop_sign(frame_rgb)

        for frame in frame_list:
            blue_mask = getMask(frame, key_color['blue'])
            if not isinstance(blue_mask, type(None)):
                compare(blue_mask, 'blue')
                cv.imshow("blue", cv.resize(blue_mask, (128, 128)))

            red_mask = getMask(frame, key_color['red'])
            if not isinstance(red_mask, type(None)):
                compare(red_mask, 'red')
                # cv.imshow("red", cv.resize(red_mask, (128, 128)))

        cv.imshow("frame", frame_rgb)
        
    if (cv.waitKey(30)) == 27:
        break

capture.release()
cv.destroyAllWindows()
