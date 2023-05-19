import cv2
import numpy as np


def nothing(x):
    pass

trace = 'Trackbar Color Palette'
cv2.namedWindow("trace", cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('R', "trace", 0, 255, nothing)
cv2.createTrackbar('G', "trace", 0, 255, nothing)
cv2.createTrackbar('B', "trace", 0, 255, nothing)
cv2.createTrackbar('R1', "trace", 0, 255, nothing)
cv2.createTrackbar('G1', "trace", 0, 255, nothing)
cv2.createTrackbar('B1', "trace", 0, 255, nothing)

r = cv2.getTrackbarPos('R', "trace")
g = cv2.getTrackbarPos('G', "trace")
b = cv2.getTrackbarPos('B', "trace")
r1 = cv2.getTrackbarPos('R1', "trace")
g1 = cv2.getTrackbarPos('G1', "trace")
b1 = cv2.getTrackbarPos('B1', "trace")

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    # frame = np.uint8(frame)
    im_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # r,g,b=trackbar()
    r = cv2.getTrackbarPos("R", "trace")
    g = cv2.getTrackbarPos("G", "trace")
    b = cv2.getTrackbarPos("B", "trace")

    r1 = cv2.getTrackbarPos("R1", "trace")
    g1 = cv2.getTrackbarPos("G1", "trace")
    b1 = cv2.getTrackbarPos("B1", "trace")

    l_blue=np.array([50,121,80])
    u_blue=np.array([179,255,255])

    l_red=np.array([0,121,123])
    u_red=np.array([184,213,219])

    mask_b=cv2.inRange(im_hsv,l_blue,u_blue)
    mask_r=cv2.inRange(im_hsv,l_red,u_red)

    blue=cv2.bitwise_and(frame,frame,mask=mask_b)
    red =cv2.bitwise_and(frame,frame,mask=mask_r)
    # cv2.imshow('red', red)
    # cv2.imshow('blue',blue)

    # cv2.imshow('red_mask',mask_r)
    # cv2.imshow('blue_mask', mask_b)
    gray_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    gray_red = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('red_mask',mask_r)
    # cv2.imshow('blue_mask', mask_b)
    ret,thresh_b = cv2.threshold(gray_blue,10,255,cv2.THRESH_BINARY)
    ret,thresh_r = cv2.threshold(gray_red, 10, 255, cv2.THRESH_BINARY)

    # cv2.imshow('thresh_b',thresh_b)
    # cv2.imshow('thresh_r', thresh_r)

    contours, hierarchy = cv2.findContours(thresh_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_b=contours
    if(len(contours))!=0:
        c = max(contours, key=cv2.contourArea)
        x , y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y - 5)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 1
        frame = cv2.putText(frame, 'blue', org, font, fontScale, color, thickness, cv2.LINE_AA)
    # cv2.drawContours(frame, cnt_b, -1, (0, 255, 0), 3)
    # cv2.imshow('cnt_b',frame)

    contours, hierarchy = cv2.findContours(thresh_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_r = contours
    # cv2.drawContours(frame, cnt_r, -1, (0, 255, 0), 3)
    # cv2.imshow('cnt_r',frame)

    # areas_r = [cv2.contourArea(c) for c in cnt_r[0]]
    # max_area_r = np.argmax(areas_r)
    # print(max_area_r)
    if len(contours)!=0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y - 5)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 1
        frame = cv2.putText(frame, 'red', org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('output', frame)


    if cv2.waitKey(1) & 0xFF == ord('a'):
        break


cap.release()
cv2.destroyAllWindows()

dbehjhbfsdsdb