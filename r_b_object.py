import  cv2
import numpy as np


def nothing(x):
    pass
class range:
    def __init__(self):
        pass

    def create_track(self):
        trace = 'Trackbar Color Palette'
        cv2.namedWindow("trace", cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar('R', "trace", 0, 255, nothing)
        cv2.createTrackbar('G', "trace", 0, 255, nothing)
        cv2.createTrackbar('B', "trace", 0, 255, nothing)
        cv2.createTrackbar('R1', "trace", 0, 255, nothing)
        cv2.createTrackbar('G1', "trace", 0, 255, nothing)
        cv2.createTrackbar('B1', "trace", 0, 255, nothing)

    def get_track(self):
        r = cv2.getTrackbarPos('R', "trace")
        g = cv2.getTrackbarPos('G', "trace")
        b = cv2.getTrackbarPos('B', "trace")
        r1 = cv2.getTrackbarPos('R1', "trace")
        g1 = cv2.getTrackbarPos('G1', "trace")
        b1 = cv2.getTrackbarPos('B1', "trace")
        lower=np.array([r,g,b])
        upper=np.array([r1,g1,b1])
        return lower,upper

    def get_range_blue(self):
        l_blue = np.array([106, 118, 49])
        u_blue = np.array([118, 255, 168])
        return l_blue,u_blue

    # [106, 118, 49])
    # u_blue = np.array([118, 255, 168])
    # [92, 114, 95])
    # u_blue = np.array([111, 255, 199])
    # np.array([50, 121, 60])
    # u_blue = np.array([179, 255, 255])

    def get_range_red(self):
        l_red = np.array([0,114,141])
        u_red = np.array([187,172,213])
        return l_red,u_red

    # [0, 109, 141])
    # u_red = np.array([187, 293, 213]
# [121,127,160])
        # u_red = np.array([179,168,215])
# [0, 121, 123]
# [184, 213, 219]


class object:
    def __init__(self,lower,upper,hsv,frame):
        self.lower=lower
        self.upper=upper
        self.hsv=hsv
        self.frame=frame

    def get_mask(self):
        mask = cv2.inRange(self.hsv, self.lower, self.upper)
        return mask

    def get_threshold(self,mask):
        bit = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        gray = cv2.cvtColor(bit, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return thresh

    def get_erosion(self,thresh):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        return erosion

    def get_dilation(self,thresh):
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        return dilation

    def get_cnt_rect(self,thresh,color_name,color_text,c_x,c_y):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours

        if (len(contours)) != 0:
            c = max(contours, key=cv2.contourArea)
            area=cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area>500:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (x, y - 5)
                fontScale = 1
                thickness = 1
                self.frame = cv2.putText(self.frame, color_name, org, font, fontScale, color_text, thickness, cv2.LINE_AA)

            # cv2.drawContours(self.frame, cnt, -1, (0, 255, 0), 3)
            # cv2.imshow('cnt',self.frame)


                c_x=x+w/2
                c_y=y+h/2
        return self.frame,c_x,c_y
            # else:
            #     return self.frame,0,0




cap = cv2.VideoCapture(0)

r1=range
# r1.create_track(r1)

l_blue,u_blue=r1.get_range_blue(r1)
l_red,u_red=r1.get_range_red(r1)

while True:
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # l_red,u_red=r1.get_track(r1)

    red = object(l_red,u_red,im_hsv,frame)
    blue = object(l_blue,u_blue,im_hsv,frame)

    mask_red=red.get_mask()
    mask_blue=blue.get_mask()

    threshold_red=red.get_threshold(mask_red)
    threshold_blue=blue.get_threshold(mask_blue)

    # morphological red
    erosion_red=red.get_erosion(threshold_red)
    dilation_red=red.get_dilation(erosion_red)

    # morphological blue
    erosion_blue = red.get_erosion(threshold_blue)
    dilation_blue = red.get_dilation(erosion_blue)

    # cv2.imshow('th_r',threshold_red)

    # cv2.imshow('erosion', erosion_red)
    # cv2.imshow('dilation', dilation_red)

    frame,c_x_r,c_y_r=red.get_cnt_rect(dilation_red,'red',(0, 0, 255),0,0)
    frame,c_x_b,c_y_b=blue.get_cnt_rect(dilation_blue,'blue',(255, 0, 0),0,0)

    rect1center=(int(c_x_r),int(c_y_r))
    rect2center=(int(c_x_b),int(c_y_b))
    if (c_x_r>0 or c_y_r>0) and (c_x_b>0 or c_y_b>0):
        cv2.line(frame, rect1center, rect2center, (255,255,0), 2)

        center_x=int((c_x_r+c_x_b)/2)
        center_y=int((c_y_r+c_y_b)/2)

        frame = cv2.circle(frame, (center_x, center_y), radius=0, color=(0, 0, 255), thickness=3)

        frame=cv2.putText(frame, f'({center_x},{center_y})', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('output', frame)



    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()