#from CrossroadDefiner import *
from os import listdir as ld
from os.path import isfile as isf, isdir as isd, join as j
import cv2
from win32api import GetSystemMetrics
from controlBars import erode_dilate_bars as edb, thresh_bars_hsv as tb
import numpy as np

kernels = []
with open("kernels.txt") as op:
    for line in op.readlines():
        l = list(map(np.uint8, line.split()))
        k_size = int(len(l) ** 0.5)
        kernels.append(np.array([[ l[i*k_size + j] for j in range(k_size)] 
            for i in range(k_size)], dtype=np.uint8)) 

QR_PATH = "QR/"
QRR_PATH = "Road_QR/"

QR_DATA = {d: tuple([ f for f in ld(j(QRR_PATH, d)) if isf(j(j(QRR_PATH, d), f)) ]) 
                    for d in ld(QRR_PATH) if isd(j(QRR_PATH, d))
}
print(QR_DATA)

template = cv2.imread(QR_PATH + "SHR_1") 
img      = cv2.imread(QRR_PATH + "UP/u3.jpg")

gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = cv2.resize(hsv, (GetSystemMetrics(0), GetSystemMetrics(1)))

#tb("thresh", hsv)

#thresh metrics min = (0,0,52) max = (255,62,71)

thresh = cv2.inRange(hsv, (0, 9, 47), (255, 69, 77))#(0,0,52), (255,62,71)) ((0,0,44), (255,26, 83)) ((88, 16, 46), (255,58,83))
#(0, 9, 47), (255, 69, 77)

#edb("vgolovu", thresh) #(2, 1, 2, 3)

morph = cv2.erode(thresh, kernel=kernels[1], iterations=2)
morph = cv2.dilate(morph, kernel=kernels[3], iterations=2)

res = cv2.matchTemplate(yellow_edges, dot_edges, cv2.TM_CCOEFF_NORMED)
#contours = cv2.findContours(morph.astype(np.uint8), cv2.RETR_EXTERNAL, 
#                        cv2.CHAIN_APPROX_SIMPLE)

#qr_c = sorted(contours[0], key=lambda x: cv2.contourArea(x))[-1]
#print(qr_c)

#min_h, max_h = min(qr_c[:,:, 0]), max(qr_c[:,:, 0])
#min_w, max_w = min(qr_c[:,:, 1]), max(qr_c[:,:, 1])

#img_q = cv2.resize(img.copy(), (GetSystemMetrics(0), GetSystemMetrics(1)))

#cv2.rectangle(img_q, (min_w, min_h), (max_w, max_h), (0,255,0))

cv2.imshow("thresh", img_q)

cv2.waitKey(0)

cv2.destroyAllWindows()

#crd = CrossroadDefiner(QR_PATH)