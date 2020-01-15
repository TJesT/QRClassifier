import numpy as np
import cv2


class CrossroadDefiner:
    def __init__(self, path):
        self.__path = path
        self.direction = None
        return

    def define(self):
        img = cv2.imread(self.__path)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    def change_path(self, path):
        self.__path = path