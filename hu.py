from math import copysign, log10
import cv2
import numpy as np

class hu:
    def HuDescriptor(self, image):
        im = cv2.resize(image, (50,50))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        desc = np.zeros((128,1))
        for i in range (8):
            for j in range (2):
                lower_hue = np.array([(30*i), 40, 40])
                upper_hue = np.array([(30*i + (j+1)*20), 255, 255])
                mask = cv2.inRange(hsv, lower_hue, upper_hue)
                moments = cv2.moments(mask)
                huMoments = cv2.HuMoments(moments)
                for k in range(0, 7):
                    huMoments[k] = -1 * copysign(1.0, huMoments[k]) * log10(max(1e-30,abs(huMoments[k])))
                desc[i*14 + j*7 : i*14 + j*7 + 7] = huMoments

        desc[112:119] = cv2.HuMoments(
          cv2.moments(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([150, 255, 255]))))
        desc[119:126] = cv2.HuMoments(
          cv2.moments(cv2.inRange(hsv, np.array([120, 50, 50]), np.array([255, 255, 255]))))
        desc = desc.T
        return desc

    def detectAndCompute(self,image, ignoredparam = None):
        img = cv2.resize(image, (150, 150))
        descs = self.HuDescriptor(image)
        for i in range(3):
            for j in range(3):
                desc = self.HuDescriptor(image[i*50:i*50+50,j*50:j*50+50])
                descs = np.vstack((descs, desc))
        for i in range(5):
            for j in range(5):
                desc = self.HuDescriptor(image[i*30:i*30+30,j*30:j*30+30])
                descs = np.vstack((descs, desc))

        for i in range(10):
            for j in range(10):
                desc = self.HuDescriptor(image[i*15:i*15+15,j*15:j*15+15])
                descs = np.vstack((descs, desc))
        kp = 0
        return kp, descs