from time import sleep

import cv2
import numpy as np

from samples.iterative_training.Utils import Utils


class ImshowWindow:
    def __init__(self, name, maxSize=None):
        self.name = name
        self.created = False
        self.img = None
        self.maxSize = maxSize

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    def setBusy(self):
        if not self.created:
            return
        overlay = np.full_like(self.img, 255)
        overlayedImg = cv2.addWeighted(self.img, .3, overlay, .7, 0, overlay)
        cv2.imshow(self.name, overlayedImg)
        cv2.waitKey(2)

    @staticmethod
    def _fitToMaxSize(img, maxSize):
        if not maxSize:
            return img
        maxH, maxW = maxSize
        return img  # TODO
        return cv2.resize(img, maxSize)

    def imshow(self, img, imgInRgb=True):
        img = self._fitToMaxSize(img, self.maxSize)

        if imgInRgb:
            img = Utils.rgb2bgr(img)
        self.img = img
        cv2.imshow(self.name, self.img)
        self.created = True

    def setTitle(self, title):
        if not self.created:
            return
        cv2.setWindowTitle(self.name, title)

    def destroy(self):
        self.img = None
        if self.created:
            cv2.destroyWindow(self.name)
            self.created = False


if __name__ == '__main__':
    with ImshowWindow('TEST-TEST') as w:
        img = np.full([400, 400, 3], [0, 200, 0], np.uint8)
        cv2.circle(img, (200, 200), 100, (200, 0, 0), -1)

        w.imshow(img)
        cv2.waitKey()
        w.setBusy()
        sleep(2)
        cv2.waitKey()
