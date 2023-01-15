import numpy as np
import cv2


class PostureClassifier:
    def __init__(self, cls=None):
        if cls:
            self.cls = cls

    def classification(self, image):
        """
        This method must return class of posture on transferred image
        """
        image = ((255 - image.reshape(image.shape[0] * image.shape[1])) / 255 * 0.99) + 0.01
        if self.cls:
            res = self.cls.query(image)
        else:
            return None

