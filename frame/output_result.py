import numpy as np
import cv2


class Frame:
    def __init__(self, width, height, frames=None):
        self.width = width
        self.height = height
        self.frame_image = 255 * np.ones((height, width, 3))
        self.frames = frames

        if frames is None:
            self.frames = []

        if self.frames:
            self.fill_in()

    def fill_in(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        for img in self.frames:
            x1, y1 = img.xy_1
            x2, y2 = img.xy_2
            org = (x1, y1 - 5)
            self.frame_image[y1:y2, x1:x2, ] = img.sub_image.copy()
            self.frame_image = cv2.putText(self.frame_image, img.name,
                                           org, font, fontScale, color,
                                           thickness, cv2.LINE_AA)


class SubFrame:
    def __init__(self, xy_1, xy_2, sub_image, name=None):
        self.name = name
        self.sub_image = sub_image
        self.xy_1 = xy_1
        self.xy_2 = xy_2

