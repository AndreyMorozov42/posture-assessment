import numpy as np
import random
import math
import cv2


class DataSet:
    def __init__(self, amount=1000):
        # data contains the dataset
        self.data = []

        # # test data for check
        # self.test_data = []
        # # train data for learn
        # self.train_data = []

        # amounts data in dataset
        self.amount = amount
        self.random_cos = lambda sin_a: random.choice([-1, 1]) * ((1 - sin_a ** 2) ** 0.5)

    def build_data(self):
        image = np.ones((200, 100))
        for i in range(0, self.amount, 2):
            self.gen_correct_posture(image)
            self.gen_wrong_posture(image)

    def gen_correct_posture(self, image, thickness=2):
        image = image.copy()
        L = random.randint(180, 195) // 8
        x1, y1 = random.randint(40, 60), 190
        alpha, dl = math.radians(random.uniform(88, 90)), 7 * L // 4
        x2, y2 = x1 + round(math.cos(alpha) * dl), y1 - round(math.sin(alpha) * dl)
        alpha, dl = math.radians(random.uniform(81, 90)), 7 * L // 4
        x3, y3 = x2 + round(math.cos(alpha) * dl), y2 - round(math.sin(alpha) * dl)
        x4, y4 = x3, y3 - 8 * L // 3
        x5, y5 = x3, y4 - 5 * L // 6
        cv2.circle(image, (x1, y1), 3, (0, 0, 0), -1)
        cv2.circle(image, (x2, y2), 3, (0, 0, 0), -1)
        cv2.circle(image, (x3, y3), 3, (0, 0, 0), -1)
        cv2.circle(image, (x4, y4), 3, (0, 0, 0), -1)
        cv2.circle(image, (x5, y5), 3, (0, 0, 0), -1)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness)
        cv2.line(image, (x2, y2), (x3, y3), (0, 0, 0), thickness)
        cv2.line(image, (x3, y3), (x4, y4), (0, 0, 0), thickness)
        cv2.line(image, (x4, y4), (x5, y5), (0, 0, 0), thickness)
        self.data.append((1, (255 - image.reshape(20000)) / 255 * 0.99 + 0.01))

    def gen_wrong_posture(self, image, thickness=2):
        image = image.copy()
        L = random.randint(180, 195) // 8
        x1, y1 = random.randint(40, 60), 190
        dl, sin_a = 7 * L // 4, math.sin(math.radians(random.uniform(70, 80)))
        x2, y2 = x1 - round(self.random_cos(sin_a) * dl), y1 - round(dl * sin_a)
        dl, sin_a = 7 * L // 4, math.sin(math.radians(random.uniform(70, 80)))
        x3, y3 = x2 - round(self.random_cos(sin_a) * dl), y2 - round(dl * sin_a)
        dl, sin_a = 8 * L // 3, math.sin(math.radians(random.uniform(70, 80)))
        x4, y4 = x3 - round(self.random_cos(sin_a) * dl), y3 - round(dl * sin_a)
        dl, sin_a = 5 * L // 6, math.sin(math.radians(random.uniform(70, 80)))
        x5, y5 = x4 - round(self.random_cos(sin_a) * dl), y4 - round(dl * sin_a)
        cv2.circle(image, (x1, y1), 3, (0, 0, 0), -1)
        cv2.circle(image, (x2, y2), 3, (0, 0, 0), -1)
        cv2.circle(image, (x3, y3), 3, (0, 0, 0), -1)
        cv2.circle(image, (x4, y4), 3, (0, 0, 0), -1)
        cv2.circle(image, (x5, y5), 3, (0, 0, 0), -1)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness)
        cv2.line(image, (x2, y2), (x3, y3), (0, 0, 0), thickness)
        cv2.line(image, (x3, y3), (x4, y4), (0, 0, 0), thickness)
        cv2.line(image, (x4, y4), (x5, y5), (0, 0, 0), thickness)
        self.data.append((0, (255 - image.reshape(20000)) / 255 * 0.99 + 0.01))

    def split(self, train, test):
        # ToDo: make better
        # self.train_data = self.data[0:train]
        # self.test_data = self.data[train:train+test]
        return self.data[0:train], self.data[train:train+test]


if __name__ == "__main__":
    ds = DataSet()
    ds.build_data()
    # return dataset
    img_data = ds.data
    # split dataset for train and test
    ds.split(train=600, test=400)
    # return shape test data
    print(len(ds.test_data))
    # return shape train data
    print(len(ds.train_data))


