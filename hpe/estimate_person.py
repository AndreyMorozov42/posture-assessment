import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self, mode=False):
        self.LANDMARKS = ["LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER",
                          "RIGHT_SHOULDER", "RIGHT_HIP", "LEFT_HIP",
                          "LEFT_KNEE", "RIGHT_KNEE", "RIGHT_ANKLE",
                          "LEFT_ANKLE"]

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
        )
        self.points_posture = None

    def process(self, name_image):
        img = cv2.imread(name_image)
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.points_posture = self.points_selection(results)

    def points_selection(self, results):
        if not results.pose_landmarks:
            return None

        posture = dict()

        for LANDMARK in self.LANDMARKS:
            point = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark[LANDMARK]]
            if point.visibility > 0.5:
                posture[LANDMARK] = (point.x, point.y)

        return posture


if __name__ == "__main__":
    IMAGE_FILES = ["image1.png", "image2.jpg"]
    cls = PoseEstimator()
    res = cls.process(IMAGE_FILES[1])
    print(cls.points_posture)
