import cv2
import mediapipe as mp
import numpy as np

IMAGE_FILES = ["image1.png", "image2.jpg"]
BG_COLOR = (192, 192, 192)

mp_pose = mp.solutions.pose

# recieved marked up image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)

        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])

        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        annotated_image = np.where(condition, annotated_image, bg_image)

        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        cv2.imwrite(str(idx) + ".png", annotated_image)
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


class PoseEstimator:
    def __init__(self):
        pass

    def process(self):
        pass




