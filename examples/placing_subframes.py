import cv2
import numpy as np

from frame.output_result import Frame, SubFrame


def main():

    sfr_1 = SubFrame(
        xy_1=(100, 100),
        xy_2=(300, 300),
        sub_image=np.zeros((200, 200, 3)),
        name="subFrame1"
    )

    sfr_2 = SubFrame(
        xy_1=(400, 100),
        xy_2=(600, 300),
        sub_image=np.zeros((200, 200, 3)),
        name="subFrame2"
    )

    fr = Frame(
        width=700,
        height=400,
        frames=[sfr_1, sfr_2]
    )

    img = fr.frame_image

    cv2.imshow("img.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
