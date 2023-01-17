import argparse
import os
import cv2

from networks.nn_0h import NeuralNetworkH0
# from posture_processing.posture_estimate import PoseEstimator
# from posture_processing.posture_classifier import PostureClassifier


FILES_EXTENTION = [".png", ".jpg", ".jpeg"]


def search_files(path):
    """
    :param path:
    :return:
    """
    files = []
    # TODO: make check path on files or dir
    if os.path.exists(path):
        if True in [ex in path for ex in FILES_EXTENTION]:
            files.append(path)
        else:
            # leave files with some extensions
            files = map(lambda x: f"{path}/{x}", os.listdir(path))
            files = list(filter(lambda x: True in [ex in x for ex in FILES_EXTENTION], files))
    return files


def main(path):
    clf_nnh0 = NeuralNetworkH0(input_size=20000, output_size=2, path_to_weights="test_weigths.pickle")

    files = search_files(path)
    for file in files:
        print(file)
        image = cv2.imread(file)
        cv2.imshow("fig", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    # param type has possible argument: foto, video, camera
    # parser.add_argument("--type", "-t", type=str, default="foto")
    parser.add_argument("--path", "-p", default="./")
    args = parser.parse_args()

    main(args.path)
