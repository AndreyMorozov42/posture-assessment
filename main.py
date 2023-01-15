import argparse
import os

from networks.nn_0h import NeuralNetworkH0
# from posture_processing.posture_estimate import PoseEstimator
# from posture_processing.posture_classifier import PostureClassifier


FILES_EXTENTION = [".png", ".jpg", ".jpeg"]


def search_files(path):
    files = []
    # TODO: make check path on files or dir
    if os.path.exists(path):
        if True in [ex in path for ex in FILES_EXTENTION]:
            files.append(path)
        else:
            # leave files with some extensions
            files = list(filter(lambda x: True in [ex in x for ex in FILES_EXTENTION], os.listdir(path)))
    return files


def main(path):
    files = search_files(path)
    print(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    # param type has possible argument: foto, video, camera
    # parser.add_argument("--type", "-t", type=str, default="foto")
    parser.add_argument("--path", "-p", default="./")
    args = parser.parse_args()

    main(args.path)
