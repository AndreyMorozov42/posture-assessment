import argparse
import pickle
import os
import cv2

from networks.nn_0h import NeuralNetworkH0
from generate_dataset.dataset import DataSet
# from posture_processing.posture_estimate import PoseEstimator
# from posture_processing.posture_classifier import PostureClassifier


FILES_EXTENTION = [".png", ".jpg", ".jpeg"]
PATH_TO_WEIGHTS = "./weights/test_weigths.pickle"
EPOCH = 30


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
    clf_nnh0 = NeuralNetworkH0(input_size=20000, output_size=1, path_to_weights=PATH_TO_WEIGHTS)

    # train neural network
    if clf_nnh0.train_flag:
        dts = DataSet()

        # fill im attr data
        dts.build_data()
        # split dataset on train_data and test_data
        train_data, test_data = dts.split(train=600, test=400)

        # train neural network
        for ep in range(EPOCH):
            for i in range(len(train_data)):
                clf_nnh0.train(*train_data[i][1:], train_data[i][0])
        # ToDo: add accuracy calculation

        # ToDo: save weights in .pickle format
        with open(PATH_TO_WEIGHTS, "wb") as file_weights:
            pickle.dump(clf_nnh0.w, file_weights, pickle.HIGHEST_PROTOCOL)

    files = search_files(path)

    # for file in files:
    #     print(file)
    #     image = cv2.imread(file)
    #     cv2.imshow("fig", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    # param type has possible argument: foto, video, camera
    # parser.add_argument("--type", "-t", type=str, default="foto")
    parser.add_argument("--path", "-p", default="./")
    args = parser.parse_args()

    main(args.path)
