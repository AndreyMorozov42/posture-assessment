import numpy as np
import pickle
import os


class NeuralNetworkH0:
    
    def __init__(self, input_size, output_size, alpha=0.1, path_to_weights=None):
        self.alpha = alpha
        self.input_size = input_size
        self.output_size = output_size

        # if weights not exists - initialize weights and train neural network
        if path_to_weights is None or not os.path.exists(path=path_to_weights):
            self.w = 0.2 * np.random.rand(output_size, input_size) - 0.1
            self.train_flag = True
        else:
            with open(path_to_weights, "rb") as file_weights:
                self.w = pickle.load(file_weights)
                self.train_flag = False

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        pass

    def train(self, i, t):
        i = np.array([i], dtype=float).T
        t = np.array([t], dtype=float).T
        o = self.sigmoid(np.dot(self.w, i))
        e = (t - o)
        self.w += self.alpha * np.dot(e * o * (1 - o), i.T)
        pass

    def query(self, i):
        i = np.array([i], dtype=float).T
        o = self.sigmoid(np.dot(self.w, i))
        return o
    
    pass
