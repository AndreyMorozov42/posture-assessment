import numpy as np


class NeuralNetworkH0:
    
    def __init__(self, input_size, output_size, alpha=0.1):
        self.alpha = alpha
        self.input_size = input_size
        self.output_size = output_size
        self.w = 0.2 * np.random.rand(output_size, input_size) - 0.1
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
