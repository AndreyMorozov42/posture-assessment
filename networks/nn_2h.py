import numpy as np


class NeuralNetwork:
    def __init__(self, alpha, input_size, hide_size1, hide_size2, output_size):
        self.alpha = alpha
        self.input_size = input_size
        self.hide_size1 = hide_size1
        self.hide_size2 = hide_size2
        self.output_size = output_size
        self.w_h1 = 0.2 * np.random.rand(hide_size1, input_size) - 0.1
        self.w_h1h2 = 0.2 * np.random.rand(hide_size2, hide_size1) - 0.1
        self.w_h2 = 0.2 * np.random.rand(output_size, hide_size2) - 0.1
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.relu = lambda x: (x > 0) * x
        self.relu2deriv = lambda x: x > 0
        pass

    def train(self, i, t):
        i = np.array([i], dtype=float).T
        t = np.array([t], dtype=float).T
        i_h1 = self.relu(np.dot(self.w_h1, i))
        i_h1h2 = self.relu(np.dot(self.w_h1h2, i_h1))
        i_h2 = self.sigmoid(np.dot(self.w_h2, i_h1h2))
        e = t - i_h2
        e_h2 = np.dot(self.w_h2.T, e)
        e_h1 = np.dot(self.w_h1h2.T, e_h2)
        self.w_h2 += self.alpha * np.dot(e * i_h2 * (1 - i_h2), i_h1h2.T)
        self.w_h1h2 += self.alpha * np.dot(e_h2 * self.relu2deriv(i_h1h2), i_h1.T)
        self.w_h1 += self.alpha * np.dot(e_h1 * self.relu2deriv(i_h1), i.T)
        pass

    def query(self, i):
        i = np.array([i], dtype=float).T
        i_h1 = self.relu(np.dot(self.w_h1, i))
        i_h1h2 = self.relu(np.dot(self.w_h1h2, i_h1))
        i_h2 = self.sigmoid(np.dot(self.w_h2, i_h1h2))
        return i_h2

