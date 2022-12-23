import numpy as np


class NeuralNetworkH1:
    
    def __init__(self, input_size, hide_size, output_size, alpha=0.1):
        self.alpha = alpha
        self.input_size = input_size
        self.hide_size = hide_size
        self.output_size = output_size
        self.w_ih = 0.2 * np.random.rand(hide_size, input_size) - 0.1
        self.w_ho = 0.2 * np.random.rand(output_size, hide_size) - 0.1
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.relu = lambda x: (x > 0) * x
        self.relu2deriv = lambda x: x > 0
        pass
        
    def train(self, i, t):
        i = np.array([i], dtype=float).T
        t = np.array([t], dtype=float).T
        o_ih = self.relu(np.dot(self.w_ih, i))
        o_ho = self.sigmoid(np.dot(self.w_ho, o_ih))
        e = t - o_ho
        e_h = np.dot(self.w_ho.T, e)
        self.w_ho += self.alpha * np.dot(e * o_ho * (1 - o_ho), o_ih.T)
        self.w_ih += self.alpha * np.dot(e_h * self.relu2deriv(o_ih), i.T)
        pass
    
    def query(self, i):
        i = np.array([i], dtype=float).T
        o_ih = self.relu(np.dot(self.w_ih, i))
        o_ho = self.sigmoid(np.dot(self.w_ho, o_ih))
        return o_ho
