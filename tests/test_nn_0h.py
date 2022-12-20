import unittest
import sys
import numpy as np

sys.path.insert(0, "..")

from networks.nn_0h import NeuralNetworkH0


class TestNeuralNetworkH0(unittest.TestCase):
    def setUp(self):
        self.n = NeuralNetworkH0(
            input_size=2,
            output_size=2
        )
        self.n.w = np.array([[1.0, 2.0],
                             [0.0, 1.0]])

    def test_query(self):
        inp = np.array([1, 0])
        out = self.n.sigmoid(np.array([[1, 0]])).T
        n_out = self.n.query(inp)
        self.assertTrue(np.alltrue(n_out == out))

    def test_train(self):
        inp = np.array([1, 0])
        trn = np.array([1, 1])
        for i in range(1000):
            self.n.train(i=inp, t=trn)
        out_n = self.n.query(i=inp)
        self.assertGreater(0.3, sum(np.array([trn]).T - out_n))


if __name__ == "__main__":
    unittest.main()
