import unittest
from parameterized import parameterized
import numpy as np

from resunet.model import ResUNet


class ResUNetTests(unittest.TestCase):
    @parameterized.expand([
        [(128, 128, 1), 4],
        [(64, 64, 3), 3]
    ])
    def test_smoke(self, input_shape, depth):
        filters_root = 4
        classes = 2
        inputs = np.ones(shape=input_shape)[np.newaxis, ...]

        model = ResUNet(input_shape, classes, filters_root, depth)
        output = model(inputs).numpy().squeeze()

        self.assertEqual(output.shape[0], input_shape[0])
        self.assertEqual(output.shape[1], input_shape[1])
        self.assertEqual(output.shape[2], classes)

