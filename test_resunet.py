import unittest
from parameterized import parameterized
import numpy as np

from resunet.model import ResUNet


class ResUNetTests(unittest.TestCase):
    @parameterized.expand([
        [(128, 128, 1), 4, 4, False],
        [(64, 64, 3), 3, 4, False],
        [(128, 128, 1), 6, 2, False],
        [(128, 128, 1), 8, 2, True],
        [(128, 128, 1), 7, 2, False]
    ])
    def test_smoke(self, input_shape, depth, filters_root, error=False):
        classes = 2
        inputs = np.ones(shape=input_shape)[np.newaxis, ...]

        try:
            model = ResUNet(input_shape, classes, filters_root, depth)
        except Exception:
            if not error:
                # if it's not supposed to end as error => raise the error
                raise

            # otherwise end the test
            return
        output = model(inputs).numpy().squeeze()

        self.assertEqual(output.shape[0], input_shape[0])
        self.assertEqual(output.shape[1], input_shape[1])
        self.assertEqual(output.shape[2], classes)

