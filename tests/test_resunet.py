import numpy as np

from resunet.model import ResUNet
import pytest


@pytest.mark.parametrize(
    "input_shape,depth,filters_root,classes,error",
    [
        ((128, 128, 1), 4, 4, 2, False),
        ((64, 64, 3), 3, 4, 2, False),
        ((128, 128, 1), 6, 2, 2, False),
        ((128, 128, 1), 8, 2, 2, True),
        ((128, 128, 1), 7, 2, 2, False),
        ((128, 128, 1), 7, 2, 1, False),
    ]
)
def test_smoke(input_shape, depth, filters_root, classes, error):
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

    assert output.shape[0] == input_shape[0]
    assert output.shape[1] == input_shape[1]

    if classes > 1:
        assert output.shape[2] == classes
    else:
        assert len(output.shape) == 2

