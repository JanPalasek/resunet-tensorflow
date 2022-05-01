from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from resunet.blocks import ResBlock

import math


def ResUNet(input_shape, classes: int, filters_root: int = 64, depth: int = 3):
    """
    Builds ResUNet model.
    :param input_shape: Shape of the input images (h, w, c). Note that h and w must be powers of 2.
    :param classes: Number of classes that will be predicted for each pixel. Number of classes must be higher than 1.
    :param filters_root: Number of filters in the root block.
    :param depth: Depth of the architecture. Depth must be <= min(log_2(h), log_2(w)).
    :return: Tensorflow model instance.
    """
    if classes < 1:
        raise ValueError("There has to be prediction for at least 2 classes.")
    if not math.log(input_shape[0], 2).is_integer() or not math.log(input_shape[1], 2):
        raise ValueError(f"Input height ({input_shape[0]}) and width ({input_shape[1]}) must be power of two.")
    if 2 ** depth > min(input_shape[0], input_shape[1]):
        raise ValueError(f"Model has insufficient height ({input_shape[0]}) and width ({input_shape[1]}) compared to its desired depth ({depth}).")

    input = Input(shape=input_shape)

    layer = input

    # ENCODER
    encoder_blocks = []

    filters = filters_root
    layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(layer)

    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(layer)
    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=True)(branch)
    layer = Add()([branch, layer])

    encoder_blocks.append(layer)

    for _ in range(depth - 1):
        filters *= 2
        layer = ResBlock(filters, strides=2)(layer)

        encoder_blocks.append(layer)

    # BRIDGE
    filters *= 2
    layer = ResBlock(filters, strides=2)(layer)

    # DECODER
    for i in range(1, depth + 1):
        filters //= 2
        skip_block_connection = encoder_blocks[-i]

        layer = UpSampling2D()(layer)
        layer = Concatenate()([layer, skip_block_connection])
        layer = ResBlock(filters, strides=1)(layer)

    layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(layer)

    if classes == 1:
        layer = Activation(activation="sigmoid")(layer)
    else:
        layer = Softmax()(layer)

    output = layer

    return Model(input, output)
