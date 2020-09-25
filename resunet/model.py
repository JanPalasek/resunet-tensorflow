from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from resunet.blocks import ResBlock


def ResUNet(input_shape, classes: int, filters_root: int = 64, depth: int = 3):
    input = Input(shape=input_shape)

    layer = input

    # ENCODER
    encoder_blocks = []

    filters = filters_root
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
    layer = Softmax()(layer)

    output = layer

    return Model(input, output)
