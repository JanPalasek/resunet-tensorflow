from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


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
        layer = _down_res_block(layer, filters)

        encoder_blocks.append(layer)

    # BRIDGE
    filters *= 2
    layer = _down_res_block(layer, filters)

    # DECODER
    for i in range(1, depth + 1):
        filters //= 2
        layer = _up_res_block(layer, encoder_blocks[-i], filters)

    layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(layer)
    layer = Softmax()(layer)

    output = layer

    return Model(input, output)

def _down_res_block(layer: Layer, filters: int) -> Layer:
    branch = BatchNormalization()(layer)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=2, padding="same", use_bias=False)(branch)

    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(branch)

    layer = Conv2D(filters=filters, kernel_size=1, strides=2, padding="same", use_bias=False)(layer)
    layer = BatchNormalization()(layer)
    layer = Add()([branch, layer])

    return layer

def _up_res_block(layer: Layer, skip_block_connection: Layer, filters: int) -> Layer:
    layer = UpSampling2D()(layer)
    layer = Concatenate()([layer, skip_block_connection])

    branch = BatchNormalization()(layer)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(branch)

    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(branch)

    layer = Conv2D(filters=filters, kernel_size=1, strides=1, padding="same", use_bias=False)(layer)
    layer = BatchNormalization()(layer)
    layer = Add()([branch, layer])

    return layer
