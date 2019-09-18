from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from keras.models import Model


def res_block(x, ch, kernel_size=3, in_conv_size=1):
    xi = x
    for i in range(in_conv_size):
        xi = Conv2D(ch, kernel_size, activation='linear', padding='same')(xi)
        xi = BatchNormalization()(xi)
        xi = Activation('relu')(xi)
    xi = Conv2D(ch, kernel_size, activation='linear', padding='same')(xi)
    xi = BatchNormalization()(xi)
    xi = Add()([xi, x])  # Add function receive two same-shape tensors, and return a also same-shape tensor
    xi = Activation('relu')(xi)
    return xi


def conv_batch(x, ch, kernel_size):
    output = Conv2D(ch, kernel_size, activation='linear', padding='same')(x)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def end_block(x):
    xprobs = Conv2D(2, 3, activation='softmax', padding='same')(x)
    xbbox = Conv2D(6, 3, activation='linear', padding='same')(x)
    return Concatenate(3)([xprobs, xbbox])


def model_WPOD():
    input_layer = Input(shape=(None, None, 3), name='input')

    x = conv_batch(input_layer, 16, 3)
    x = conv_batch(x, 16, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 32, 3)
    x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 64, 3)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 64, 3)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_batch(x, 128, 3)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)

    x = end_block(x)

    return Model(inputs=input_layer, outputs=x)