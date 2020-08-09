import tensorflow as tf
import os
import etl
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, MaxPooling2D, SeparableConv2D\
    , concatenate, GlobalAveragePooling2D

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
IMAGE_SIZE = [512, 512]


def fire_module(x, num_squeeze_filters, num_expand_filters, bnmoment=0.9):
    """" Creates a squeezenet style block.
        Args: num_squeeze_filters should be about half of num_expand_filters"""

    squeeze = Conv2D(num_squeeze_filters, kernel_size=(1, 1), activation='relu', padding='same')(x)
    squeeze = BatchNormalization(momentum=bnmoment)(squeeze)
    expand3 = Conv2D(num_expand_filters, kernel_size=(3, 3), activation='relu', padding='same')(squeeze)
    expand3 = BatchNormalization(momentum=bnmoment)(expand3)
    expand1 = Conv2D(num_expand_filters, kernel_size=(1, 1), activation='relu', padding='same')(squeeze)
    expand1 = BatchNormalization(momentum=bnmoment)(expand1)
    return concatenate([expand3, expand1])


def fire(num_squeeze_filters, num_expand_filters, bnmoment=0.9):
    """" Creates a squeezenet style layer (functional style).
        Args: num_squeeze_filters should be about half of num_expand_filters"""

    return lambda x: fire_module(x, num_squeeze_filters, num_expand_filters, bnmoment)


def get_model(image_size=IMAGE_SIZE):
    """Return a model that maps images to polynomial coefficients"""

    x = Input(shape=[*image_size, 3])
    c1 = Conv2D(10, kernel_size=(3, 3), activation='relu', padding='valid')(x)
    c1 = BatchNormalization(momentum=0.9)(c1)
    c2 = Conv2D(20, kernel_size=(3, 3), activation='relu', padding='valid')(c1)
    c2 = BatchNormalization(momentum=0.9)(c2)
    mp1 = MaxPooling2D(pool_size=(2, 2))(c2)

    # add fire blocks
    fire1 = fire(20, 40)(mp1)
    fire2 = fire(40, 80)(fire1)
    fire3 = fire(80, 160)(fire2)
    mp2 = MaxPooling2D(pool_size=(2, 2))(fire3)
    fire4 = fire(160, 320)(mp2)
    fire5 = fire(320, 640)(fire4)

    ga = GlobalAveragePooling2D()(fire5)
    dense_out = Dense(3, activation='linear')(ga)

    model = tf.keras.Model(x, dense_out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


if __name__ == '__main__':
    model = get_model(image_size=(128, 128))
    train_set = etl.get_tfrecord_train_dataset(image_size=(128,128))
    train_set = train_set.batch(16)
    model.fit(train_set)



