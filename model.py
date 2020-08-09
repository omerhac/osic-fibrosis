import tensorflow as tf
import os
import etl
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, MaxPooling2D, SeparableConv2D\
    , concatenate, GlobalAveragePooling2D

# Just disables the warning, doesn't enable AVX/FMA
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

    # dimensionality reduction
    x = Input(shape=[*image_size, 3])
    c1 = Conv2D(96, kernel_size=(7, 7), strides=(2,2), activation='relu', padding='same')(x)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c1)

    # first section
    fire1 = fire(16, 64)(mp1)
    fire2 = fire(16, 64)(fire1)
    fire3 = fire(32, 128)(fire2)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(fire3)

    # second section
    fire4 = fire(32, 128)(mp2)
    fire5 = fire(48, 192)(fire4)
    fire6 = fire(48, 192)(fire5)
    fire7 = fire(64, 256)(fire6)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire7)

    # third section
    fire8 = fire(64, 256)(mp3)
    last_conv = Conv2D(1000, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same')(fire8)
    gap = GlobalAveragePooling2D()(last_conv)

    # output
    dense_out = Dense(3, activation='linear')(gap)

    # compile
    model = tf.keras.Model(x, dense_out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    model = get_model(image_size=(224, 224))
    train_set = etl.get_tfrecord_train_dataset(image_size=(224, 224))
    train_set = train_set.batch(16)
    model.fit(train_set)




