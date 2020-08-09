import tensorflow as tf
import os
import etl
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, MaxPooling2D, SeparableConv2D\
    , concatenate, GlobalAveragePooling2D

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
IMAGE_SIZE = [512, 512]


def inception_block(x, num_3_filters, num_1_filters, batch_norm_moment=0.9):
    """ Get InceptionV2 style block"""
    side_pool = MaxPooling2D(pool_size=(2, 2))(x)
    # middle branch
    middle_branch_1 = SeparableConv2D(num_1_filters, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                                      padding='same')(x)
    middle_branch_1 = BatchNormalization(momentum=batch_norm_moment)(middle_branch_1)
    middle_branch_3 = SeparableConv2D(num_3_filters, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                                      padding='same')(middle_branch_1)
    middle_branch_3 = BatchNormalization(momentum=batch_norm_moment)(middle_branch_3)

    # big branch
    big_branch_1 = SeparableConv2D(num_1_filters, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                                   padding='same')(x)
    big_branch_1 = BatchNormalization(momentum=batch_norm_moment)(big_branch_1)
    big_branch_3_1 = SeparableConv2D(num_3_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     activation='relu')(big_branch_1)
    big_branch_3_1 = BatchNormalization(momentum=batch_norm_moment)(big_branch_3_1)
    big_branch_3_2 = SeparableConv2D(num_3_filters, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                                     padding='same')(big_branch_3_1)
    big_branch_3_2 = BatchNormalization(momentum=batch_norm_moment)(big_branch_3_2)

    depth_concat = concatenate([side_pool, middle_branch_3, big_branch_3_2])

    return depth_concat


def inception(num_3_filters, num_1_filters, batch_norm_moment=0.9):
    """ Creates an Inception v2 style layer."""

    return lambda x: inception_block(x, num_3_filters, num_1_filters, batch_norm_moment=0.9)


def get_model():
    """Return a model that maps images to polynomial coefficients"""

    # stem
    x = Input(shape=[*IMAGE_SIZE, 3])
    c1 = SeparableConv2D(6, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    c1 = BatchNormalization(momentum=0.9)(c1)
    c2 = SeparableConv2D(12, kernel_size=(3, 3), padding='valid', activation='relu')(c1)
    c2 = BatchNormalization(momentum=0.9)(c2)
    c3 = SeparableConv2D(24, kernel_size=(3, 3), padding='valid', activation='relu')(c2)
    c3 = BatchNormalization(momentum=0.9)(c3)
    mp1 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = SeparableConv2D(48, kernel_size=(3, 3), padding='valid', activation='relu')(mp1)
    c4 = BatchNormalization(momentum=0.9)(c4)
    c5 = SeparableConv2D(96, kernel_size=(3, 3), padding='valid', activation='relu')(c4)
    c5 = BatchNormalization(momentum=0.9)(c5)
    mp2 = MaxPooling2D(pool_size=(2, 2))(c5)

    # inception blocks
    i1 = inception(192, 96)(mp2)
    i2 = inception(192, 384)(i1)
    #i3 = inception(384, 768)(i2)
    g = GlobalAveragePooling2D()(i2)
    d = Dense(3, activation='linear')(g)
    model = tf.keras.Model(inputs=x, outputs=d)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


if __name__ == '__main__':
    ds = etl.get_tfrecord_train_dataset()
    ds = ds.batch(16)
    model = get_model()
    model.fit(ds, epochs=1)


