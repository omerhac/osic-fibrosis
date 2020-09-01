import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, MaxPooling2D, SeparableConv2D\
    , concatenate, GlobalAveragePooling2D


# Constants
IMAGE_SIZE = [224, 224]
_lambda = 0.65


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


def get_sqeezenet_model(image_size=IMAGE_SIZE):
    """Return a CNN model that maps images to polynomial coefficients. Squeezenet architecture"""

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
    dense_out = Dense(1, activation='linear')(gap)  # predict the exponential coefficient

    # compile
    model = tf.keras.Model(x, dense_out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


# This section is taken from 'https://www.kaggle.com/chrisden/6-82-quantile-reg-lr-schedulers-checkpoints'
# I don't know who created the original kernel so I thank 'from coffee import *'
# from kaggle at https://www.kaggle.com/chrisden' from whom I copied this...

def score(y_true, y_pred):
    """Calculate the competition metric"""
    # create constants for the loss function
    C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

    # cast dtypes
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)

    # compute sigma as the difference betwwen the marginal quantiles
    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma_clip = tf.maximum(sigma, C1)

    # compute fvc as the median quantile
    fvc_pred = y_pred[:, 1]

    # compute delta as the error between ground truth and the computed median
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)

    # compute metric
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)

    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    """Calculate Pinball loss"""
    # IMPORTANT: define quartiles, feel free to change here!
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.keras.backend.mean(v)


def mloss(_lambda):
    """Combine Score and qloss.
    Args:
        _lambda: weighting constant for how much competition metric should be in the loss functio.
                 higher _lambda -> lower weight for the competition metric
    """

    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def get_qreg_model(input_shape):
    """Return a NN model that maps patient records, CNN FVC prediction and exponent coefficients to quantile FVC
     prediction"""

    inp = Input(shape=input_shape)

    # dense layers
    d1 = Dense(128, activation='relu')(inp)
    d2 = Dense(128, activation='relu')(d1)

    # quantile predictions
    pred1 = Dense(3, activation='relu')(d2)
    pred2 = Dense(3, activation='linear')(d2)  # adjusting predictions
    # this is somehow making the model more robust
    quantiles = tf.keras.layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1))([pred1, pred2])

    # compile
    model = tf.keras.Model(inp, quantiles)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
                  loss=mloss(_lambda), metrics=score)

    return model


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))






