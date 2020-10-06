import tensorflow as tf
import metrics
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Input, MaxPooling2D, SeparableConv2D\
    , concatenate, GlobalAveragePooling2D


# Constants
IMAGE_SIZE = [256, 256]
_lambda = 0.8


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

    return model


def get_qreg_model(input_shape):
    """Return a NN model that maps patient records, CNN FVC prediction and exponent coefficients to quantile FVC
     prediction"""

    inp = Input(shape=input_shape)

    # dense layers
    d1 = Dense(100, activation='relu')(inp)
    d2 = Dense(100, activation='relu')(d1)

    # quantile predictions
    pred1 = Dense(3, activation='linear')(d2)
    pred2 = Dense(3, activation='relu')(d2)  # adjusting predictions
    # this is somehow making the model more robust
    quantiles = tf.keras.layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1))([pred1, pred2])

    # compile
    model = tf.keras.Model(inp, quantiles)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
                  loss=metrics.mloss(_lambda), metrics=metrics.score)

    return model


def enlarge_cnn_model(model_path=None):
    """Take a 256x256 model from model_path and add a first convolutional layer to enlarge the model to 512x512.
    Args:
        model_path: path to model weights which we want to enlarge
    """

    prior = get_sqeezenet_model(image_size=[*IMAGE_SIZE])
    if model_path:
        prior.load_weights(model_path)

    # freeze prior layers
    for layer in prior.layers:
        layer.trainable = False

    # add conv layer
    model = tf.keras.Sequential([
        Input(shape=[512, 512, 3]),
        Conv2D(64, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same'),
        Conv2D(3, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same'),
        prior
    ])

    return model


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(get_sqeezenet_model(image_size=[256, 256]).summary())
    print(enlarge_cnn_model('models_weights/cnn_model/model_v3.ckpt').summary())





