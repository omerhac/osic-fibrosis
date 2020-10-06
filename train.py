import etl
import models
import tensorflow as tf
import visualize
import pandas as pd
import numpy as np
import metrics
from LargestValuesHolder import LargestValuesHolder


# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'

# CNN Constants
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 16
CNN_STEPS_PER_EPOCH = 32994 // CNN_BATCH_SIZE
CNN_IMAGE_SIZE = (256, 256)
AUTO = tf.data.experimental.AUTOTUNE

# Theta constants
THETA_EPOCHS = 1550
THETA_BATCH_SIZE = 256
THETA_STEPS_PER_EPOCH = 32994 // THETA_BATCH_SIZE


def train_cnn_model(save_path, load_path=None, enlarge_model=False, hard_examples_training=True):
    """Train the CNN model. Save weights to models_weights/cnn_model. Return history dict
        Arguments:
            save_path: where to save the model
            load_path: path to load pretrained model checkpoint
            enlarge_model: flag whether to train en enlarged cnn model
            hard_examples_training: flag whether to train the model on the HARDEST_EXAMPLES examples again after each
            epoch.
    """

    # get image size
    image_size = [512, 512] if enlarge_model else CNN_IMAGE_SIZE

    # get datasets
    train_dataset = etl.get_tfrecord_dataset(image_size=image_size, type='train')
    val_dataset = etl.get_tfrecord_dataset(image_size=image_size, type='validation')

    # batch and repeat dataset
    train_dataset = train_dataset.shuffle(buffer_size=CNN_BATCH_SIZE)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(CNN_BATCH_SIZE)
    val_dataset = val_dataset.batch(CNN_BATCH_SIZE)

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')
    ])

    train_dataset = train_dataset.map(lambda image, coeff: (data_augmentation(image, training=True), coeff),
                                      num_parallel_calls=AUTO)

    # get model
    network = models.get_sqeezenet_model(CNN_IMAGE_SIZE)
    if load_path:
        network.load_weights(load_path)
    if enlarge_model:
        network = models.enlarge_cnn_model(load_path)

    # add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', mode='auto',
                                                      restore_best_weights=True, patience=5, verbose=1)
    callback_list = [early_stopping]

    # define optimizer
    optimizer = tf.optimizers.Adam()

    # custom training loop doesnt work for some reason on an enlarged model...
    if not hard_examples_training:
        network.compile(optimizer='adam', loss='mse')
        hist = network.fit(train_dataset, validation_data=val_dataset, epochs=CNN_EPOCHS, steps_per_epoch=CNN_STEPS_PER_EPOCH,
                           callbacks=callback_list)
        return hist.history()

    # go threw custom training loop:
    # train operation
    @tf.function
    def train_op(x, y):
        """Helper function to compute loss, gradients and train the network. Return loss scalar"""
        with tf.GradientTape() as tape:
            preds = network(x, training=True)
            loss = tf.keras.losses.MSE(y, preds)
            grads = tape.gradient(loss, network.trainable_variables)

        # train
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

        return loss

    # validation operation
    @tf.function
    def val_op(x, y):
        """Helper function to evaluate the network on x and y"""
        preds = network(x, training=False)
        loss = tf.keras.losses.MSE(y, preds)
        return loss

    # initialize accumulators
    train_losses = []
    validation_losses = []
    step = 0
    HARDEST_EXAMPLES = 200
    hardest_examples = LargestValuesHolder(n_elements=HARDEST_EXAMPLES)

    # training loop
    for batch_x, batch_y in train_dataset:
        # print epoch
        if step % CNN_STEPS_PER_EPOCH == 0:
            epoch = step // CNN_STEPS_PER_EPOCH + 1
            if epoch > CNN_EPOCHS:
                break
            else:
                print("EPOCH {}/{}".format(epoch, CNN_EPOCHS))

        # train network and get loss
        loss = np.mean(train_op(batch_x, batch_y).numpy())

        # try adding batch to hardest examples
        hardest_examples.add_item((batch_x, batch_y), loss)

        # print step
        if (step % CNN_STEPS_PER_EPOCH) % 100 == 0:
            print("Step {}/{}, current loss {:.5e}, highest loss {:.5e}".format(
                step % CNN_STEPS_PER_EPOCH,
                CNN_STEPS_PER_EPOCH,
                loss,
                hardest_examples.get_max_value() # check max loss during last epoch
            ))

        # update step
        step = optimizer.iterations.numpy()

        # eval on validation data
        if step % CNN_STEPS_PER_EPOCH == 0:
            val_losses = []
            for val_batch_x, val_batch_y in val_dataset:
                val_loss = np.mean(val_op(val_batch_x, val_batch_y).numpy())
                val_losses.append(val_loss)
            print("---Validation loss {:.5e}---".format(val_loss))

            # add losses
            train_losses.append(loss)
            validation_losses.append((np.mean(val_losses)))

            # train on hardest examples
            if hard_examples_training:
                print("Training on hardest {} examples...""".format(HARDEST_EXAMPLES))
                for hard_x, hard_y in hardest_examples.get_items():
                  _ = train_op(hard_x, hard_y)

                hardest_examples = LargestValuesHolder(n_elements=HARDEST_EXAMPLES)

    print("### Training finished ###")
    # save model
    network.save_weights(save_path)

    return {'loss': train_losses, 'val_loss': validation_losses}


def get_lr_callback(batch_size=64, plot=False, epochs=50, lr_start=0.0002, lr_min=0.00001):
    """Returns a lr_scheduler callback which is used for training.
    Credit to 'from coffee import *' from kaggle at https://www.kaggle.com/chrisden'
    """
    lr_max = lr_start * batch_size  # higher batch size --> higher lr
    # 30% of all epochs are used for ramping up the LR and then declining starts
    lr_ramp_ep = epochs * 0.3
    lr_sus_ep = 0
    lr_decay = 0.991

    def lr_scheduler(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    if not plot:
        # get the Keras-required callback with our LR for training
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=False)
        return lr_callback

    else:
        return lr_scheduler


def train_qreg_model(save_path,
                     pp_train_data=None,
                     without_validation=False):

    """Train the theta predicting model. Save weights to models_weights/qreg_model. Return history dict.
    Args:
        save_path: path to save model weights

        provided.
        pp_train_data: optional preprocessed train data. Will create it if not provided.
        without_validation: a flag to order training on the whole training set without validation.
    """

    # get datasets
    if not pp_train_data:
        dataset = etl.create_nn_train('models_weights/qreg_model/processor.pickle')
    else:
        dataset = pd.read_csv(pp_train_data)

    train_ids, val_ids = etl.get_train_val_split()

    # cast dtypes
    numeric_columns = dataset.select_dtypes(include=np.number).columns
    dataset[numeric_columns] = dataset[numeric_columns].astype('float32')

    train_dataset = dataset.loc[dataset["Patient"].isin(train_ids)]  # get train rows
    val_dataset = dataset.loc[dataset["Patient"].isin(val_ids)]  # get validation rows

    # split target
    train_y = train_dataset["GT_FVC"].values
    val_y = val_dataset["GT_FVC"].values
    train_x = train_dataset.drop(["GT_FVC", "Patient"], axis=1).values  # drop target and patient id
    val_x = val_dataset.drop(["GT_FVC", "Patient"], axis=1).values  # drop target and patient id

    # get model
    theta_model = models.get_qreg_model(train_x.shape[1])  # input vector n_dim is n columns

    # add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_score', mode='auto',
                                                      restore_best_weights=True, patience=70, verbose=1)
    lr_schedule = get_lr_callback(batch_size=THETA_BATCH_SIZE, epochs=THETA_EPOCHS, plot=False)

    # train
    if without_validation:
        history = theta_model.fit(x=np.concatenate([train_x, val_x]), y=np.concatenate([train_y, val_y]),
                                  epochs=THETA_EPOCHS, batch_size=THETA_BATCH_SIZE, shuffle=True,
                                  callbacks=[lr_schedule])
    else:
        history = theta_model.fit(x=train_x, y=train_y, epochs=THETA_EPOCHS, batch_size=THETA_BATCH_SIZE,
                                  validation_data=(val_x, val_y), shuffle=True, callbacks=[lr_schedule])

    # save model
    theta_model.save_weights(save_path)

    return history.history


if __name__ == '__main__':
    hist = train_qreg_model('models_weights/qreg_model/model_v4.ckpt', pp_train_data='theta_data/pp_train.csv',
                            without_validation=True)


# v6 is good 256x256
# v4 is good default qreg
