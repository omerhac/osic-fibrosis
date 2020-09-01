import etl
import models
import tensorflow as tf
import visualize
import pandas as pd
import numpy as np

# CNN Constants
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 16
CNN_STEPS_PER_EPOCH = 32994 // CNN_BATCH_SIZE
CNN_IMAGE_SIZE = (224, 224)

# Theta constants
THETA_EPOCHS = 1500
THETA_BATCH_SIZE = 128
THETA_STEPS_PER_EPOCH = 32994 // CNN_BATCH_SIZE


def train_cnn_model(save_path):
    """Train the CNN model. Save weights to models_weights/cnn_model. Return history dict"""
    # get datasets
    train_dataset = etl.get_tfrecord_dataset(image_size=CNN_IMAGE_SIZE, type='train')
    val_dataset = etl.get_tfrecord_dataset(image_size=CNN_IMAGE_SIZE, type='validation')

    # batch and repeat dataset
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(CNN_BATCH_SIZE)
    val_dataset = val_dataset.batch(CNN_BATCH_SIZE)

    # get model
    network = models.get_sqeezenet_model(CNN_IMAGE_SIZE)

    # add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', mode='auto',
                                                      restore_best_weights=True, patience=3, verbose=1)
    # train
    history = network.fit(train_dataset, epochs=CNN_EPOCHS, steps_per_epoch=CNN_STEPS_PER_EPOCH,
                          batch_size=CNN_BATCH_SIZE, validation_data=val_dataset, callbacks=[early_stopping])

    # save model
    network.save_weights(save_path)

    return history.history


def train_theta_model(save_path, cnn_model_path='models_weights/cnn_model/model_v2.ckpt'):
    """Train the theta predicting model. Save weights to models_weights/theta_model. Return history dict"""
    # get datasets
    dataset = pd.read_csv('theta_data/pp_train.csv').drop(["Unnamed: 0"], axis=1)#etl.create_nn_train(model_path=cnn_model_path)
    train_ids, val_ids = etl.get_train_val_split()

    # cast dtypes
    numeric_columns = dataset.select_dtypes(include=np.number).columns
    dataset[numeric_columns] = dataset[numeric_columns].astype('float32')

    train_dataset = dataset.loc[dataset["Patient"].isin(train_ids)]  # get train rows
    val_dataset = dataset.loc[dataset["Patient"].isin(val_ids)]  # get validation rows

    # split target
    train_y = train_dataset["Theta"].values
    val_y = val_dataset["Theta"].values
    train_x = train_dataset.drop(["Theta", "Patient"], axis=1).values  # drop target and patient id
    val_x = val_dataset.drop(["Theta", "Patient"], axis=1).values  # drop target and patient id

    # get model
    theta_model = models.get_theta_model(train_x.shape[1])  # input vector n_dim is n columns

    # add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                                                      restore_best_weights=True, patience=3, verbose=1)

    # train
    history = theta_model.fit(x=train_x, y=train_y, epochs=THETA_EPOCHS, batch_size=THETA_BATCH_SIZE,
                              validation_data=(val_x, val_y), shuffle=True, callbacks=[early_stopping])

    # save model
    theta_model.save_weights(save_path)

    return history.history


if __name__ == '__main__':
    # hist = train_cnn_model()
    hist = train_theta_model('models_weights/theta_model/theta_model_v1')
    # visualize.plot_training_curves(hist)
    pd.set_option('display.max_columns', None)
    #dataset = pd.read_csv('theta_data/pp_train.csv').drop(["Unnamed: 0"], axis=1)
    #print(dataset.head(5))

