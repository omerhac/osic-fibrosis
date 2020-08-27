import etl
import models
import tensorflow as tf
import visualize

# CNN Constants
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 16
CNN_STEPS_PER_EPOCH = 32994 // CNN_BATCH_SIZE
CNN_IMAGE_SIZE = (224, 224)

# Theta constants
THETA_EPOCHS = 20
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
    dataset = etl.create_nn_train(model_path=cnn_model_path)
    train_ids, val_ids = etl.get_train_val_split()
    train_dataset = dataset.loc[dataset["Patient"].isin(train_ids)]  # get train rows
    val_dataset = dataset.loc(dataset["Patient"].isin(val_ids))  # get validation rows

    # split target
    train_y = train_dataset["Theta"]
    val_x = val_dataset["Theta"]
    train_x = train_dataset.drop(["Theta"], axis=1)
    val_y = val_dataset.drop(["Theta"], axis=1)

    # get model
    theta_model = models.get_theta_model(len(dataset.columns))

    # add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', mode='auto',
                                                      restore_best_weights=True, patience=3, verbose=1)

    # train
    history = theta_model.fit(x=train_x, y=train_y, epochs=THETA_EPOCHS, batch_size=THETA_BATCH_SIZE,
                              validation_data=(val_x, val_y), shuffle=True, callbacks=[early_stopping])

    # save model
    theta_model.save_weights(save_path)

    return history.history


if __name__ == '__main__':
    hist = train_cnn_model()
    visualize.plot_training_curves(hist)
