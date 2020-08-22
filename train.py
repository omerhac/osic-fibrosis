import etl
import models
import tensorflow as tf
import visualize

# Constants
EPOCHS = 20
BATCH_SIZE = 16
STEPS_PER_EPOCH = 32994 // BATCH_SIZE
IMAGE_SIZE = (224, 224)


def train_model():
    """Train the model. Save weights to model_weights. Return history dict"""
    # get datasets
    train_dataset = etl.get_tfrecord_dataset(image_size=IMAGE_SIZE, type='train')
    val_dataset = etl.get_tfrecord_dataset(image_size=IMAGE_SIZE, type='validation')

    # batch and repeat dataset
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # get model
    network = models.get_sqeezenet_model(IMAGE_SIZE)

    # add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', mode='auto',
                                                      restore_best_weights=True, patience=3, verbose=1)
    # train
    history = network.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                          batch_size=BATCH_SIZE, validation_data=val_dataset, callbacks=[early_stopping])

    # save model
    network.save_weights('model_weights/model_v2.ckpt')

    return history.history


if __name__ == '__main__':
    hist = train_model()
    visualize.plot_training_curves(hist)
