import etl
import model

# Constants
EPOCHS = 10
BATCH_SIZE = 16
STEPS_PER_EPOCH = 32994 // BATCH_SIZE
IMAGE_SIZE = (224, 224)


def train_model():
    """Train the model. Save weights to model_weights. Return history dict"""
    # get datasets
    train_dataset = etl.get_tfrecord_dataset(image_size=IMAGE_SIZE)
    val_dataset = etl.get_tfrecord_dataset(image_size=IMAGE_SIZE, validation=True)

    # batch and repeat dataset
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # get model
    network = model.get_model(IMAGE_SIZE)

    # train
    history = network.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                          batch_size=BATCH_SIZE, validation_data=val_dataset)

    # save model
    network.save('model_weights/model')

    return history.history


if __name__ == '__main__':
    train_model()
