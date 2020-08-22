import models
import image_data
import tensorflow as tf

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'

# image size
IMAGE_SIZE = (224, 224)


def exponent_generator(path):
    """Create a generator which returns exponent function for patient whose images are at path.
    Take a dataset of patient directories. Generate an exponent coefficient describing
    FVC progression for each patient CT image. Average those coefficients and return an
    exponent function and the id of the patient.

    Args:
        path--path to the directory with the images
    """

    image_dataset = image_data.get_images_dataset_by_id(path)

    # get model
    network = tf.keras.models.load_model('model_weights/model')

    # iterate threw every patient
    for patient, images in image_dataset:
        images = images.batch(1)  # batch for model digestion
        # iterate threw every image
        for image in images:
            # predict exponent
            exp_coeff = network.predict(image)[0][0]
            yield exp_coeff


if __name__ == '__main__':
    g = exponent_generator(IMAGES_GCS_PATH + '/validation')
    print(next(g))