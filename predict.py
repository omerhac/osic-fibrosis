import image_data
import tensorflow as tf
import table_data
import numpy as np
import visualize

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# image size
IMAGE_SIZE = (224, 224)


def exponent_generator(path):
    """Create a generator which returns exponent function for patients whose images are at path.
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
        coeff_sum = 0  # maintain sum
        # iterate threw every image
        for n, image in enumerate(images):
            # predict exponent
            exp_coeff = network.predict(image)[0][0]
            coeff_sum += exp_coeff

        # average predictions
        avg_coeff = coeff_sum / n

        # create exponent
        id = patient.numpy().decode('utf-8')
        initial_week, initial_fvc = table_data.get_initial_fvc(id)
        func = lambda week: initial_fvc * np.exp(-avg_coeff * (week - initial_week))

        yield id, func


if __name__ == '__main__':
    g = exponent_generator(IMAGES_GCS_PATH + '/validation')
    id, exp_func = next(g)
    visualize.plot_patient_exp(id, exp_function=exp_func)