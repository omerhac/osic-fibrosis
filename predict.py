import image_data
import tensorflow as tf
import table_data
import numpy as np
import visualize
import pandas as pd
import models

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# image size
IMAGE_SIZE = (224, 224)


class ExpFunc:
    """A class for describing an exponent function
    Attributes:
        _initial_value--initial value to start decay
        _exponent_coefficient--k value from A*e^-kt
        _shift--a constant shift to center time to
    """

    def __init__(self, initial_value, exponential_coeff, shift):
        self._initial_value = initial_value
        self._exponential_coefficient = exponential_coeff
        self._shift = shift

    def __call__(self, time):
        return self._initial_value * np.exp(-self._exponential_coefficient * (time - self._shift))


def exponent_generator(path, for_test=False):
    """Create a generator which returns exponent function for patients whose images are at path.
    Take a dataset of patient directories. Generate an exponent coefficient describing
    FVC progression for each patient CT image. Average those coefficients and return an
    exponent function and the id of the patient.

    Args:
        path--path to the directory with the images
        for_test--flag if the generator is for the test set
    """

    image_dataset = image_data.get_images_dataset_by_id(path)

    # get model
    network = models.get_sqeezenet_model()
    network.load_weights('model_weights/model_v1.ckpt')

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
        initial_week, initial_fvc = table_data.get_initial_fvc(id, for_test=for_test)
        exp_func = ExpFunc(initial_fvc, avg_coeff, initial_week)  # create exponent function

        yield id, exp_func


def predict_test(save_path, test_path=IMAGES_GCS_PATH + '/test', new_submission_form=True):
    """Predict test set and generate a submission file
    Args:
        save_path: where to save predictions
        test_path: path to test images
        new_submission_form: flag whether to create the submission form from scratch. Else the form should be present
        at save path.
    """

    # get generator
    exp_gen = exponent_generator(test_path, for_test=True)

    # gather patient exponents
    exp_dict = {id: exp_func for id, exp_func in exp_gen}  # a dictionary with mapping patient -> FVC exponent function

    # get submission form
    if new_submission_form:
        create_submission_form(save_path=save_path, images_path=test_path)
    submission = pd.read_csv(save_path)

    # broadcast 50 Confidence level
    submission["Confidence"] = 50  # TODO: solve how to predict it...

    # predict FVC
    predict_form(exp_dict, submission)

    # save
    submission.to_csv(save_path, index=False)


def predict_form(exp_dict, form):
    """Predict FVC on a premade form of patient week couples"""
    for index, row in form.iterrows():
        id, week = row["Patient_Week"].split('_')
        week = float(week)

        # check whether the key exists
        if id in exp_dict:
            form.loc[index, "FVC"] = exp_dict[id](week)


def create_submission_form(save_path=None, images_path=IMAGES_GCS_PATH + '/test'):
    """Create a submission form to fill later"""
    image_dataset = image_data.get_images_dataset_by_id(images_path)

    # weeks to predict range
    weeks = range(-12, 134)

    # create form
    form = pd.DataFrame(columns=["Patient_Week", "FVC", "Confidence"], index=None)
    for id, images in image_dataset:
        id = id.numpy().decode('utf-8')
        for week in weeks:
            form = form.append({"Patient_Week": id + '_' + str(week), "FVC": 0, "Confidence": 0}, ignore_index=True)

    # save
    if save_path:
        form.to_csv(save_path, index=False)
    else:
        return form


if __name__ == '__main__':
    predict_test('submissions/sub_2.csv')