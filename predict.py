import image_data
import tensorflow as tf
import table_data
import numpy as np
import visualize
import pandas as pd
import models
import pickle
import etl

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-hue/images-hue'

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

    def get_coeff(self):
        """Return the exponential coefficient of the function"""
        return self._exponential_coefficient


def exponent_generator(path, for_test=False, model_path='models_weights/cnn_model/model_v2.ckpt'):
    """Create a generator which returns exponent function for patients whose images are at path.
    Take a dataset of patient directories. Generate an exponent coefficient describing
    FVC progression for each patient CT image. Average those coefficients and return an
    exponent function and the id of the patient.

    Args:
        path--path to the directory with the images
        for_test--flag if the generator is for the test set
        model_path--path to models_weights
    """

    image_dataset = image_data.get_images_dataset_by_id(path)

    # get model
    network = models.get_sqeezenet_model()
    network.load_weights(model_path)

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


def predict_test(save_path, test_table, test_path=IMAGES_GCS_PATH + '/test',
                 cnn_model_path='models_weights/cnn_model/model_v2.ckpt',
                 qreg_model_path='models_weights/qreg_model/model_v1.ckpt',
                 exp_gen=None, processor_path='models_weights/qreg_model/processor.pickle'):
    """Predict test set and generate a submission file.
    Args:
        save_path: where to save predictions
        test_path: path to test images
        test_table: DataFrame with test patients data
        cnn_model_path: path to cnn model weights -- only needed if exponent generator is not provided
        qreg_model_path: path to quantile regression model weights
        exp_gen: a generator for exponent functions based on cnn predictions, this function will create one
        if its not provided
        processor_path: path to pickled preprocessor for table data
    """

    # get generator
    if not exp_gen:
        exp_gen = exponent_generator(test_path, for_test=True, model_path=cnn_model_path)

    # get preprocessor
    processor = pickle.load(open(processor_path, 'rb'))

    # get processed test table
    test_data = etl.create_nn_test(test_table, processor, test_images_path=test_path, exp_gen=exp_gen)

    # get submission form format
    submission = test_data[["Patient", "Weeks"]]
    test_data = test_data.drop(["Patient"], axis=1).astype('float32').values

    # get model
    model = models.get_qreg_model(test_data.shape[1])
    model.load_weights(qreg_model_path)

    # predict
    preds = model.predict(test_data)
    submission["FVC"] = preds[:, 1]  # fvc prediction is the median prediction
    submission["Confidence"] = (preds[:, 2]-preds[:, 0])  # confidence prediction is (top quant - bottom quant)

    # inverse transform weeks
    processor.inverse_transform(submission, "Weeks")
    submission["Weeks"] = submission["Weeks"].round(decimals=0).astype('int16')  # round weeks

    # combine weeks and patient
    submission["Patient_Week"] = submission["Patient"] + "_" + submission["Weeks"].astype('string')

    # remove redundant features
    submission = submission.drop(["Patient", "Weeks"], axis=1)

    # save
    submission.to_csv(save_path, index=False)


def predict_form(exp_dict, form, submission=True):
    """Predict FVC on a premade form of patient week couples.
    Args:
        exp_dict: dictionary with mapping id->exponent function
        form: pd table to predict
        submission: flag whether its a submission type from (has Patient_Week column)
    """

    for index, row in form.iterrows():

        # check whether its a submission type form
        if submission:
            id, week = row["Patient_Week"].split('_')

        else:
            id, week = row["Patient"], row["Weeks"]

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
    predict_test('submissions/sub_4.csv', table_data.get_test_table())
