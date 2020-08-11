import image_data
import tensorflow as tf
import table_data
import numpy as np
import visualize

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# image size
IMAGE_SIZE = (224, 224)


class ExpFunc:
    """A class for describing an exponent function
    Attributes:
        _initial_value--initial value to start decay
        _exponent_coefficient--k value from A*e^-kt
    """
    def __init__(self, initial_value, exponential_coeff):
        self._initial_value = initial_value
        self._exponential_coefficient = exponential_coeff

    def __call__(self, time):
        return self._initial_value * np.exp(-self._exponential_coefficient * time)
    

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
        initial_week, initial_fvc = table_data.get_initial_fvc(id, for_test=for_test)
        print(initial_week, initial_fvc, avg_coeff)

        def exp_func(week):
            return initial_fvc * np.exp(-avg_coeff * (week - initial_week))
        e = exp_func
        yield id, e


def predict_test(save_path):
    """Predict test set and generate a submission file"""
    # test images path
    test_path = IMAGES_GCS_PATH + '/test'

    # get generator
    exp_gen = exponent_generator(test_path, for_test=True)

    # gather patient exponents
    exp_dict = {id: exp_func for id, exp_func in exp_gen}  # a dictionary with mapping patient -> FVC exponent function

    # get submission form
    submission = table_data.get_submission_table()

    # broadcast 50 Confidence level
    submission["Confidence"] = 50  # TODO: solve how to predict it...

    # predict FVC
    for index, row in submission.iterrows():
        id, week = row["Patient_Week"].split('_')
        week = float(week)
        # predict
        print(exp_dict[id](week))
        submission.loc[index, "FVC"] = exp_dict[id](week)

    # save
    submission.to_csv(save_path)


if __name__ == '__main__':
    #predict_test('submissions/sub_1.csv')
    #sub = table_data.get_submission_table()
    #sub.loc[0, "FVC"] = 0
    #print(sub.head(5))
    g = exponent_generator(IMAGES_GCS_PATH + '/train')
    id1, f1 = next(g)
    id2, f2 = next(g)
    print(id1, f1(14))
    print(id2, f2(14))