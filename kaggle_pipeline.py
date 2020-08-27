import pandas as pd
import image_data
import os
import glob
import matplotlib.pyplot as plt
import models
import predict

# images size
IMAGE_SIZE = [224, 224]


def get_jpeg_image(dcm_path):
    """Return jpeg image from dcm file with dcm_path, pixels range is [0, 255]"""
    dcm = pydicom.dcmread(dcm_path)
    pixels = dcm.pixel_array
    plt.imsave('image.jpeg', pixels)
    image = plt.imread('image.jpeg')
    return image


def patient_image_generator(patient_path, image_size=IMAGE_SIZE):
    """Generate one image from patient_path dir. Resize and crop image to image size"""
    image_paths = glob.glob(patient_path + '/*')

    # generate images
    for image_path in image_paths:
        image = get_jpeg_image(image_path)
        resized_image = image_data.resize_and_crop_image(image).numpy().astype('uint8')  # TODO: every image preprocessing step should come here to
        yield resized_image


def test_generator(test_dir):
    """Generate a tuple of (ID, patient_images) where patient_images is a generator that yields patient images."""
    patient_dirs = glob.glob(test_dir + '/*')
    for patient in patient_dirs:
        id = os.path.basename(patient)
        yield id, patient_image_generator(patient)


def get_initial_fvc(id, test_table_path):
    """Return the week number and FVC value of the first measurement. Test table csv at test_table_path"""
    table = pd.read_csv(test_table_path)
    return float(table.loc[table["Patient"] == id]["Weeks"]), float(table[table["Patient"] == id]["FVC"])


def exponent_generator(path, model_path='models_weights/model_v2.ckpt', test_table_path='test.csv'):
    """Create a generator which returns exponent function for patients whose images are at path.
    Take a dataset of patient directories. Generate an exponent coefficient describing
    FVC progression for each patient CT image. Average those coefficients and return an
    exponent function and the id of the patient.

    Args:
        path--path to the directory with the images
        for_test--flag if the generator is for the test set
        model_path--path to models_weights
    """

    image_dataset = test_generator(path)

    # get model
    network = models.get_sqeezenet_model()
    network.load_weights(model_path)

    # iterate threw every patient
    for patient, images in image_dataset:
        coeff_sum = 0  # maintain sum
        # iterate threw every image
        for n, image in enumerate(images):
            # batch for model digestion
            image = image[None]
            # predict exponent
            exp_coeff = network.predict(image)[0][0]
            coeff_sum += exp_coeff

        # average predictions
        avg_coeff = coeff_sum / n

        # create exponent
        initial_week, initial_fvc = get_initial_fvc(patient, test_table_path=test_table_path)
        exp_func = predict.ExpFunc(initial_fvc, avg_coeff, initial_week)  # create exponent function
        yield patient, exp_func