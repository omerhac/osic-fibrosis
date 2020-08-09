import image_data
import table_data
import tensorflow as tf
import numpy as np
AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'


def create_train_dataset():
    """Create a numpy memory mapping of tuples of format (image, poly_coeffs).
    poly_coeffs is a (3, ) np.array with coefficients of the polynomial function
    describing the patient with image CT FVC progression.
    The mapping is stored at data/train.dat
    """

    # define train images path
    train_path = IMAGES_GCS_PATH + '/train'

    # get images dataset
    images_dataset = image_data.get_images_dataset(train_path)

    # get polynomial dict for each patient
    poly_dict = table_data.get_poly_fvc_dict()

    # create memory mapping
    dataset = np.memmap('data/train.dat', shape=(32994, 2), dtype=object, mode='w+')

    # iterate threw all images
    for i, (patient_id, image) in enumerate(images_dataset):
        # open memory mapping
        dataset = np.memmap('data/train.dat', shape=(32994, 2), dtype=object, mode='r+')

        # write data to numpy memory mapping
        decoded_id = patient_id.numpy().decode('utf-8')  # decode id
        dataset[i, 0] = image.numpy()  # get image from tensor
        dataset[i, 1] = poly_dict[decoded_id]  # get poly coefficients from dict

        # delete data to clear memory
        del dataset


if __name__ == "__main__":
    dataset = np.memmap('data/train.dat', shape=(32994, 2), dtype=object, mode='r')
    import matplotlib.pyplot as plt
    print(dataset[32993, 1])
    plt.imshow(dataset[32993, 0])

