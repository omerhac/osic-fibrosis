import image_data
import table_data
import tensorflow as tf
import numpy as np
AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'


def get_train_dataset():
    """Return an array of tuples of format (image, poly_coeffs).
    poly_coeffs is a (3, ) np.array with coefficients of the polynomial function
    describing the patient with image CT FVC progression.
    """

    # define train images path
    train_path = IMAGES_GCS_PATH + '/train'

    # get images dataset
    images_dataset = image_data.get_images_dataset(train_path)

    # get polynomial dict for each patient
    poly_dict = table_data.get_poly_fvc_dict()

    dataset = []
    # iterate threw all images
    for patient_id, image in images_dataset.take(5):
        decoded_id = patient_id.numpy().decode('utf-8')
        dataset.append((image.numpy(), poly_dict[decoded_id]))
 
    return np.stack(dataset)


def create_train_dataset():
    """Create train dataset and save it to train.npz"""
    train_dataset = get_train_dataset()

    # save
    np.savez_compressed('data/train.npz', train=train_dataset)


# TODO: delete this
if __name__ == "__main__":
    create_train_dataset()
    """train_dataset = np.load('data/train.npz', allow_pickle=True)
    for k in train_dataset.keys():
        print(train_dataset['a'][0,0].shape)
        print(train_dataset['a'][0,1])"""