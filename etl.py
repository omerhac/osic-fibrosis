import image_data
import table_data
import tensorflow as tf
import numpy as np
import tf_record_writer
import tests
AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# images size
IMAGE_SIZE = [512, 512]

# GCS tfrecords path
TF_RECORDS_PATH = 'gs://osic_fibrosis/tfrecords-jpeg-512x512'

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

    # create memory mappings
    train_x = np.memmap('data/train_x.dat', shape=(32994, *IMAGE_SIZE, 3), dtype='uint8', mode='w+')
    train_y = np.memmap('data/train_y.dat', shape=(32994, 3), dtype='float32', mode='w+')

    # iterate threw all images
    for i, (patient_id, image) in enumerate(images_dataset):
        # open memory mappings
        train_x = np.memmap('data/train_x.dat', shape=(32994, *IMAGE_SIZE, 3), dtype='uint8', mode='r+')
        train_y = np.memmap('data/train_y.dat', shape=(32994, 3), dtype='float32', mode='r+')

        # write data to numpy memory mapping
        decoded_id = patient_id.numpy().decode('utf-8')  # decode id
        train_x[i, :, :, :] = image.numpy()  # get image from tensor
        train_y[i, :] = poly_dict[decoded_id]  # get poly coefficients from dict

        # delete data to clear memory
        del train_x
        del train_y


def create_test_dataset():
    """Create a numpy memory mapping of test set images.
    The dataset will be of format (id, image of that id).
    The mapping is stored at data/test.dat
    """

    # define test images path
    test_path = IMAGES_GCS_PATH + '/test'

    # get images dataset
    images_dataset = image_data.get_images_dataset(test_path)

    # create memory mapping
    test_images = np.memmap('data/test_images.dat', shape=(32994, *IMAGE_SIZE, 3), dtype='uint8', mode='w+')
    test_ids = np.memmap('data/test_ids.dat', shape=(32994, 1), dtype='<U25', mode='w+')

    # iterate threw all images
    for i, (patient_id, image) in enumerate(images_dataset):
        # open memory mapping
        test_images = np.memmap('data/test_images.dat', shape=(32994, *IMAGE_SIZE, 3), dtype='uint8', mode='r+')
        test_ids = np.memmap('data/test_ids.dat', shape=(32994, 1), dtype='<25U', mode='r+')

        # write data to numpy memory mapping
        decoded_id = patient_id.numpy().decode('utf-8')  # decode id
        test_images[i, :, :, 3] = image.numpy()  # save id
        test_ids[i] = decoded_id  # get image from tensor

        # delete data to clear memory
        del test_images
        del test_ids


def get_tfrecord_dataset(image_size=IMAGE_SIZE, validation=False):
    """Read from TFRecords. For optimal performance, read from multiple
    TFRecord files at once and set the option experimental_deterministic = False
    to allow order-altering optimizations.
    Args:
        image_size--common image size in dataset
    """

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    # check whether for validation
    if validation:
        filenames = tf.io.gfile.glob(TF_RECORDS_PATH + "/val/*.tfrec")
    else:
        filenames = tf.io.gfile.glob(TF_RECORDS_PATH + "/*.tfrec")

    train_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    train_dataset = train_dataset.with_options(option_no_order)

    # add resizing capability
    read_tfrecord_and_resize = lambda example: tf_record_writer.read_tfrecord(example, image_size=image_size)

    train_dataset = train_dataset.map(read_tfrecord_and_resize, num_parallel_calls=AUTO)

    return train_dataset


if __name__ == "__main__":
    train_data = get_tfrecord_train_dataset(image_size=(256, 256))
    for image, coeffs in train_data.take(2):
        print(image.numpy().shape)

