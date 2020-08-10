import time
import image_data
import tensorflow as tf
import numpy as np
import etl
import tf_record_writer

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# images size
IMAGE_SIZE = [512, 512]


def time_images_dataset_by_id_fetching():
    """Time 5 full image retrival from patient records"""
    ds = image_data.get_images_dataset_by_id(IMAGES_GCS_PATH + '/train')

    for id, images in ds.take(5):
        t = time.time()
        print(id.numpy())
        for image in images:
            pass
        print(time.time() - t)

    print("--finish--")


def test_images_dataset():
    """Test tf dataset of (id, image)"""
    images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/train')

    for id, image in images_dataset.take(3):
        assert tf.is_tensor(id), "ID should be a tensor"
        assert image.numpy().shape == (*IMAGE_SIZE, 3), "Images size should be {}".format((*IMAGE_SIZE, 3))

    print("**TEST PASSED!**")


def test_dataset_creation():
    """Test full dataset, as np memmap saved at data"""
    train_x = np.memmap('data/train_x.dat', shape=(32994, *IMAGE_SIZE, 3), dtype='uint8', mode='r')
    train_y = np.memmap('data/train_y.dat', shape=(32994, 3), dtype='float32', mode='r')

    for i in range(20):
        assert train_x[i].shape == (*IMAGE_SIZE, 3), train_x[i].shape
        assert train_y[i].shape == (3, ), train_y[i].shape

    print("**TEST PASSED!**")


def test_tfrecords_dataset():
    """Test tfrcords format on the cloud"""
    tfrecords = etl.get_tfrecord_dataset()
    for image, k in tfrecords.take(5):
        #assert isinstance(id.numpy().decode('utf-8'), str)
        assert k.numpy().dtype == np.float32, k.numpy().dtype
        assert image.numpy().shape == (*IMAGE_SIZE, 3), image.numpy().shape

    print("*TEST PASSED!**")