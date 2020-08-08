import time
import image_data
import tensorflow as tf

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# images size
IMAGE_SIZE = [512, 512]


def time_dataset_fetching():
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
    images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/train')

    for id, image in images_dataset.take(3):
        assert tf.is_tensor(id), "ID should be a tensor"
        assert image.numpy().shape == (*IMAGE_SIZE, 3), "Images size should be {}".format((*IMAGE_SIZE, 3))

    print("**TEST PASSED!**")


