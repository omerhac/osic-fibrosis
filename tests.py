import time
import image_data

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# images size
IMAGE_SIZE = [512, 512]


def time_dataset_fetching():
    """Time 5 full image retrival from patient records"""
    ds = image_data.get_images_dataset(IMAGES_GCS_PATH + '/train')

    for id, images in ds.take(5):
        t = time.time()
        print(id.numpy())
        for image in images:
            pass
        print(time.time() - t)

