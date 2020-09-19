import tensorflow as tf
import tests
import os
AUTO = tf.data.experimental.AUTOTUNE

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-hue'

# images size
IMAGE_SIZE = [256, 256]


def get_patient_directories_dataset(path):
    """Return a tf dataset with paths to each pf the patients directories"""
    dataset = tf.data.Dataset.list_files(path + '/*')
    return dataset


def get_images_paths_dataset(path):
    """Return a tf dataset with all of the images from path"""
    dataset = tf.data.Dataset.list_files(path + '/*/*')
    return dataset


def get_images_dataset_by_id(path, image_size=IMAGE_SIZE):
    """Create images dataset partitioned by id from gcs path.
    The dataset contains tuples of the format (tensor(id), dataset of images of that id)
    """

    directories_dataset = get_patient_directories_dataset(path)

    # helper funciton
    def read_images(patient_directory):
        """Return a tuple of patient id and his images.
        patient id is a string tensor.
        patient images is a dataset of images resized and cropped to IMAGE_SIZE.

        Args:
            patient_directory--patient directory of files
        """
        # get patient id
        id = tf.strings.split(patient_directory, sep='/')[-1]

        # create images path
        images_paths = tf.strings.join([patient_directory, '/*'])

        # create dataset
        images = tf.data.Dataset.list_files(images_paths)
        images = images.map(read_image, num_parallel_calls=AUTO)  # read images
        images = images.map(lambda image: resize_and_crop_image(image, image_size), num_parallel_calls=AUTO)  # resize and crop images

        return id, images

    images_dataset = directories_dataset.map(read_images, num_parallel_calls=AUTO)
    return images_dataset


def get_images_dataset(path, decode_images=True, image_size=IMAGE_SIZE):
    """Create images dataset from gcs path
    The dataset contains tuples of format (tensor(id), one image of that id)

    Args:
        path--path to images directory
        decode_images--whether to decode the images, defaults to True
    """

    image_path_dataset = get_images_paths_dataset(path)

    # helper funciton
    def read_image_and_id(image_path):
        """Return a tuple of patient id and one of his images.
        patient id is a string tensor.
        patient image is a an image.
        """

        # get patient id
        id = tf.strings.split(image_path, sep='/')[-2]

        # read image
        image = read_image(image_path, decode=decode_images)

        if decode_images:
            image = resize_and_crop_image(image, image_size)
            image = tf.cast(image, tf.uint8)  # memory usage gonna be quite intense..

        return id, image

    images_dataset = image_path_dataset.map(read_image_and_id, num_parallel_calls=AUTO)
    return images_dataset


def resize_and_crop_image(image, new_size=IMAGE_SIZE):
    """ Resize and crop using "fill" algorithm:
    always make sure the resulting image
    is cut out from the source image so that
    it fills the TARGET_SIZE entirely with no
    black bars and a preserved aspect ratio.
    """

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = new_size[1]
    th = new_size[0]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w*tw/w, h*tw/w]),  # if true
                    lambda: tf.image.resize(image, [w*th/h, h*th/h])   # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image


def read_image(im_path, decode=True):
    """Read a jpeg image from image path

    Args:
        im_path--path to image
        decode--whether to decode the image
    """

    if decode:
        return tf.io.decode_jpeg(tf.io.read_file(im_path))
    else:
        return tf.io.read_file(im_path)


def normalize_image(image):
    """Return a sacled by 255 image"""
    norm_image = image / 255
    return norm_image


# credit to Carlos Sueza from https://www.kaggle.com/carlossouza/end-to-end-model-ct-scans-tabular
def get_morphological_mask(image, threshold):
    m = image < threshold
    m = clear_border(m)
    m = label(m)
    areas = [r.area for r in regionprops(m)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(m):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    m[coordinates[0], coordinates[1]] = 0
    return m > 0


# TODO: delete this
if __name__ == "__main__":
    pass
