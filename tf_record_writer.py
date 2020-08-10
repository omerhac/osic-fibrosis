import image_data
import table_data
import tensorflow as tf
import numpy as np
import tests
import math
import os

# ENVIRONMENT
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/gcloud_key.json"  # gcs authentication
AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# GCS tfrecords path
TF_RECORDS_PATH = 'gs://osic_fibrosis/tfrecords-jpeg-512x512'

# images size
IMAGE_SIZE = [512, 512]

# shards size
SHARDS = 16
SHARD_SIZE = math.ceil(1.0 * 33000 / SHARDS)


# Three types of data can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element will be a list of size 1


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(tfrec_filewriter, img_bytes, id, poly_coeffs):
    """Convert data to one tf.train.Example"""
    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "id": _bytestring_feature([id]),  # one id in the list
        "coeffs": _float_feature(poly_coeffs) # 3 floats in the list
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(for_val=False):
    """Write the train dataset to tfrecords format on the cloud.
    Args:
        for_val--flag to wirte validation data
    format:
            example['image']--bytestring image
            example['id']--bytestring id
            example['coeffs']--float list
    """

    # get data
    if for_val:
        images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/validation', decode_images=False)
    else:
        images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/train', decode_images=False)  # dont decode images for tfrecords writing

    poly_dict = table_data.get_poly_fvc_dict()  # polynomial coefficiants for every patient

    # batch
    images_dataset = images_dataset.batch(SHARD_SIZE)

    print("Writing TFRecords")
    for shard, (ids, images) in enumerate(images_dataset):
        # batch size used as shard size here
        shard_size = images.numpy().shape[0]
        # good practice to have the number of records in the filename
        if for_val:
            filename = TF_RECORDS_PATH + "/validation/{:02d}-{}.tfrec".format(shard, shard_size)
        else:
            filename = TF_RECORDS_PATH + "/{:02d}-{}.tfrec".format(shard, shard_size)

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                poly_coeffs = poly_dict[ids.numpy()[i].decode('utf-8')].tolist() # get patient coeffs

                example = to_tfrecord(out_file,
                                      images.numpy()[i],  # compressed image: already a byte string
                                      ids.numpy()[i],
                                      poly_coeffs)
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))


def read_tfrecord(example, image_size=IMAGE_SIZE):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "id": tf.io.FixedLenFeature([], tf.string),
        "coeffs": tf.io.FixedLenFeature([3], tf.float32)  # 3 floats
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    # FixedLenFeature fields are now ready to use: exmple['size']

    # decode image
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = image_data.resize_and_crop_image(image, image_size)  # crop to image_size
    image = tf.cast(image, tf.uint8)  # cast for memory efficiency

    # get id
    id = example['id']

    # get polynomial coeffs
    poly_coeffs = example['coeffs']

    return image, poly_coeffs # TODO: ADD ID TO RETURN


if __name__ == '__main__':
    write_tfrecords()
    write_tfrecords(for_val=True)
