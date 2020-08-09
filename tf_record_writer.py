import image_data
import table_data
import tensorflow as tf
import numpy as np
import tests
import math
import os

# ENVIRONMENT
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/gcloud_key.json"
AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# GCS tfrecords path
TF_RECORDS_PATH = 'gs://osic_fibrosis/tfrecords-jpeg-512x512'

# images size
IMAGE_SIZE = [512, 512]

# shards size
SHARDS = 1000
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


def write_train_tfrecords():

    # get data
    images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/train', decode_images=False)  # dont decode images for tfrecords writing
    poly_dict = table_data.get_poly_fvc_dict()  # polynomial coefficiants for every patient

    # batch
    images_dataset = images_dataset.batch(SHARD_SIZE)

    print("Writing TFRecords")
    for shard, (ids, images) in enumerate(images_dataset):
        # batch size used as shard size here
        shard_size = images.numpy().shape[0]
        # good practice to have the number of records in the filename
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


if __name__ == '__main__':
    write_train_tfrecords()
