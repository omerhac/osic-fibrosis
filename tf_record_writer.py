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
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-hue'

# GCS tfrecords path
TF_RECORDS_PATH = 'gs://osic_fibrosis/tfrecords-jpeg-512x512-images-hue'

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


def to_tfrecord(tfrec_filewriter, img_bytes, id, exp_coeff):
    """Convert data to one tf.train.Example"""
    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "id": _bytestring_feature([id]),  # one id in the list
        "coeff": _float_feature(exp_coeff) # 1 floats in the list
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(type='train'):
    """Write the train dataset to tfrecords format on the cloud.
    Args:
        type--type of dataset, available values: train, validation, test
    format:
            example['image']--bytestring image
            example['id']--bytestring id
            example['coeff']--float list of one value
    """

    # check type
    assert type == 'train' or type == 'validation' or type =='test', "type should be train/validion/test"

    # get data
    images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/' + type, decode_images=False)  # dont decode images for tfrecords writing

    if type != 'test':  # not for test patients..
        exp_dict = table_data.get_exp_fvc_dict(remove_outliers=True)  # exponential coefficient for every patient

    # batch
    images_dataset = images_dataset.batch(SHARD_SIZE)

    print("Writing TFRecords")
    for shard, (ids, images) in enumerate(images_dataset):
        # batch size used as shard size here
        shard_size = images.numpy().shape[0]
        # good practice to have the number of records in the filename
        filename = TF_RECORDS_PATH + '/' + type + "/{:02d}-{}.tfrec".format(shard, shard_size)

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                # different files for val/train and test
                if type != 'test':
                    exp_coeff = exp_dict[ids.numpy()[i].decode('utf-8')].tolist() # get patient coeffs
                    example = to_tfrecord(out_file,
                                          images.numpy()[i],  # compressed image: already a byte string
                                          ids.numpy()[i],
                                          [exp_coeff])  # must be in a list to be iterable

                else:
                    example = to_tfrecord(out_file,
                                          images.numpy()[i],  # compressed image: already a byte string
                                          ids.numpy()[i])

                # write file to cloud
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))


def read_tfrecord(example, image_size=IMAGE_SIZE):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "id": tf.io.FixedLenFeature([], tf.string),
        "coeff": tf.io.FixedLenFeature([], tf.float32)  # 1 floats
    }

    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    # FixedLenFeature fields are now ready to use: exmple['size']

    # decode image
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = image_data.resize_and_crop_image(image, image_size)  # crop to image_size
    image = tf.cast(image, tf.float32)  # cast for normalizing
    image = image_data.normalize_image(image)  # normalize image

    # get id
    id = example['id']

    # get polynomial coeffs
    exp_coeff = example['coeff']

    return image, exp_coeff  # TODO: ADD ID TO RETURN


if __name__ == '__main__':
    write_tfrecords(type='train')
    write_tfrecords(type='validation')
    #write_tfrecords(type='test')

