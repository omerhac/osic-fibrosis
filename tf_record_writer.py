import image_data
import table_data
import tensorflow as tf
import numpy as np
import tests
AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# GCS tfrecords path
TF_RECORDS_PATH = 'gs://osic_fibrosis/tfrecords-jpeg-512x512'

# images size
IMAGE_SIZE = [512, 512]


# Three types of data can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element will be a list of size 1

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(tfrec_filewriter, img_bytes, label, height, width):
    class_num = np.argmax(np.array(CLASSES) == label)  # 'roses' => 2 (order defined in CLASSES)
    one_hot_class = np.eye(len(CLASSES))[class_num]  # [0, 0, 1, 0, 0] for class #2, roses

    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "class": _int_feature([class_num]),  # one class in the list

        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label": _bytestring_feature([label]),  # fixed length (1) list of strings, the text label
        "size": _int_feature([height, width]),  # fixed length (2) list of ints
        "one_hot_class": _float_feature(one_hot_class.tolist())  # variable length  list of floats, n=len(CLASSES)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_train_tfrecords():

    images_dataset = image_data.get_images_dataset(IMAGES_GCS_PATH + '/train')

    print("Writing TFRecords")
    for shard, (id, image) in enumerate(images_dataset):
        # batch size used as shard size here
        shard_size = image.numpy().shape[0]
        # good practice to have the number of records in the filename
        filename = GCS_OUTPUT + "{:02d}-{}.tfrec".format(shard, shard_size)

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                example = to_tfrecord(out_file,
                                      image.numpy()[i],  # re-compressed image: already a byte string
                                      label.numpy()[i],
                                      height.numpy()[i],
                                      width.numpy()[i])
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))