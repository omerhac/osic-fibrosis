import image_data
import table_data
import tensorflow as tf
import numpy as np
import tf_record_writer
import pandas as pd
import predict
from itertools import chain
from TablePreprocessor import TablePreprocessor
import pickle

AUTO = tf.data.experimental.AUTOTUNE

# GCS PATH to images
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'

# images size
IMAGE_SIZE = [256, 256]

# GCS tfrecords path
TF_RECORDS_PATH = 'gs://osic_fibrosis/tfrecords-jpeg-512x512-exp-norm-outliers'


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


def get_tfrecord_dataset(image_size=IMAGE_SIZE, type='train'):
    """Read from TFRecords. For optimal performance, read from multiple
    TFRecord files at once and set the option experimental_deterministic = False
    to allow order-altering optimizations.
    Args:
        image_size--common image size in dataset
    """

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    # get files according to type
    filenames = tf.io.gfile.glob(TF_RECORDS_PATH + '/' + type + '/*.tfrec')

    train_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    train_dataset = train_dataset.with_options(option_no_order)

    # add resizing capability
    read_tfrecord_and_resize = lambda example: tf_record_writer.read_tfrecord(example, image_size=image_size)

    train_dataset = train_dataset.map(read_tfrecord_and_resize, num_parallel_calls=AUTO)

    return train_dataset


def create_nn_train(model_path='models_weights/cnn_model/model_v3.ckpt',
                    processor_save_path=None,
                    enlarged_model=False,
                    image_size=IMAGE_SIZE):
    """Create NN train table for finding optimal theta.
    Args:
        model_path: path to CNN model weights to predict the fvc for each week
        processor_save_path: path to save pickled preprocessor
        enlarged_model: flag whether the provided model is an enlarged one
        image_size: feeding image size to the cnn model
    """

    data = table_data.get_train_table()

    # remove patient with corrupted images
    data = data[data["Patient"] != 'ID00011637202177653955184']

    # get predictions on train and val set
    data["GT_FVC"] = data["FVC"]
    train_exp_gen = predict.exponent_generator(IMAGES_GCS_PATH + '/train', model_path=model_path,
                                               for_test=False,
                                               enlarged_model=enlarged_model,
                                               image_size=image_size)  # train gen
    val_exp_gen = predict.exponent_generator(IMAGES_GCS_PATH + '/validation', model_path=model_path,
                                             for_test=False,
                                             enlarged_model=enlarged_model,
                                             image_size=image_size)  # validation gen

    exp_dict = {id: exp_func for id, exp_func in chain(train_exp_gen, val_exp_gen)}  # get exponential functions dict

    # predict
    predict.predict_form(exp_dict, data, submission=False)  # this sets the table FVC values to CNN predictions

    # add base FVC and week, predicted percent columns
    data = table_data.get_initials(data)
    data = table_data.get_predicted_percent(data)
    
    # get exponent coeffs
    for index, row in data.iterrows():
        coeff = exp_dict[row["Patient"]].get_coeff()  # get the exponential coeff of every patient
        data.loc[index, "Coeff"] = coeff

    # preprocess
    processor = TablePreprocessor()
    processor.fit(data)
    data = processor.transform(data)

    # sort columns
    data = data.sort_index(axis=1)

    # save pre processor
    if processor_save_path:
        pickle.dump(processor, open(processor_save_path, 'wb'))

    return data


def get_train_val_split():
    """Return a list of ids of train patients and a list of ids of validation patients"""
    train_ids = []
    val_ids = []

    # get image dataset of train patients
    train_image_dataset = image_data.get_images_dataset_by_id(IMAGES_GCS_PATH + '/train')
    val_image_dataset = image_data.get_images_dataset_by_id(IMAGES_GCS_PATH + '/validation')

    for patient, images in train_image_dataset:
        train_ids.append(patient.numpy().decode('utf-8'))

    for patient, images in val_image_dataset:
        val_ids.append(patient.numpy().decode('utf-8'))

    return train_ids, val_ids


def create_nn_test(test_table, processor, test_images_path=IMAGES_GCS_PATH + '/test',
                   cnn_model_path='models_weights/cnn_model/model_v3.ckpt',
                   exp_gen=None,
                   enlarged_model=False,
                   image_size=IMAGE_SIZE):
    """Create test table for NN predictions.
    Args:
        test_table: DataFrame with test patients data
        processor: TablePreprocessor instance fit on train set for preprocessing the test data
        test_images_path: path to directory with test images. To generate CNN predictions and suitable prediction form
        cnn_model_path: path to cnn model used to predict test data
        exp_gen: exponent functions generator. This function will create one if its not provided
        enlarged_model: flag whether the provided model is an enlarged one
        image_size: feeding image size to the cnn model
    """

    # get standard form
    weekly_data = predict.create_submission_form(images_path=test_images_path)
    weekly_data["Patient"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[0])
    weekly_data["Weeks"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[1]).astype('float32')

    # predict weekly fvc
    if not exp_gen:
        exp_gen = predict.exponent_generator(test_images_path,
                                             model_path=cnn_model_path,
                                             for_test=True,
                                             enlarged_model=enlarged_model,
                                             image_size=image_size)
    exp_dict = {id: exp_func for id, exp_func in exp_gen}
    predict.predict_form(exp_dict, weekly_data)

    # get initial fvc and week column
    test_table["Initial_Week"] = test_table["Weeks"]
    test_table["Initial_FVC"] = test_table["FVC"]

    # merge
    data = test_table.drop(["Weeks", "FVC"], axis=1).merge(weekly_data, on="Patient")

    # get norm weeks column
    data["Norm_Week"] = data["Weeks"] - data["Initial_Week"]

    # get exponent coeffs
    for index, row in data.iterrows():
        coeff = exp_dict[row["Patient"]].get_coeff()  # get the exponential coeff of every patient
        data.loc[index, "Coeff"] = coeff

    # remove unused features
    data = data.drop(["Patient_Week", "Confidence"], axis=1)

    # preprocess
    data = processor.transform(data)

    # sort columns
    data = data.sort_index(axis=1)

    return data


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pp_train = create_nn_train(model_path='models_weights/cnn_model/model_v4.ckpt',
                               enlarged_model=True,
                               image_size=[512, 512],
                               processor_save_path='models_weights/qreg_model/processor.pickle')
    pp_test = create_nn_test(table_data.get_test_table(),
                             pickle.load(open('models_weights/qreg_model/processor.pickle', 'rb')),
                             cnn_model_path='models_weights/cnn_model/model_v4.ckpt',
                             enlarged_model=True,
                             image_size=[512, 512])
    pp_train.to_csv('pp_train.csv', index=False)
    pp_test.to_csv('pp_test.csv', index=False)
