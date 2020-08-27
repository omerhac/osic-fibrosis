import pandas as pd
import numpy as np
import visualize
import predict
import image_data

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'


def get_train_table():
    """Return a dataframe with train records"""
    return pd.read_csv('train.csv')


def get_test_table():
    """Return a dataframe with test records"""
    return pd.read_csv('test.csv')


def get_submission_table():
    """Return a dataframe with submission format"""
    return pd.read_csv('submissions/sample_submission.csv')


def get_fvc_hist(table, id):
    """Return the FVC hist of a patient wit ID [id] from [table]"""
    id_table = table[table["Patient"] == id].sort_values("Weeks")
    return id_table[["Weeks", "FVC"]]


def get_percent_hist(table, id):
    """Return the Percent hist of a patient wit ID [id] from [table]"""
    id_table = table[table["Patient"] == id].sort_values("Percent")
    return id_table[["Weeks", "Percent"]]


def get_patient_fvc_poly(table, id, order=2):
    """Fit a polynomial regression threw patients FVC checks. Return the coefficients.

        Args:
        table--data table
        id--patients id
        order--polynomial degree
    """
    hist = get_fvc_hist(table, id)  # get fvc hist
    coeffs = np.polyfit(hist["Weeks"], hist["FVC"], deg=order)
    return coeffs


def get_patient_percent_poly(table, id, order=2):
    """Fit a polynomial regression threw patients Percent checks. Return the coefficients.

    Args:
        table--data table
        id--patients id
        order--polynomial degree
    """
    hist = get_percent_hist(table, id)  # get fvc hist
    coeffs = np.polyfit(hist["Weeks"], hist["Percent"], deg=order)
    return coeffs


def get_poly_fvc_dict():
    """Return a dict with a mapping from id to their polynomial fvc estimation coeffs"""
    table = get_train_table()
    polys_dict = {}

    # get unique ids
    ids = table["Patient"].unique()

    # iterate threw every id
    for id in ids:
        poly = get_patient_fvc_poly(table, id, order=2)
        polys_dict[id] = poly

    return polys_dict


def get_patient_fvc_exp(table, id):
    """Get a patient exponential coefficient.
    Each patient has his own fvc progression that can be described by Ie^-kt.
    Wheres I is the initial FCV measure, k is the exponential coefficient and t is time.
    Args:
        table--data table with records
        id--id of the patient
    """

    hist = get_fvc_hist(table, id)

    # compute logs
    weeks = hist["Weeks"]
    log_fvc = np.log(hist["FVC"])

    # center weeks - set 0 for first measurement
    weeks = weeks - hist.iloc[0, 0]

    # regress to find logI -kt
    neg_k, logI = np.polyfit(weeks, log_fvc, deg=1)

    return -neg_k


def get_exp_fvc_dict():
    """Return a dict with a mapping from id to their exponential fvc estimation coefficient"""
    table = get_train_table()
    exp_dict = {}

    # get unique ids
    ids = table["Patient"].unique()

    # iterate threw every id
    for id in ids:
        k = get_patient_fvc_exp(table, id)
        exp_dict[id] = k

    return exp_dict


def get_exp_function_dict():
    """Create ground truth exponent functions dict for train records"""
    exp_dict = get_exp_fvc_dict()

    # create exponent functiosn dict
    func_dict = {}
    for id in exp_dict:
        i_week, i_fvc = get_initial_fvc(id)
        func_dict[id] = predict.ExpFunc(i_fvc, exp_dict[id], i_week)

    return func_dict


def get_initial_fvc(id, for_test=False):
    """Return the week number and FVC value of the first measurement"""
    if for_test:
        table = get_test_table()
        return float(table.loc[table["Patient"] == id]["Weeks"]), float(table[table["Patient"] == id]["FVC"])

    else:
        table = get_train_table()
        hist = get_fvc_hist(table, id)
        return hist.iloc[0, 0], hist.iloc[0, 1]


def normalize_feature(table, feature):
    """Normalize a numeric feature in pandas DataFrame"""
    # get stats
    min = table[feature].min()
    max = table[feature].max()
    mean = table[feature].mean()

    # transform
    table[feature] = (table[feature] - min) / (max - min)


def preprocess_table_for_nn(table):
    """Prepare table data for NN digestion. One hot encode categorical data."""
    # one hot encode categorical
    sex = pd.get_dummies(table["Sex"], prefix='Sex')  # one hot encode sex variable
    smoking_status = pd.get_dummies(table["SmokingStatus"], prefix="SmokingStatus")  # one hot encode smokingstatus

    # concat
    ohe_table = pd.concat([table, sex, smoking_status], axis=1).drop(["Sex", "SmokingStatus"], axis=1)

    # normalize numeric columns
    normalize_feature(ohe_table, "Weeks")
    normalize_feature(ohe_table, "FVC")
    normalize_feature(ohe_table, "Percent")
    normalize_feature(ohe_table, "Age")

    return ohe_table


def patient_week_exists(patient, week):
    """Check of a patient week couple exists in train records, return True if it is and False otherwise"""
    train_table = get_train_table()
    return ((train_table["Patient"] == patient) & (train_table["Weeks"] == week)).any()


def create_nn_test(test_table, test_images_path=IMAGES_GCS_PATH + '/test'):
    """Create test table for NN"""
    # get standard form
    weekly_data = predict.create_submission_form(images_path=test_images_path)
    weekly_data["Patient"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[0])
    weekly_data["Weeks"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[1]).astype('float32')

    # predict weekly fvc
    exp_gen = predict.exponent_generator(test_images_path, for_test=True)
    exp_dict = {id: exp_func for id, exp_func in exp_gen}
    predict.predict_form(exp_dict, weekly_data)

    # merge
    data = test_table.drop(["FVC", "Weeks"], axis=1).merge(weekly_data, on="Patient")

    # remove unused features
    data = data.drop(["Patient_Week", "Confidence"], axis=1)

    return data


# TODO: delete this
if __name__ == "__main__""":
    import glob
    import os
    pd.set_option('display.max_columns', None)
    t = get_train_table()
    a = glob.glob("/Users/nurithofesh/Desktop/omer/osic-pulmonary-fibrosis-progression/images-norm/*/*")
    a = set([os.path.basename(path) for path in a])
    print(a)
    b = set(t["Patient"].values)
    print(b)
    print(b-a)




