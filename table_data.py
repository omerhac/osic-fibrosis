import pandas as pd
import numpy as np
import visualize
import predict

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

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


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """

    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


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
    table[feature] = (table[feature] - mean) / (max - min)


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


def create_expanded_table(type='train', images_path=IMAGES_GCS_PATH + 'train'):
    """Create an expanded table dataset with a record for every patient+week couple"""
    # check type
    assert type == 'train' or type == 'validation' or type == 'test', "Type should be train / validation / test"

    # get raw table data
    if type == 'test':
        data = get_test_table()
    else:
        data = get_train_table()

    # get weekly patient form
    weekly_data = predict.create_submission_form(images_path=images_path)
    weekly_data["Patient"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[0])
    weekly_data["Week"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[1])


# TODO: delete this
if __name__ == "__main__""":
    pd.set_option('display.max_columns', None)
    t = get_train_table()
    print(preprocess_table_for_nn(t))