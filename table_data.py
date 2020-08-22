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
    # normalize_feature(ohe_table, "Week")
    # normalize_feature(ohe_table, "FVC")
    normalize_feature(ohe_table, "Percent")
    normalize_feature(ohe_table, "Age")

    return ohe_table


def create_expanded_table(images_path=IMAGES_GCS_PATH + '/train', for_test=False):
    """Create an expanded table dataset with a record for every patient+week couple. Also predicts FVC for each week."""
    # get raw table data
    if for_test:
        data = get_test_table()
    else:
        data = get_train_table()

    # get weekly patient form
    weekly_data = predict.create_submission_form(images_path=images_path)
    weekly_data["Patient"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[0])
    weekly_data["Week"] = weekly_data["Patient_Week"].apply(lambda x: x.split('_')[1]).astype('int8')

    # predict weekly fvc
    exp_gen = predict.exponent_generator(images_path, for_test=for_test)
    exp_dict = {id: exp_func for id, exp_func in exp_gen}
    predict.predict_form(exp_dict, weekly_data)

    # merge
    data = data.drop(["Weeks", "FVC"], axis=1).merge(weekly_data, on="Patient")

    # remove unused features
    data = data.drop(["Patient_Week", "Confidence"], axis=1)

    return data


def create_nn_dataset(image_path=IMAGES_GCS_PATH + '/train', for_test=False, save_path=None):
    """Create dataset for NN training"""

    # get data
    data = create_expanded_table(images_path=image_path, for_test=for_test)

    # preprocess data
    data = preprocess_table_for_nn(data)

    if save_path:
        data.to_csv(save_path)
    else:
        return data


def get_theta_labels(table, save_path=None):
    """Return ground truth theta labels.
    Compute the optimal theta by |GT_FVC - pred_FVC| if GT FVC is available. Else default to 150
    """

    theta = pd.DataFrame([])  # create new dataframe
    train_table = get_train_table()

    # iterate threw all of the train records
    for index, row in table.iterrows():
        patient = row["Patient"]
        week = row["Week"]

        # get prediction FVC
        pred_fvc = float(table.loc[((table["Patient"] == patient) & (table["Week"] == week)), "FVC"])
        # check whether a record exists in ground truth
        if patient_week_exists(patient, week):
            # get GT FVC
            gt_fvc = train_table.loc[(train_table["Patient"] == patient) & (train_table["Weeks"] == week), "FVC"].iloc[0]
            gt_fvc = float(gt_fvc)
            theta = theta.append([np.abs(gt_fvc - pred_fvc)])

        # default to 150
        else:
            theta = theta.append([150])

    if save_path:
        theta.to_csv(save_path)
    else:
        return theta


def patient_week_exists(patient, week):
    """Check of a patient week couple exists in train records, return True if it is and False otherwise"""
    train_table = get_train_table()
    return ((train_table["Patient"] == patient) & (train_table["Weeks"] == week)).any()


# TODO: delete this
if __name__ == "__main__""":
    pd.set_option('display.max_columns', None)
    t = create_nn_dataset(IMAGES_GCS_PATH + '/test', for_test=True)
    get_theta_labels(t, save_path='theta_data/theta.csv')

