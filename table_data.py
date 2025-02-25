import pandas as pd
import numpy as np
import visualize
import predict
import image_data
from statsmodels.formula.api import ols

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'


def get_train_table():
    """Return a dataframe with train records"""
    return pd.read_csv('train.csv').drop([1522])  # duplicated row


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


def get_patient_fvc_exp(table, id, remove_outliers=False):
    """Get a patient exponential coefficient.
    Each patient has his own fvc progression that can be described by Ie^-kt.
    Wheres I is the initial FCV measure, k is the exponential coefficient and t is time.
    Args:
        table--data table with records
        id--id of the patient
        remove_outliers--flag whether to remove outliers using cooks distance
    """

    hist = get_fvc_hist(table, id)

    # remove outliers if commanded
    if remove_outliers:
        hist = remove_outlier(remove_outlier(remove_outlier(hist)))  # remove 3 outliers

    # compute logs
    weeks = hist["Weeks"]
    log_fvc = np.log(hist["FVC"])

    # center weeks - set 0 for first measurement
    weeks = weeks - hist.iloc[0, 0]

    # regress to find logI -kt
    neg_k, logI = np.polyfit(weeks, log_fvc, deg=1)

    return -neg_k


def get_exp_fvc_dict(remove_outliers=False):
    """Return a dict with a mapping from id to their exponential fvc estimation coefficient.
    Args:
        remove_outliers: flag whether to remove ouliers using cooks distance
    """
    table = get_train_table()
    exp_dict = {}

    # get unique ids
    ids = table["Patient"].unique()

    # iterate threw every id
    for id in ids:
        k = get_patient_fvc_exp(table, id, remove_outliers=remove_outliers)
        exp_dict[id] = k

    return exp_dict


def get_exp_function_dict(remove_outliers=False):
    """Create ground truth exponent functions dict for train records.
    Args:
        remove_outliers: flag whether to remove outlier
    """

    exp_dict = get_exp_fvc_dict(remove_outliers=remove_outliers)

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


def get_patient_week_gt_fvc(patient_week):
    """Get the patient_week pair ground truth FVC from train table"""
    train_table = get_train_table()
    patient = patient_week.split('_')[0]
    week = int(patient_week.split('_')[1])

    if ((train_table["Patient"] == patient) & (train_table["Weeks"] == week)).any():
        return train_table.loc[(train_table["Patient"] == patient) & (train_table["Weeks"] == week)]["FVC"].values[0]
    else:
        return np.nan


def get_initials(table):
    """Create a table with initial FVC and week columns. Create normalized week column"""
    # get initial week and normalized week
    table["Initial_Week"] = table.groupby(["Patient"])["Weeks"].transform('min')
    table["Norm_Week"] = table["Weeks"] - table["Initial_Week"]

    # get initial fvc
    initial_fvc = table.loc[table["Norm_Week"] == 0][["Patient", "FVC"]]
    initial_fvc = initial_fvc.rename(columns={"FVC": "Initial_FVC"})  # rename for the merge

    # broadcast first percent value to all patient records
    initial_percent = table.loc[table["Norm_Week"] == 0][["Patient", "Percent"]]

    # merge
    table = table.merge(initial_fvc, on="Patient")  # merge initial fvc
    table = table.drop(["Percent"], axis=1)  # drop original percent column
    table = table.merge(initial_percent, on="Patient")  # merge new perencet

    return table


def get_predicted_percent(table):
    """Create a table with predicted percent column. initial percent * factor initial = fvc ->
    predicted percent = predicted fvc / factor
    """

    # create factors
    initials = table.loc[table["Norm_Week"] == 0][["Patient", "Percent", "FVC"]]
    initials["Factor"] = initials["FVC"] / initials["Percent"]

    # merge factors
    table = table.merge(initials[["Patient", "Factor"]], on="Patient")

    # create predicted fvc and delete residuals
    table["Predicted_Percent"] = table["FVC"] / table["Factor"]
    table = table.drop(["Factor"], axis=1)

    return table


def get_cooks_distance(observations):
    """Return the cooks distance of every point in observations"""
    x, y = observations.columns  # get predictor and response variable

    m = ols("{} ~ {}".format(x, y), observations).fit() # fit a statsmodels ols
    infl = m.get_influence() # check influens on every point
    cooks_dists = infl.summary_frame()["cooks_d"] # show cooks distance for every poin

    return cooks_dists


def remove_outlier(fvc_hist):
    """Remove the point with the largest cooks distance from the fvc observations.
    Do not remove the first nor the last point"""

    max_ind = get_cooks_distance(fvc_hist)[1:-1].argmax() + 1  # remove first and last
    fvc_hist = fvc_hist.reset_index(drop=True)
    return fvc_hist.drop(max_ind, axis=0)


# TODO: delete this
if __name__ == "__main__""":
    pd.set_option('display.max_columns', None)
    t = get_initials(get_train_table())
    print(t.head(5))
    t = get_predicted_percent(t)
    print(t.head(30))

