import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_train_table():
    """Returns a dataframe with train records"""
    return pd.read_csv('train.csv')


def get_test_table():
    """Returns a dataframe with test records"""
    return pd.read_csv('test.csv')


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
    weeks = weeks - hist.iloc[0,0]

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


# TODO: delete this
if __name__ == "__main__""":
    ds = get_train_table()
    plot_patient_exp(ds, "ID00030637202181211009029")
    e = get_exp_fvc_dict()
    for k in e.keys():
        print(e[k])