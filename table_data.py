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


def plot_patient_fvc(table, id, order=2):
    """Plot patients FVC history along with a polynomial regression"""
    plt.figure(figsize=(8,8))
    hist = get_fvc_hist(table, id)  # get fvc hist
    coeffs = get_patient_fvc_poly(table, id, order=order)

    # plot
    sns.regplot(hist["Weeks"], hist["FVC"], order=order)

    # get title
    title = "y = "
    for deg, coeff in enumerate(coeffs):
        title = title +"{}x^{} + ".format(coeff, deg)

    title = title[:-2] # remove end
    plt.title(title)
    plt.show()


def plot_patient_percent(table, id, order=2):
    """Plot patients Percent history along with a polynomial regression"""
    plt.figure(figsize=(8,8))
    hist = get_percent_hist(table, id)  # get fvc hist
    coeffs = get_patient_percent_poly(table, id, order=order)

    # plot
    sns.regplot(hist["Weeks"], hist["Percent"], order=order)

    # get title
    title = "y = "
    for deg, coeff in enumerate(coeffs):
        title = title +"{}x^{} + ".format(coeff, deg)

    title = title[:-2]  # remove end
    plt.title(title)
    plt.show()


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


def get_dataset_from_dict(dict):
    """Create a dataset from a dict with mapping from patient id and their polynomial fvc coeffs."""
    id_list, poly_list = [], []

    for id in dict.keys():
        id_list.append(id)
        poly_list.append(dict[id])

    return np.stack(id_list), np.stack(poly_list)


# TODO: delete this
if __name__ == "__main__""":
    ds = get_train_table()
    print(get_patient_fvc_poly(ds, "ID00007637202177411956430"))