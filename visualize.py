import table_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_patient_exp(table, id, exp_function=None):
    """Plot patient exponential regression
    Args:
        exp_functiion--function describing FVC progression
    """
    hist = table_data.get_fvc_hist(table, id)

    # get coeff
    k = table_data.get_patient_fvc_exp(table, id)

    # plot fvc
    plt.figure()
    sns.scatterplot(hist["Weeks"], hist["FVC"], color='red')

    # get exponent
    weeks = np.linspace(hist.iloc[0, 0], hist.iloc[-1, 0], 100)
    func = lambda week: hist.iloc[0, 1] * np.exp(-k * week)

    # evaluate values
    eval_values = func(weeks - weeks[0])  # center weeks before feeding to exp function

    # plot evaluated values
    sns.scatterplot(weeks, eval_values, color='blue')
    plt.show()


def plot_patient_percent(table, id, order=2):
    """Plot patients Percent history along with a polynomial regression"""
    plt.figure(figsize=(8, 8))
    hist = table_data.get_percent_hist(table, id)  # get fvc hist
    coeffs = table_data.get_patient_percent_poly(table, id, order=order)

    # plot
    sns.regplot(hist["Weeks"], hist["Percent"], order=order)

    # get title
    title = "y = "
    for deg, coeff in enumerate(coeffs[::-1]):
        title = title +"{}x^{} + ".format(coeff, deg)

    title = title[:-2]  # remove end
    plt.title(title)
    plt.show()


def plot_patient_fvc(table, id, order=2):
    """Plot patients FVC history along with a polynomial regression"""
    plt.figure(figsize=(8,8))
    hist = table_data.get_fvc_hist(table, id)  # get fvc hist
    coeffs = table_data.get_patient_fvc_poly(table, id, order=order)

    # plot
    sns.regplot(hist["Weeks"], hist["FVC"], order=order)

    # get title
    title = "y = "
    for deg, coeff in enumerate(coeffs[::-1]):
        title = title +"{}x^{} + ".format(coeff, deg)

    title = title[:-2]  # remove end
    plt.title(title)
    plt.show()