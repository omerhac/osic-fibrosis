import table_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_patient_exp(id, exp_function=None, ax=None):
    """Plot patient exponential regression
    Args:
        exp_functiion--function describing FVC progression
    """

    table = table_data.get_train_table()
    hist = table_data.get_fvc_hist(table, id)

    # get coeff
    k = table_data.get_patient_fvc_exp(table, id)

    # plot fvc
    if ax:
        ax.scatter(hist["Weeks"], hist["FVC"], color='red')

    else:
        plt.figure()
        sns.scatterplot(hist["Weeks"], hist["FVC"], color='red')

    # get exponent
    weeks = np.linspace(hist.iloc[0, 0], hist.iloc[-1, 0], 100)

    if exp_function:
        func = exp_function
    else:
        func = lambda week: hist.iloc[0, 1] * np.exp(-k * (week - weeks[0]))

    # evaluate values
    eval_values = func(weeks)  # center weeks before feeding to exp function

    # plot evaluated values
    if ax:
        ax.scatter(weeks, eval_values, color='blue')
        ax.set_ylim([hist["FVC"].min() - 500, hist["FVC"].max() + 500]) # set y axis lim

    else:
        sns.scatterplot(weeks, eval_values, color='blue')
        plt.ylim([hist["FVC"].min() - 500, hist["FVC"].max() + 500])  # set y axis lim
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


def plot_training_curves(hist):
    """Plot training and validation curves"""

    plt.figure()

    # plot loss from epoch 3 onwarde
    plt.plot(hist["loss"][3:], color='blue', label='loss')
    plt.plot(hist["val_loss"][3:], color='orange', label='val_loss')
    plt.show()


def plot_training_metrics(hist, metrics=[]):
    """Plot learning metrics.
        Args:
        metrics--list of metrics to monitor
    """

    plt.figure()
    for metric in metrics:
        plt.plot(hist[metric], label=metric)
        plt.plot(hist['val_' + metric], label='val_'+metric)
    plt.show()

