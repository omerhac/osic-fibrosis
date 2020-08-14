import predict
import numpy as np
import table_data

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images'

# image size
IMAGE_SIZE = (224, 224)


def laplace_log_likelihood(y_true, y_pred, theta):
    """Compute the Laplace Lop Likelihood score for the predictions y_pred.
    Args:
        y_true-ground truth values
        y_pred-predicted values
        theta-confidence measurements
    """

    theta_clipped = np.clip(theta, 70, None)
    delta = np.clip(np.abs(y_true - y_pred), 0, 1000)
    metric = - (np.sqrt(2) * delta) / theta_clipped - np.log(np.sqrt(2) * theta_clipped)
    return metric.sum() / len(y_true)


def get_lll_value_exp_function(id, exp_function, theta=100):
    """Return the laplace log likelihood score for a given patient and his exponent function."""
    hist = table_data.get_fvc_hist(table_data.get_train_table(), id)  # get ground truth

    # initiate predictions
    y_pred = []

    # get predictions
    for week in hist["Weeks"]:
        week = float(week)
        y_pred = y_pred.append(exp_function(week))  # prediction

    y_pred = np.stack(y_pred)
    print(y_pred)
    y_true = hist["FVC"]
    print(y_true)
    metric = laplace_log_likelihood(y_true, y_pred, theta)
    return metric


g = predict.exponent_generator(IMAGES_GCS_PATH + '/train')
id, func = next(g)
print(get_lll_value_exp_function(func, id))
