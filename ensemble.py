import pandas as pd
import models
import predict
import pickle
import metrics
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.model_selection import train_test_split

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'


def create_ensemble_table(save_path='ensemble_table.csv', qreg_model_path='models_weights/qreg_model/model_v4.ckpt',
                          cnn_model_path='models_weights/cnn_model/model_v6.ckpt', image_size=[256, 256],
                          processor_path='models_weights/qreg_model/processor.pickle', pp_train='theta_data/pp_train.csv'):
    """
    Create a train table for ensemble training. Predict for both qreg and CNN models.
    Parameters:
        :param save_path: table save path
        :param qreg_model_path: quantile regression model path
        :param cnn_model_path: CNN model path
        :param image_size: CNN model image size
        :param processor_path: table preprocessor pickle path
        :param pp_train: preprocessed train table path
    """

    # train set
    train_table = pd.read_csv(pp_train)
    nn_x = train_table.drop(['GT_FVC', 'Patient'], axis=1).values
    ensemble_table = train_table[['Patient', 'Weeks', 'GT_FVC']]
    processor = pickle.load(open(processor_path, 'rb'))
    processor.inverse_transform(ensemble_table, 'Weeks')

    # models
    qreg_model = models.get_qreg_model(nn_x.shape[1])
    qreg_model.load_weights(qreg_model_path)
    cnn_gen = predict.exponent_generator(IMAGES_GCS_PATH + '/train',
                                         model_path=cnn_model_path,
                                         image_size=image_size)

    # preds
    qreg_preds = qreg_model.predict(nn_x)[:, 1]  # qreg predict
    ensemble_table['Qreg_FVC'] = qreg_preds
    cnn_exp_dict = {id: exp_func for id, exp_func in cnn_gen}
    predict.predict_form(cnn_exp_dict, ensemble_table, submission=False)  # cnn predict

    # save
    ensemble_table.to_csv(save_path, index=False)


def fix_ensemble_table():
    fixed_ensemble = pd.read_csv('ensemble_table.csv')
    fixed_ensemble = fixed_ensemble.drop_duplicates(subset=['Weeks', 'Patient', 'GT_FVC', 'FVC'])
    fixed_ensemble = fixed_ensemble.loc[fixed_ensemble["FVC"].notnull()]
    fixed_ensemble.to_csv('fixed_ensemble_table.csv', index=False)


def ensemble_metric_check():
    ensemble_table = pd.read_csv('fixed_ensemble_table.csv')
    cnn_metric = metrics.laplace_log_likelihood(ensemble_table["GT_FVC"].values, ensemble_table["FVC"].values, 200)
    qreg_metric = metrics.laplace_log_likelihood(ensemble_table["GT_FVC"].values, ensemble_table["Qreg_FVC"].values, 200)
    print("Qreg score {}, CNN score {}".format(qreg_metric, cnn_metric))


if __name__ == '__main__':
    # dataset
    ensemble_table = pd.read_csv('fixed_ensemble_table.csv')
    qreg_values = ensemble_table["Qreg_FVC"].values.reshape(-1, 1)
    cnn_values = ensemble_table["FVC"].values.reshape(-1, 1)
    y = ensemble_table["GT_FVC"].values
    X = np.concatenate([qreg_values, cnn_values], axis=1)

    # fit regressor
    linear_regressor = LinearRegression(fit_intercept=True, normalize=False)
    linear_regressor.fit(X, y)

    print(linear_regressor.coef_, linear_regressor.intercept_)
    regressor_preds = linear_regressor.predict(X)
    print('Regressor score {}'.format(metrics.laplace_log_likelihood(y, regressor_preds, 200)))

    # check simple blend
    qreg_values = qreg_values.reshape((-1, ))
    cnn_values = cnn_values.reshape((-1, ))
    pred1 = qreg_values * 0.4 + cnn_values * 0.6
    pred2 = qreg_values * 0.25 + cnn_values * 0.75
    pred3 = qreg_values * 0.6 + cnn_values * 0.4
    score1 = metrics.laplace_log_likelihood(y, pred1, 200)
    score2 = metrics.laplace_log_likelihood(y, pred2, 200)
    score3 = metrics.laplace_log_likelihood(y, pred3, 200)
    print("40qreg + 60cnn {}, 25qreg + 75cnn {}, 60qreg + 40cnn {}".format(score1, score2, score3))

    # simple nn
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input([2]),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2)
    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=200, batch_size=10)

    print("NN score {}".format(metrics.laplace_log_likelihood(y, model.predict(X)[0], 200)))
