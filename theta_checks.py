import pandas as pd
import pickle as pickle
import numpy as np
import etl
import matplotlib.pyplot as plt
import metrics
import predict

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'

if __name__ == '__main__':
    pp_train = pd.read_csv('theta_data/pp_train.csv')
    processor = pickle.load(open('models_weights/qreg_model/processor.pickle', 'rb'))

    # inverse transform FVC
    processor.inverse_transform(pp_train, 'FVC')
    pp_train['Theta'] = np.abs(pp_train['FVC'] - pp_train["GT_FVC"])

    print(pp_train['Theta'].mean())
    print(pp_train['Theta'].max())
    print(pp_train['Theta'].argmax())

    avg_thetas = []
    stds = []

    val_gen = predict.exponent_generator(
        IMAGES_GCS_PATH + '/validation',
        model_path='models_weights/cnn_model/model_v4.ckpt',
        image_size=[512,512],
        enlarged_model=True,
        yield_std=True
    )

    for id, exp_func, std in val_gen:
        stds.append(std)

        score, avg_theta = metrics.get_lll_value_exp_function(id, exp_func, return_theta=True)
        print(score)
        avg_thetas.append(avg_theta)

    # plot
    plt.figure()
    plt.scatter(stds, avg_thetas)
    plt.show()

