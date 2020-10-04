import pandas as pd
import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import metrics
import predict
import seaborn as sns

# images path
IMAGES_GCS_PATH = 'gs://osic_fibrosis/images-norm/images-norm'

if __name__ == '__main__':
    pp_train = pd.read_csv('theta_data/pp_train.csv')
    processor = pickle.load(open('models_weights/qreg_model/processor.pickle', 'rb'))

    # inverse transform FVC
    processor.inverse_transform(pp_train, 'FVC')
    pp_train['Theta'] = np.abs(pp_train['FVC'] - pp_train["GT_FVC"])
    pp_train['']
    print("Average optimal theta: {}".format(pp_train['Theta'].mean()))

    avg_thetas = []
    stds = []
    scores = []
    val_gen = predict.exponent_generator(
        IMAGES_GCS_PATH + '/train',
        model_path='models_weights/cnn_model/model_v4.ckpt',
        image_size=[512,512],
        enlarged_model=True,
        yield_std=True
    )

    i = 0
    for id, exp_func, std in val_gen:
        stds.append(std)
        i+=1
        score, avg_theta = metrics.get_lll_value_exp_function(id, exp_func, return_theta=True)
        scores.append(score)
        print("Patient {}:".format(i))
        print("CNN score: {}".format(score))
        qreg_score = metrics.metric_check(get_patient=id)
        print("Qreg score: {}".format(qreg_score))
        avg_thetas.append(avg_theta)
        if i == 40:
            break

    print("AVRG CNN SCRORE: {}".format(np.mean(scores)))
    # plot
    plt.figure()
    sns.regplot(stds, avg_thetas)
    plt.show()

