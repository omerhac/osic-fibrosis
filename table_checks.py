import metrics
import etl
import models
import pickle
import pandas as pd
import table_data

sub = pd.read_csv('submissions/sub_4.csv')
sub["GT_FVC"] = sub["Patient_Week"].apply(table_data.get_patient_week_gt_fvc)

sub_with_gt = sub.loc[sub["GT_FVC"].isna() == False]
y_true = sub_with_gt["GT_FVC"].values
y_pred = sub_with_gt["FVC"].values
theta = sub_with_gt["Confidence"].values
score = metrics.laplace_log_likelihood(y_true, y_pred, theta)
print(score)

pd.set_option('display.max_columns', None)
t = table_data.get_train_table()
t = table_data.get_initials(t)
print(t.loc[t["Weeks"] == t["Weeks"].max()])
print(t.loc[t["Patient"] == "ID00165637202237320314458"])
print(sub_with_gt["Patient_Week"].apply(lambda x: x.split('_')[1]).astype('float32').max())
print(sub_with_gt.loc[sub_with_gt["Patient_Week"].apply(lambda x: x.split('_')[1]).astype('float32') == 70])
print(len(sub_with_gt))

# check validation closely
t = pd.read_csv('theta_data/pp_train.csv')
model = models.get_qreg_model(len(t.columns) - 2)
model.load_weights('models_weights/qreg_model/model_v1.ckpt')
train_ids, val_ids = etl.get_train_val_split()
vals = t.loc[t["Patient"].isin(val_ids)]
y_true = vals["GT_FVC"]
x = vals.drop(["GT_FVC", "Patient"], axis=1).values
preds = model.predict(x)
y_pred = preds[:, 1]
print(y_pred.shape)
theta = (preds[:, 2] - preds[:, 0]) / 2
score = metrics.laplace_log_likelihood(y_true, y_pred, theta)
print(score)

# check prediction on test ids
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split('_')[0])
test_ids = sub["Patient"].unique()
t = t.drop_duplicates()
tests = t.loc[t["Patient"].isin(test_ids)]
print(len(tests))
y_true = tests["GT_FVC"]
x = tests.drop(["Patient", "GT_FVC"], axis=1).values
preds = model.predict(x)
y_pred = preds[:, 1]
theta = (preds[:, 2] - preds[:, 0]) / 2
score = metrics.laplace_log_likelihood(y_true, y_pred, theta)
tests["Preds"] = y_pred
tests["Theta"] = theta
print(score)
processor = pickle.load(open('models_weights/qreg_model/processor.pickle', 'rb'))
processor.inverse_transform(tests, "Weeks")
#tests[["Patient", "Weeks", "Preds"]].to_csv('tests.csv')
#sub_with_gt["test_preds_on_trainset"] = y_pred

# check preprocessed test
pp_test = pd.read_csv('theta_data/pp_test.csv')
preds = model.predict(pp_test.drop(["Patient"], axis=1).values)
pp_test["Pred"] = preds[:, 1]
pp_test["Theta"] = (preds[:, 2] - preds[:, 0]) / 2
processor.inverse_transform(pp_test, "Weeks")
processor.inverse_transform(pp_test, "Norm_Week")
processor.inverse_transform(pp_test, "Initial_Week")
processor.inverse_transform(pp_test, "Initial_FVC")
#sub_with_gt["test_preds_on_test_set"] = preds[:, 1]
#sub_with_gt["Patient"] = pp_test["Patient"]
#sub_with_gt["Weeks"] = pp_test["Weeks"]
#sub_with_gt.to_csv('check.csv')


