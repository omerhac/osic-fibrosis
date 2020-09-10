import pandas as pd


class TablePreprocessor:
    """A class to preprocess data before feeding to the quantiles regression model.
    Can fit() on the training data and then transform the train / test data.
    Min-Max scale numeric columns and one hot encodes categorical data.
    """

    def __init__(self):
        self._scale_dict = {}

    def fit(self, table):
        """Fit the preprocessor to the data"""
        self._scale_dict["Weeks"] = (table["Weeks"].min(), table["Weeks"].max())
        self._scale_dict["FVC"] = (table["FVC"].min(), table["FVC"].max())
        self._scale_dict["Percent"] = (table["Percent"].min(), table["Percent"].max())
        self._scale_dict["Age"] = (table["Age"].min(), table["Age"].max())
        self._scale_dict["Initial_Week"] = (table["Initial_Week"].min(), table["Initial_Week"].max())
        self._scale_dict["Initial_FVC"] = (table["Initial_FVC"].min(), table["Initial_FVC"].max())
        self._scale_dict["Norm_Week"] = (table["Norm_Week"].min(), table["Norm_Week"].max())

    def normalize_feature(self, table, feature):
        """Min-Max scale a numeric feature in pandas DataFrame"""
        # get stats
        min, max = self._scale_dict[feature]

        # transform
        table[feature] = (table[feature] - min) / (max - min)

    def transform(self, table):
        """Preprocess table for NN digestion"""
        # one hot encode sex variable
        sex = table["Sex"].astype(
            pd.CategoricalDtype(categories=["Male", "Female"]))
        sex = pd.get_dummies(sex, prefix='Sex')

        # one hot encode smokingstatus
        smoking_status = table["SmokingStatus"].astype(
            pd.CategoricalDtype(categories=["Currently smokes", "Ex-smoker", "Never smoked"]))
        smoking_status = pd.get_dummies(smoking_status, prefix="SmokingStatus")

        # concat
        ohe_table = pd.concat([table, sex, smoking_status], axis=1).drop(["Sex", "SmokingStatus"], axis=1)

        # normalize numeric columns
        self.normalize_feature(ohe_table, "Weeks")
        self.normalize_feature(ohe_table, "FVC")
        self.normalize_feature(ohe_table, "Percent")
        self.normalize_feature(ohe_table, "Age")
        self.normalize_feature(ohe_table, "Initial_Week")
        self.normalize_feature(ohe_table, "Initial_FVC")
        self.normalize_feature(ohe_table, "Norm_Week")

        return ohe_table

    def inverse_transform(self, table, feature):
        """Inverse transform the feature. Assumes the feature is already transformed"""
        min, max = self._scale_dict[feature]  # get values
        table[feature] = table[feature] * (max - min) + min