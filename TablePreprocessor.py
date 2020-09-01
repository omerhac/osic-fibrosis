import pandas as pd


class TablePreprocessor:
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

    @staticmethod
    def normalize_feature(table, feature):
        """Normalize a numeric feature in pandas DataFrame"""
        # get stats
        min, max = self._scale_dict[feature]

        # transform
        table[feature] = (table[feature] - min) / (max - min)

    def transform(self, table):
        """Preprocess table for NN digestion"""
        # one hot encode categorical
        sex = pd.get_dummies(table["Sex"], prefix='Sex')  # one hot encode sex variable
        smoking_status = pd.get_dummies(table["SmokingStatus"], prefix="SmokingStatus")  # one hot encode smokingstatus

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

