import numpy as np
import pandas as pd

class RegPredictionDataPrep(object):

    def __init__(self, dataset, skewed_features, numerical_features, dataset_features):
        self.dataset = dataset.copy(deep=True)
        self.skewed_features = skewed_features
        self.numerical_features = numerical_features
        self.dataset_features = dataset_features
        self.features_transformed = {}

        for feature in self.dataset_features:
            self.features_transformed[feature] = 0.0


    def parse_features(self, features_dict):
        for key, value in features_dict.items():
            if isinstance(value, str):
                dict_key = key + '_' + value
                self.features_transformed[dict_key] = 1.0
            else:
                if key in self.skewed_features:
                    feature_max_val, feature_min_val = self.get_max_min_val(key)
                    log_transformed_feature = np.log(value)
                    scaled_feature = self.scale_feature_value(log_transformed_feature, feature_max_val, feature_min_val)
                    self.features_transformed[key] = scaled_feature
                else:
                    feature_max_val, feature_min_val = self.get_max_min_val(key)
                    scaled_feature = self.scale_feature_value(value, feature_max_val, feature_min_val)
                    self.features_transformed[key] = scaled_feature

        
        return pd.DataFrame(self.features_transformed, index=[0])


    def scale_feature_value(self, feature_value, max_val, min_val):
        feature_std = (feature_value - min_val) / (max_val - min_val)
        return feature_std

    def get_max_min_val(self, feature):
        feature_data = self.dataset[feature]
        max_value = feature_data.max()
        min_value = feature_data.min()
        return max_value, min_value