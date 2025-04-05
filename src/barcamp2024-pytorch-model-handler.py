"""
Manejador del modelo para weight change model hecho en PyTorch para Barcamp 2024.
"""

import torch
import pandas as pd
import os
import importlib
import preprocessing_pipelines as pp
from barcamp2024_pytorch_model import DeepNeuralNetwork
from ts.torch_handler.base_handler import BaseHandler

class WeightChangeModelHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        self._context = context
        self.manifest = self._context.manifest
        properties = self._context.system_properties
        model_dir = properties.get("model_dir")

        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = DeepNeuralNetwork()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.initialized = True

    def preprocess(self, data):

        dataset_path = '/Users/juanjpolanco/Documents/Barcamp2024/barcamp2024-onnx/Dataset/weight_change_dataset.csv'
        dataset_data = pd.read_csv(dataset_path)
        dataset_data = dataset_data.drop(['participant_id'], axis=1)

        pred_pipeline_data = dataset_data.drop('weight_change', axis = 1)
        datapoint_features_altamente_sesgados = ['daily_calories_consumed']
        datapoint_features_numericos = ['age', 'current_weight', 'bmr', 'daily_calories_consumed', 'daily_caloric_surplus_deficit', 'duration', 'stress_level']
        one_hot_encoded_features = pd.get_dummies(pred_pipeline_data, dtype='int')
        features_names =  list(one_hot_encoded_features.keys())
        prediction_pipeline = pp.RegPredictionDataPrep(pred_pipeline_data, datapoint_features_altamente_sesgados, datapoint_features_numericos, features_names)
        sample_pre_processed_datapoint = prediction_pipeline.parse_features(data[0]['body'])

        preprocessed_data = torch.tensor(sample_pre_processed_datapoint.values, dtype=torch.float32)

        return preprocessed_data

    def inference(self, model_input):
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        return [inference_output.float().item()]

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
