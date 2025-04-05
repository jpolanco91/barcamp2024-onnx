import onnxruntime as ort
import argparse
import tensorflow as tf
import torch
import pandas as pd
import json
from preprocessing_pipelines import RegPredictionDataPrep

tf_model_path = "barcamp2024-model-tensorflow.onnx"
torch_model_path = "barcamp2024-model-pytorch.onnx"
providers = ['CPUExecutionProvider']

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapoint", help = "Dato del cual se quiere una prediccion (json)")
parser.add_argument("-p", "--pytorch", help = "Seleccionar PyTorch como modelo")
parser.add_argument("-t", "--tensorflow", help = "Seleccionar Tensorflow como modelo")

args = parser.parse_args()

if args.datapoint:
    datapoint_dict = json.loads(args.datapoint)
    dataset_path = '../Dataset/weight_change_dataset.csv'
    dataset_data = pd.read_csv(dataset_path)
    dataset_data = dataset_data.drop(['participant_id'], axis=1)

    pred_pipeline_data = dataset_data.drop('weight_change', axis = 1)
    datapoint_features_altamente_sesgados = ['daily_calories_consumed']
    datapoint_features_numericos = ['age', 'current_weight', 'bmr', 'daily_calories_consumed', 'daily_caloric_surplus_deficit', 'duration', 'stress_level', 'final_weight']
    one_hot_encoded_features = pd.get_dummies(pred_pipeline_data, dtype='int')
    features_names =  list(one_hot_encoded_features.keys())
    prediction_pipeline = RegPredictionDataPrep(pred_pipeline_data, datapoint_features_altamente_sesgados, datapoint_features_numericos, features_names)
    sample_pre_processed_datapoint = prediction_pipeline.parse_features(datapoint_dict)

    if args.tensorflow:
        print("Selected Tensorflow ONNX model.")
        converted_dp_tf_tensor = tf.convert_to_tensor(sample_pre_processed_datapoint)
        tf_onnx_session = ort.InferenceSession(tf_model_path, providers=providers)
        numpy_tensorflow_tensor = converted_dp_tf_tensor.numpy()
        tf_predictions = tf_onnx_session.run(None, {"args_0": numpy_tensorflow_tensor})
        print("Prediction: ", tf_predictions[0][0])
    elif args.pytorch:
        print("Selected PyTorch ONNX model.")
        converted_dp_torch_tensor = torch.tensor(sample_pre_processed_datapoint.values, dtype=torch.float32)
        torch_onnx_session = ort.InferenceSession(torch_model_path, providers=providers)
        numpy_torch_tensor = converted_dp_torch_tensor.numpy()
        torch_predictions = torch_onnx_session.run(None, {"modelInput": numpy_torch_tensor})
        print("Prediction: ", torch_predictions[0][0])
    else:
        print("No model selected")


		


