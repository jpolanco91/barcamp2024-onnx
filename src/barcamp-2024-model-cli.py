import requests
import sys
import argparse
import json

pytorch_inference_url = "http://localhost:8080/predictions/barcamp2024-model-pytorch"
tensorflow_inference_url = "http://localhost:8501/v1/models/barcamp2024-model-tensorflow:predict"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapoint", help = "Dato del cual se quiere una prediccion (json)")
parser.add_argument("-p", "--pytorch", help = "Seleccionar TorchServe como servidor de inferencia")
parser.add_argument("-t", "--tensorflow", help = "Seleccionar Tensorflow Serving como servidor de inferencia")
parser.add_argument("-o", "--onnx", help = "Seleccionar ONNX como servidor de inferencia")

args = parser.parse_args()

if args.datapoint:
	print(f"Dato para prediccion: %s " % args.datapoint)
	datapoint_dict = json.loads(args.datapoint)
	print(datapoint_dict)
	if args.pytorch:
		print("Seleccionando TorchServe como servidor de inferencia")
		response = requests.post(pytorch_inference_url, json = datapoint_dict)
		print(response)
	elif args.tensorflow:
		print("Seleccionando Tensorflow Serving como servidor de inferencia")
	elif args.onnx:
		print("Seleccionando ONNX runtime como servidor de inferencia")

