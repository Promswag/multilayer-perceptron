import pandas as pd
from train_test_split import train_test_split
from sklearn.datasets import load_digits
import StandardScaler
import MLP
import json
import sys
import os

def load_config():
	config_path = input("Enter the path to the config file: ").strip()
	if not os.path.exists(config_path):
		print(f"File {config_path} does not exist.")
		return None
	try:
		with open(config_path, 'r') as file:
			config = json.load(file)
		return config
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		return None

def load_dataset(dataset_path):
	try:
		sample = pd.read_csv(dataset_path, index_col=0, nrows=5)
		if all(isinstance(v, str) for v in sample.columns):
			header = 0
		else:
			header = None
		df = pd.read_csv(dataset_path, index_col=0, header=header)
		return df
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		return None


def split_dataset():
	dataset_path = input("Enter the path to the dataset: ").strip()
	target = input("Enter the target label or index: ").strip()
	validation_split = input("Enter the validation split: ").strip()

	try:
		validation_split = float(validation_split)
		if validation_split < 0 or validation_split > 1:
			raise ValueError("Validation split must be between 0 and 1")
		
		df = load_dataset(dataset_path)

		try:
			target = int(target)
		except:
			pass

		train, valid = train_test_split(df, target, float(validation_split))
		train.to_csv(dataset_path[:-4] + '_train.csv')
		print(f"Training data with shape {train.shape} saved as {dataset_path[:-4]}_train.csv ")
		valid.to_csv(dataset_path[:-4] + '_valid.csv')
		print(f"Validation data with shape {valid.shape} saved as {dataset_path[:-4]}_valid.csv")
		return train, valid
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		return None, None

def train_model(train, valid, config):
	try:
		if train is None or valid is None or config is None:
			raise ValueError("Please split the dataset and load the config file first.")
		
		target = config["model"]["target"]
		if isinstance(target, int):
			target = train.columns[target]
		numeric_features = train.select_dtypes(include='number').columns.difference([target])

		scaler = StandardScaler.StandardScaler()
		train.loc[:, numeric_features] = scaler.fit_transform(train.loc[:, numeric_features])
		valid.loc[:, numeric_features] = scaler.transform(valid.loc[:, numeric_features])

		model = MLP.MultiLayerPerceptron(train, valid, target)
		for layer in config["network"]["layers"]:
			model.add_layer(
				n_neurons=layer["n_neurons"],
				activation_function=layer["activation_function"],
				weights_initializer=layer["weights_initializer"])

		model.init_layers()

		model.train(
			epochs=config["model"]["epochs"],
			learning_rate=config["model"]["learning_rate"],
			batch_size=config["model"]["batch_size"],
			loss=config["model"]["loss"]
		)
		return model, scaler
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		raise e
		return None, None

def predict(model, scaler):
	try:
		if model is None or scaler is None:
			raise ValueError("Please train the model first.")
		dataset_path = input("Enter the path to the dataset: ").strip()
		df = load_dataset(dataset_path)
		features = model.features
		df[features] = scaler.transform(df[features])
		model.predict(df[features])
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		return

def main():
	train, valid = None, None
	model, scaler = None, None
	config = None
	try:
		if len(sys.argv) == 2:
			with open(sys.argv[1], 'r') as file:
				config = json.load(file)

		while(True):
			print("[1] Split dataset - [2] Train model - [3] Predict - [4] Load network config")
			choice = input("Enter your choice: ").strip()
			if choice == '1':
				train, valid = split_dataset()
			elif choice == '2':
				model, scaler = train_model(train, valid, config)
			elif choice == '3':
				predict(model, scaler)
			elif choice == '4':
				config = load_config()
			elif choice == 'exit' or choice == 'quit' or choice == 'q':
				return

	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		raise e

if __name__ == "__main__":
	main()