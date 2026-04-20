import pandas as pd
from train_test_split import train_test_split
from sklearn.datasets import load_digits
import StandardScaler
import MLP
import json
import sys
import os

PROJECT_NAME = "Multilayer Perceptron"


def print_header(config_loaded: bool, model_ready: bool):
	line = "=" * 56
	print("\n" + line)
	print(f" {PROJECT_NAME} CLI")
	print(line)
	print(f" Config: {'loaded' if config_loaded else 'not loaded'}")
	print(f" Model:  {'ready' if model_ready else 'not trained'}")
	print(line)


def print_menu():
	print(
		"[1] Load network config\n"
		"[2] Split dataset\n"
		"[3] Train model\n"
		"[4] Predict\n"
		"[5] Compare predictions\n"
		"[q] Quit"
	)

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


def split_dataset(config):
	dataset_path = config["model"]["dataset_path"] if "dataset_path" in config["model"] else input("Enter the path to the dataset: ").strip()
	target = config["model"]["target"] if "target" in config["model"] else input("Enter the target label or index: ").strip()
	validation_split = config["model"]["validation_split"] if "validation_split" in config["model"] else input("Enter the validation split: ").strip()

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

		patience = config["model"]["early_stopping_patience"] if "early_stopping_patience" in config["model"] else None

		model = MLP.MultiLayerPerceptron(train, valid, target, patience)
		for layer in config["network"]["layers"]:
			model.add_layer(
				n_neurons=layer["n_neurons"],
				activation_function=layer["activation_function"],
				weights_initializer=layer["weights_initializer"])

		model.train(
			epochs=config["model"]["epochs"],
			learning_rate=config["model"]["learning_rate"],
			batch_size=config["model"]["batch_size"],
			loss=config["model"]["loss"]
		)

		model.animation(
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
	
def compare(config):
	try:
		if config is None:
			raise ValueError("Please load the config file first.")

		csv_a_path = input("Enter the path to the first CSV file: ").strip()
		csv_b_path = input("Enter the path to the second CSV file: ").strip()

		target = config["model"]["target"]
		is_numeric_target = isinstance(target, int) or (isinstance(target, str) and target.strip().isdigit())

		header = None if is_numeric_target else 0
		df_a = pd.read_csv(csv_a_path, index_col=0, header=header)
		df_b = pd.read_csv(csv_b_path, index_col=0, header=header)

		if len(df_a) != len(df_b):
			raise ValueError(f"CSV files must have the same number of rows ({len(df_a)} vs {len(df_b)}).")

		if is_numeric_target:
			target_idx = int(target)
			if target_idx < 0 or target_idx >= df_a.shape[1] or target_idx >= df_b.shape[1]:
				raise ValueError(f"Target index {target_idx} is out of bounds for one of the files.")
			s1 = df_a.iloc[:, target_idx]
			s2 = df_b.iloc[:, target_idx]
			target_label = str(target_idx)
		else:
			target_name = str(target)
			if target_name not in df_a.columns:
				raise ValueError(f"Target column '{target_name}' not found in first CSV.")
			if target_name not in df_b.columns:
				raise ValueError(f"Target column '{target_name}' not found in second CSV.")
			s1 = df_a[target_name]
			s2 = df_b[target_name]
			target_label = target_name

		comp = pd.concat([s1.rename("a"), s2.rename("b")], axis=1, join="outer")
		idx_only_a = s1.index.difference(s2.index)
		idx_only_b = s2.index.difference(s1.index)

		comp.loc[:, "a"] = comp.loc[:, "a"].astype(str)
		comp.loc[:, "b"] = comp.loc[:, "b"].astype(str)
		matches = comp.loc[:, "a"] == comp.loc[:, "b"]

		n_matches = int(matches.sum())
		n_mismatches = int(len(comp) - n_matches)
		agreement = (n_matches / len(comp)) if len(comp) > 0 else 0.0

		print(f"Target used: {target_label}")
		print(f"Compared entries: {len(comp)}")
		print(f"Matches: {n_matches}")
		print(f"Mismatches: {n_mismatches}")
		print(f"Agreement: {agreement:.4f} ({agreement * 100:.2f}%)")
		if len(idx_only_a) > 0 or len(idx_only_b) > 0:
			print(f"Index mismatch - only in first CSV: {len(idx_only_a)}, only in second CSV: {len(idx_only_b)}")

		if n_mismatches > 0:
			mismatch_idx = matches[matches == False].index.tolist()[:10]
			print(f"First mismatch indices (up to 10): {mismatch_idx}")

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
			print_header(config_loaded=(config is not None), model_ready=(model is not None and scaler is not None))
			print_menu()
			choice = input("Select an action > ").strip().lower()
			if choice == '1':
				config = load_config()
			elif choice == '2':
				train, valid = split_dataset(config)
			elif choice == '3':
				model, scaler = train_model(train, valid, config)
			elif choice == '4':
				predict(model, scaler)
			elif choice == '5':
				compare(config)
			elif choice in ['exit', 'quit', 'q']:
				return

	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		raise e

if __name__ == "__main__":
	main()