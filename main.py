import pandas as pd
import numpy as np
from train_test_split import train_test_split
import StandardScaler
import MLP
import json
import sys
import os
import pickle

PROJECT_NAME = "Multilayer Perceptron"


def build_model_path_from_config(config: dict) -> str:
	config_name = "config"
	if config is not None:
		config_name = config.get("_config_name", "config")
	base_name = os.path.splitext(os.path.basename(str(config_name)))[0]
	return os.path.join("models", f"mlp_{base_name}.pkl")


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
		"[2] Load model\n"
		"[3] Split dataset\n"
		"[4] Train model\n"
		"[5] Predict\n"
		"[6] Evaluate predictions\n"
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
		config["_config_path"] = config_path
		config["_config_name"] = os.path.basename(config_path)
		if "seed" in config["model"]:
			np.random.seed(config["model"]["seed"])
			print(f"Random seed set to {config['model']['seed']}")
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


def save_model(model, scaler, config=None, path: str = None):
	try:
		if not path:
			raise ValueError("Model path is required.")
		directory = os.path.dirname(path)
		if directory:
			os.makedirs(directory, exist_ok=True)
		stored_config = None
		if config is not None:
			stored_config = {k: v for k, v in config.items() if not str(k).startswith("_")}
		with open(path, 'wb') as file:
			pickle.dump({
				'model': model,
				'scaler': scaler,
				'config': stored_config
			}, file)
		print(f"Model saved to {path}")
	except Exception as e:
		print(f"{type(e).__name__}: {e}")


def load_model(path: str):
	try:
		if not path:
			raise ValueError("Model path is required.")
		if not os.path.exists(path):
			raise FileNotFoundError(f"Model file '{path}' does not exist.")
		with open(path, 'rb') as file:
			model_data = pickle.load(file)
		model = model_data.get('model')
		scaler = model_data.get('scaler')
		stored_config = model_data.get('config')
		if model is None or scaler is None:
			raise ValueError("Invalid model file: missing model or scaler")
		print(f"Model loaded from {path}")
		return model, scaler, stored_config
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		return None, None, None


def load_model_interactive(config=None):
	model_path = input("Enter path to model file: ").strip()
	if not model_path:
		print("Model path is required.")
		return None, None, config
	model, scaler, stored_config = load_model(model_path)
	if model is None or scaler is None:
		return None, None, config

	if stored_config is not None:
		stored_config["_config_name"] = os.path.basename(model_path)
		stored_config["_config_path"] = model_path
		if "seed" in stored_config.get("model", {}):
			np.random.seed(stored_config["model"]["seed"])
			print(f"Random seed set to {stored_config['model']['seed']}")
		print("Config loaded from model file.")
		config = stored_config
	elif config is None:
		print("Warning: loaded model does not contain config. Load a config file to use predict/evaluate.")

	return model, scaler, config


def split_dataset(config):
	dataset_path = config["model"]["dataset_path"] if "dataset_path" in config["model"] else input("Enter the path to the dataset: ").strip()
	target = config["model"]["target"] if "target" in config["model"] else input("Enter the target label or index: ").strip()
	validation_split = config["model"]["validation_split"] if "validation_split" in config["model"] else input("Enter the validation split: ").strip()
	seed = config["model"].get("seed", None)

	try:
		validation_split = float(validation_split)
		if validation_split < 0 or validation_split > 1:
			raise ValueError("Validation split must be between 0 and 1")
		
		df = load_dataset(dataset_path)

		try:
			target = int(target)
		except:
			pass

		train, valid = train_test_split(df, target, float(validation_split), seed)
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
		
		if "seed" in config["model"]:
			np.random.seed(config["model"]["seed"])
		
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

		optimizers = config["model"].get("optimizers", config["model"].get("optimizer", "gradient_descent"))

		model.train(
			epochs=config["model"]["epochs"],
			learning_rate=config["model"]["learning_rate"],
			batch_size=config["model"]["batch_size"],
			loss=config["model"]["loss"],
			optimizers=optimizers
		)

		model.animation(
			epochs=config["model"]["epochs"],
			learning_rate=config["model"]["learning_rate"],
			batch_size=config["model"]["batch_size"],
			loss=config["model"]["loss"]
		)

		model_path = config["model"].get("saved_model_path")
		if not model_path:
			model_path = build_model_path_from_config(config)
			print(f"No saved_model_path in config. Using default path: {model_path}")
		if not model_path:
			raise ValueError("Model save path is required.")
		save_model(model, scaler, config, model_path)
		return model, scaler
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		raise e
		return None, None

def predict(model, scaler, config=None):
	try:
		if config is None:
			raise ValueError("Please load a config or model.")
		if model is None or scaler is None:
			raise ValueError("Please train a model or load one.")
		dataset_path = input("Enter the path to the dataset: ").strip()
		df = load_dataset(dataset_path)
		if df is None:
			raise ValueError("Could not load dataset for prediction.")
		features = model.features
		df[features] = scaler.transform(df[features])
		predictions, probabilities = model.predict(df[features])
		print("Predictions saved to datasets/predictions.csv")
		print("Prediction probabilities saved to datasets/predictions_proba.csv")
		return predictions, probabilities
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		return
	
def evaluate(config):
	try:
		if config is None:
			raise ValueError("Please load a config or model).")

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

		comp = pd.concat(
			[s1.rename("a").astype(str), s2.rename("b").astype(str)],
			axis=1,
			join="outer"
		)
		idx_only_a = s1.index.difference(s2.index)
		idx_only_b = s2.index.difference(s1.index)
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

		loss_name = config.get("model", {}).get("loss")
		if loss_name not in ["binary_crossentropy", "categorical_crossentropy"]:
			print(f"Info: unsupported loss '{loss_name}' for probabilistic evaluation.")
			return

		compute_prob_loss = input("Compute probabilistic loss from a probability CSV? [Y/n]: ").strip().lower()
		if compute_prob_loss in ["", "y", "yes"]:
			default_proba_path = os.path.join("datasets", "predictions_proba.csv")
			proba_path = input(f"Enter path to probability CSV [{default_proba_path}]: ").strip()
			proba_path = proba_path if proba_path else default_proba_path
			pred_proba = pd.read_csv(proba_path, index_col=0)

			truth_series = s1
			common_idx = pred_proba.index.intersection(truth_series.index)
			if len(common_idx) == 0:
				raise ValueError("No shared index between truth CSV and probability CSV.")

			pred_proba = pred_proba.loc[common_idx]
			y_true_raw = truth_series.loc[common_idx].astype(str)
			class_order = [str(col) for col in pred_proba.columns]
			class_to_index = {label: idx for idx, label in enumerate(class_order)}
			y_true_idx = y_true_raw.map(class_to_index)
			if y_true_idx.isna().any():
				missing_labels = sorted(y_true_raw.loc[y_true_idx.isna()].unique().tolist())
				raise ValueError(f"Unknown labels for probability columns: {missing_labels}")

			y_true_one_hot = np.zeros((len(common_idx), len(class_order)))
			y_true_one_hot[np.arange(len(common_idx)), y_true_idx.astype(int).to_numpy()] = 1
			y_pred = np.clip(pred_proba.to_numpy(), 1e-10, 1 - 1e-10)

			if loss_name == "categorical_crossentropy":
				loss_value = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
			elif loss_name == "binary_crossentropy":
				loss_value = -np.mean(np.sum(y_true_one_hot * np.log(y_pred) + (1 - y_true_one_hot) * np.log(1 - y_pred), axis=1))
			else:
				raise ValueError(f"Loss function {loss_name} not supported")

			y_pred_idx = np.argmax(y_pred, axis=1)
			accuracy = np.mean(y_pred_idx == y_true_idx.astype(int).to_numpy())

			print(f"Evaluation {loss_name} (probabilities): {loss_value:.6f}")
			print(f"Evaluation accuracy (probabilities -> argmax): {accuracy:.4f} ({accuracy * 100:.2f}%)")

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
			config["_config_path"] = sys.argv[1]
			config["_config_name"] = os.path.basename(sys.argv[1])
			if "seed" in config["model"]:
				np.random.seed(config["model"]["seed"])
				print(f"Random seed set to {config['model']['seed']}")

		while(True):
			print_header(config_loaded=(config is not None), model_ready=(model is not None and scaler is not None))
			print_menu()
			choice = input("Select an action > ").strip().lower()
			if choice == '1':
				config = load_config()
			elif choice == '2':
				model, scaler, config = load_model_interactive(config)
			elif choice == '3':
				train, valid = split_dataset(config)
			elif choice == '4':
				model, scaler = train_model(train, valid, config)
			elif choice == '5':
				predict(model, scaler, config)
			elif choice == '6':
				evaluate(config)
			elif choice in ['exit', 'quit', 'q']:
				return

	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		raise e

if __name__ == "__main__":
	main()