import numpy as np
import pandas as pd
import DenseLayer as dl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import os
from datetime import datetime

class MultiLayerPerceptron():
	def __init__(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str, early_stopping_patience: int=None):
		self.classes = {i: c for i, c in enumerate(data_train.loc[:, target].unique())}
		self.n_classes = len(self.classes)
		self.init_datasets(data_train, data_valid, target)
		self.n_features = data_train.shape[1] - 1
		self.layers = []
		self.n_layers = 0

		#Early Stopping
		self.early_stopping_patience = early_stopping_patience
		self.early_stopping_counter = 0
		self.early_stopping_best = np.inf
		self.early_stopping_best_layers = None
		self.best_optimizer = None
		self.optimizers = []

	def init_datasets(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str):
		if isinstance(target, int):
			target = data_train.columns[target]
		features = data_train.columns.difference([target])
		class_to_index = {v: k for k, v in self.classes.items()}

		data_train = data_train.copy()
		data_valid = data_valid.copy()

		data_train[target] = data_train[target].map(class_to_index).astype(int)
		self.X_train = data_train.loc[:, features]
		self.Y_train = data_train.loc[:, target].astype(int)

		data_valid[target] = data_valid[target].map(class_to_index).astype(int)
		self.X_valid = data_valid.loc[:, features]
		self.Y_valid = data_valid.loc[:, target].astype(int)

		self.target = target
		self.features = features
	
	def init_layers(self):
		for layer in self.layers:
			layer.init_weights()

	def _resolve_optimizers(self, optimizers):
		if optimizers is None:
			optimizers = ["gradient_descent"]
		elif isinstance(optimizers, str):
			optimizers = [optimizers]

		aliases = {
			"gd": "gradient_descent",
			"sgd": "gradient_descent",
			"gradient_descent": "gradient_descent",
			"adam": "adam",
		}

		resolved = []
		for optimizer in optimizers:
			key = str(optimizer).strip().lower().replace("-", "_")
			if key not in aliases:
				raise ValueError(f"Optimizer {optimizer} not supported")
			name = aliases[key]
			if name not in resolved:
				resolved.append(name)

		if len(resolved) == 0:
			resolved = ["gradient_descent"]
		return resolved

	def _apply_optimizer_step(self, layers, optimizer_name, learning_rate):
		for layer in layers:
			if optimizer_name == "adam":
				layer.update_parameters_adam(learning_rate)
			else:
				layer.update_parameters(learning_rate)

	def add_layer(self, n_neurons: int, activation_function: str, weights_initializer: str):
		self.layers.append(
			dl.DenseLayer(
				n_inputs=(self.n_features if self.n_layers == 0 else self.layers[self.n_layers - 1].n_neurons),
				n_neurons=(self.n_classes if n_neurons == 0 else n_neurons),
				activation_function=activation_function,
				weights_initializer=weights_initializer))
		self.n_layers += 1

	def one_hot(self, Y, m):
		one_hot_Y = np.zeros((m, self.n_classes))
		one_hot_Y[np.arange(m), Y] = 1
		one_hot_Y = one_hot_Y.T
		return one_hot_Y
	
	def compute_cost(self, Y_pred, Y_true, m, loss):
		Y_true = self.one_hot(Y_true, m)
		epsilon = 1e-10
		Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
		if loss == 'categorical_crossentropy':
			return - 1 / m * np.sum(Y_true * np.log(Y_pred))
		elif loss == 'binary_crossentropy':
			return - 1 / m * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
		else:
			raise ValueError(f"Loss function {loss} not supported")
		
	def early_stopping_handler(self, Y: pd.DataFrame, layers:dl.DenseLayer, loss:str):
		output = layers[-1].forward_outputs
		cost = self.compute_cost(output, Y, len(Y), loss)
		if cost < self.early_stopping_best:
			self.early_stopping_best = cost
			self.early_stopping_counter = 0
			self.early_stopping_best_layers = copy.deepcopy(layers)
		else:
			self.early_stopping_counter += 1
			if self.early_stopping_counter == self.early_stopping_patience:
				return False
		return True
		
	def animation(self, epochs: int, learning_rate: float, batch_size: int, loss: str):
		"""Display animation using pre-trained metrics from train() method."""
		if not hasattr(self, 'loss_train') or len(self.loss_train) == 0:
			print("No training metrics found. Call train() before animation().")
			return

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
		fig.suptitle('Training Neural Network')
		
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')
		ax2.set_xlabel('Epochs')
		ax2.set_ylabel('Accuracy')

		optimizer_labels = {
			"gradient_descent": "GD",
			"adam": "ADAM",
		}
		train_colors = ['blue', 'red', 'green', 'brown', 'black']
		valid_colors = ['orange', 'purple', 'cyan', 'magenta', 'gray']

		loss_train_lines = {}
		loss_valid_lines = {}
		accuracy_train_lines = {}
		accuracy_valid_lines = {}

		for idx, optimizer in enumerate(self.optimizers):
			label = optimizer_labels.get(optimizer, optimizer.upper())
			loss_train_lines[optimizer], = ax1.plot([], [], label=f'Training Loss {label}', color=train_colors[idx % len(train_colors)])
			loss_valid_lines[optimizer], = ax1.plot([], [], label=f'Validation Loss {label}', color=valid_colors[idx % len(valid_colors)])
			accuracy_train_lines[optimizer], = ax2.plot([], [], label=f'Training Accuracy {label}', color=train_colors[idx % len(train_colors)])
			accuracy_valid_lines[optimizer], = ax2.plot([], [], label=f'Validation Accuracy {label}', color=valid_colors[idx % len(valid_colors)])

		ax1.legend()
		ax2.legend()
		ax1.grid()
		ax2.grid()

		def frame_update(epoch):
			all_loss_values = []
			all_accuracy_values = []

			for optimizer in self.optimizers:
				train_loss_values = self.loss_train[optimizer][:epoch + 1]
				valid_loss_values = self.loss_valid[optimizer][:epoch + 1]
				train_acc_values = self.accuracy_train[optimizer][:epoch + 1]
				valid_acc_values = self.accuracy_valid[optimizer][:epoch + 1]

				loss_train_lines[optimizer].set_data(range(0, len(train_loss_values)), train_loss_values)
				loss_valid_lines[optimizer].set_data(range(0, len(valid_loss_values)), valid_loss_values)
				accuracy_train_lines[optimizer].set_data(range(0, len(train_acc_values)), train_acc_values)
				accuracy_valid_lines[optimizer].set_data(range(0, len(valid_acc_values)), valid_acc_values)

				all_loss_values.extend(train_loss_values)
				all_loss_values.extend(valid_loss_values)
				all_accuracy_values.extend(train_acc_values)
				all_accuracy_values.extend(valid_acc_values)

			max_epoch = max(len(self.loss_train[optimizer]) for optimizer in self.optimizers)
			ax1.set_xlim(0, max_epoch)
			ax2.set_xlim(0, max_epoch)

			if all_loss_values:
				ax1.set_ylim(0, max(all_loss_values))
			if all_accuracy_values:
				ax2.set_ylim(min(all_accuracy_values), 1)

			return [
				*loss_train_lines.values(),
				*loss_valid_lines.values(),
				*accuracy_train_lines.values(),
				*accuracy_valid_lines.values(),
			]

		max_epoch = max(len(self.loss_train[optimizer]) for optimizer in self.optimizers)
		anim = animation.FuncAnimation(fig, frame_update, frames=max_epoch, interval=1, repeat=False, blit=False)

		# Save the final training curves snapshot in the graphs directory.
		os.makedirs('graphs', exist_ok=True)
		frame_update(max_epoch - 1)
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		graph_path = os.path.join('graphs', f'training_curves_{timestamp}.png')
		fig.savefig(graph_path, dpi=150, bbox_inches='tight')
		print(f"Training curves saved to {graph_path}")

		plt.show(block=True)
		return

	def train(self, epochs: int, learning_rate: float, batch_size: int, loss: str, optimizers=None):
		self.init_layers()
		self.optimizers = self._resolve_optimizers(optimizers)
		self.optimizer_layers = {optimizer: copy.deepcopy(self.layers) for optimizer in self.optimizers}
		if batch_size == 0:
			batch_size = len(self.X_train)

		# Initialize metric storage
		self.loss_train = {optimizer: [] for optimizer in self.optimizers}
		self.loss_valid = {optimizer: [] for optimizer in self.optimizers}
		self.accuracy_train = {optimizer: [] for optimizer in self.optimizers}
		self.accuracy_valid = {optimizer: [] for optimizer in self.optimizers}

		early_stopping_state = {
			optimizer: {
				"best": np.inf,
				"counter": 0,
				"best_layers": copy.deepcopy(self.optimizer_layers[optimizer]),
			}
			for optimizer in self.optimizers
		}
		global_best_cost = np.inf
		global_best_optimizer = self.optimizers[0]

		for epoch in range(epochs):
			indexes = np.random.permutation(self.X_train.index)

			for i in range(0, len(self.X_train), batch_size):
				indexes_batch = indexes[i:i + batch_size]
				X_batch = self.X_train.loc[indexes_batch].T
				Y_batch = self.Y_train.loc[indexes_batch]

				for optimizer in self.optimizers:
					layers = self.optimizer_layers[optimizer]
					layers = self.forward_propagation(X_batch, layers)
					layers = self.backward_propagation(X_batch, Y_batch, layers)
					self._apply_optimizer_step(layers, optimizer, learning_rate)
					self.optimizer_layers[optimizer] = layers

			epoch_valid_costs = {}
			for optimizer in self.optimizers:
				layers = self.optimizer_layers[optimizer]
				train_output = self.forward_propagation(self.X_train.T, layers)[-1].forward_outputs
				valid_output = self.forward_propagation(self.X_valid.T, layers)[-1].forward_outputs

				cost_train = self.compute_cost(train_output, self.Y_train, len(self.Y_train), loss)
				cost_valid = self.compute_cost(valid_output, self.Y_valid, len(self.Y_valid), loss)
				train_pred = np.argmax(train_output, axis=0)
				valid_pred = np.argmax(valid_output, axis=0)
				train_acc = np.sum(train_pred == self.Y_train) / len(train_pred)
				valid_acc = np.sum(valid_pred == self.Y_valid) / len(valid_pred)

				self.loss_train[optimizer].append(cost_train)
				self.loss_valid[optimizer].append(cost_valid)
				self.accuracy_train[optimizer].append(train_acc)
				self.accuracy_valid[optimizer].append(valid_acc)
				epoch_valid_costs[optimizer] = cost_valid

				if cost_valid < early_stopping_state[optimizer]["best"]:
					early_stopping_state[optimizer]["best"] = cost_valid
					early_stopping_state[optimizer]["counter"] = 0
					early_stopping_state[optimizer]["best_layers"] = copy.deepcopy(layers)
				else:
					early_stopping_state[optimizer]["counter"] += 1

				if early_stopping_state[optimizer]["best"] < global_best_cost:
					global_best_cost = early_stopping_state[optimizer]["best"]
					global_best_optimizer = optimizer

				print(
					"Epoch {} [{}] - Training Cost {} - Validation Cost {} - Training Accuracy {} - Validation Accuracy {}".format(
						f"{epoch}".rjust(5),
						optimizer.upper(),
						f"{cost_train:.5f}".rjust(5),
						f"{cost_valid:.5f}".rjust(5),
						f"{train_acc:.3f}",
						f"{valid_acc:.3f}"
					)
				)

			if self.early_stopping_patience is not None:
				current_best_optimizer = min(epoch_valid_costs, key=epoch_valid_costs.get)
				if early_stopping_state[current_best_optimizer]["counter"] >= self.early_stopping_patience:
					print(
						f"Early stopping triggered by {current_best_optimizer.upper()} at epoch {epoch} "
						f"(patience={self.early_stopping_patience})."
					)
					break

		self.best_optimizer = global_best_optimizer
		self.early_stopping_best_layers = copy.deepcopy(early_stopping_state[global_best_optimizer]["best_layers"])
		self.layers = self.early_stopping_best_layers

	def predict(self, X: pd.DataFrame, layers: dl.DenseLayer|None=None):
		if layers is None:
			if self.early_stopping_best_layers is not None:
				layers = self.early_stopping_best_layers
			else:
				layers = self.layers
		layers = self.forward_propagation(X.T, layers)
		probabilities = pd.DataFrame(
			layers[-1].forward_outputs.T,
			index=X.index,
			columns=[self.classes[i] for i in range(self.n_classes)]
		)
		predictions = np.argmax(probabilities.to_numpy(), axis=1)
		df = pd.Series(predictions, name=self.target, index=X.index).map(self.classes)
		os.makedirs('datasets', exist_ok=True)
		df.to_csv('datasets/predictions.csv')
		probabilities.to_csv('datasets/predictions_proba.csv')
		return df, probabilities
	
	def forward_propagation(self, X: pd.DataFrame, layers: dl.DenseLayer):
		for idx, _ in enumerate(layers):
			layers[idx].forward_propagation(X if idx == 0 else layers[idx - 1].forward_outputs)
		return layers
	
	def backward_propagation(self, X: pd.DataFrame, Y: pd.Series, layers: dl.DenseLayer):
		for i in range(self.n_layers -1, -1, -1):
			inputs = X if i == 0 else layers[i - 1].forward_outputs

			if i == self.n_layers - 1:
				layers[i].backward_propagation(
					inputs=inputs,
					one_hot=self.one_hot(Y, len(Y))
				)
			else:
				layers[i].backward_propagation(
					inputs=inputs,
					W=layers[i + 1].weights,
					Z=layers[i + 1].dZ
				)
		return layers