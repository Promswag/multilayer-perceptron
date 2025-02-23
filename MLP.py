import numpy as np
import pandas as pd
import DenseLayer as dl

class MultiLayerPerceptron():
	def __init__(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str):
		self.classes = {i: c for i, c in enumerate(data_train.loc[:, target].unique())}
		self.n_classes = len(self.classes)
		self.init_datasets(data_train, data_valid, target)
		self.n_features = data_train.shape[1] - 1
		self.layers = []
		self.n_layers = 0

	def init_datasets(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str):
		if isinstance(target, int):
			target = data_train.columns[target]
		features = data_train.columns.difference([target])
		data_train.loc[:, target] = data_train.loc[:, target].map({v: k for k, v in self.classes.items()}).astype(int)
		self.X_train = data_train.loc[:, features]
		self.Y_train = data_train.loc[:, target].astype(int)
		data_valid.loc[:, target] = data_valid.loc[:, target].map({v: k for k, v in self.classes.items()}).astype(int)
		self.X_valid = data_valid.loc[:, features]
		self.Y_valid = data_valid.loc[:, target].astype(int)
	
	def init_layers(self):
		for layer in self.layers:
			layer.init_weights()

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

	def train(self, epochs: int, learning_rate: float, batch_size: int, loss: str):
		if batch_size == 0:
			batch_size = len(self.X_train)

		for epoch in range(epochs):
			indexes = np.random.permutation(self.X_train.index)

			for i in range(0, len(self.X_train), batch_size):
				indexes_batch = indexes[i:i + batch_size]
				m = len(indexes_batch)
				X_batch = self.X_train.loc[indexes_batch].T
				Y_batch = self.Y_train.loc[indexes_batch]

				for idx, layer in enumerate(self.layers):
					layer.forward_propagation(X_batch if idx == 0 else self.layers[idx - 1].forward_outputs)
				

				for i in range(self.n_layers -1, -1, -1):
					inputs = X_batch if i == 0 else self.layers[i - 1].forward_outputs

					if i == self.n_layers - 1:
						self.layers[i].backward_propagation(
							inputs=inputs,
							one_hot=self.one_hot(Y_batch, m)
						)
					else:
						self.layers[i].backward_propagation(
							inputs=inputs,
							W=self.layers[i + 1].weights,
							Z=self.layers[i + 1].dZ
						)

				for layer in self.layers:
					layer.update_parameters(learning_rate)


			for idx, layer in enumerate(self.layers):
				layer.forward_propagation(self.X_train.T if idx == 0 else self.layers[idx - 1].forward_outputs)
			cost_train = self.compute_cost(self.layers[-1].forward_outputs, self.Y_train, len(self.Y_train), loss)
			Y_train_pred = np.argmax(self.layers[-1].forward_outputs, axis=0)

			for idx, layer in enumerate(self.layers):
				layer.forward_propagation(self.X_valid.T if idx == 0 else self.layers[idx - 1].forward_outputs)
			cost_valid = self.compute_cost(self.layers[-1].forward_outputs, self.Y_valid, len(self.Y_valid), loss)
			Y_val_pred = np.argmax(self.layers[-1].forward_outputs, axis=0)

			print("Epoch {} - Training Cost {} - Validation Cost {} - Training Accuracy {} - Validation Accuracy {}".format(
				f"{epoch}".rjust(5),
				f"{cost_train:.5f}".rjust(5),
				f"{cost_valid:.5f}".rjust(5),
				f"{np.sum(Y_train_pred == self.Y_train)/len(Y_train_pred):.3f}",
				f"{np.sum(Y_val_pred == self.Y_valid)/len(Y_val_pred):.3f}"
			))