import numpy as np
import pandas as pd
import DenseLayer as dl

class MultiLayerPerceptron():
	def __init__(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str, layers:int=None):
		self.init_datasets(data_train, data_valid, target)
		self.classes = self.Y_train.unique()
		self.n_classes = len(self.classes)
		self.n_features = data_train.shape[1] - 1
		self.layers = []

	def init_datasets(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str):
		if isinstance(target, int):
			target = data_train.columns[target]
		features = data_train.columns.difference([target])
		self.X_train = data_train.loc[:, features]
		self.Y_train = data_train.loc[:, target].astype(int)
		self.X_valid = data_valid.loc[:, features]
		self.Y_valid = data_valid.loc[:, target].astype(int)
	
	def init_layers(self):
		for layer in self.layers:
			layer.init_weights()

	def add_layer(self, n_neurons: int, activation_function: str, weights_initializer: str):
		n_layer = len(self.layers)
		print(n_layer)
		self.layers.append(
			dl.DenseLayer(
				n_inputs=(self.n_features if n_layer == 0 else self.layers[n_layer - 1].n_neurons),
				n_neurons=(self.n_classes if n_neurons == 0 else n_neurons),
				activation_function=activation_function,
				weights_initializer=weights_initializer
			)
		)

	def one_hot(self, Y, m):
		one_hot_Y = np.zeros((m, self.n_classes))
		one_hot_Y[np.arange(m), Y] = 1
		one_hot_Y = one_hot_Y.T
		return one_hot_Y
	
	def compute_cost(self, Y_pred, Y_true, m):
		Y_true = self.one_hot(Y_true, m)
		epsilon = 1e-10
		Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
		return - 1 / m * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))

	def train(self, epochs: int, learning_rate: float, batch_size: int, loss: str):
		
		return

