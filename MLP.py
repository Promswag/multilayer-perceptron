import numpy as np
import pandas as pd
import DenseLayer as dl
import math

class MultiLayerPerceptron():
	def __init__(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str, layers:int=None):
		self.init_datasets(data_train, data_valid, target)
		self.classes = self.Y_train.unique()
		self.n_classes = len(self.classes)
		self.n_features = data_train.shape[1] - 1
		self.layers = []
		self.n_layers = 0

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
		self.layers.append(
			dl.DenseLayer(
				n_inputs=(self.n_features if self.n_layers == 0 else self.layers[self.n_layers - 1].n_neurons),
				n_neurons=(self.n_classes if n_neurons == 0 else n_neurons),
				activation_function=activation_function,
				weights_initializer=weights_initializer
			)
		)
		self.n_layers += 1

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
		if batch_size == 0:
			batch_size = len(self.X_train)
		ratio = math.ceil(epochs / batch_size)

		for epoch in range(epochs):
			i = epoch % ratio
			if i == 0:
				indexes = np.random.permutation(self.X_train.index)
			indexes_batch = indexes[i:i + batch_size]
			m = len(indexes_batch)
			X_batch = self.X_train.loc[indexes_batch].T
			Y_batch = self.Y_train.loc[indexes_batch]

			for idx, layer in self.layers:
				layer.forward_propagation(X_batch if idx == 0 else self.layers[idx - 1])
			
			cost = self.compute_cost(self.layers[-1], Y_batch, m)

			for idx, layer in self.layers:
				layer.backward_propagation(
					self.layers[-1 - idx].forward_outputs
				)

	def backward_propagation(self, inputs:None, one_hot:None=None, W:None=None, Z:None=None):

