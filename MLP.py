import numpy as np
import pandas as pd

class MultiLayerPerceptron():
	def __init__(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str, layers:int=None):
		self.init_datasets(data_train, data_valid, target)
		self.classes = self.Y_train.unique()
		self.n_classes = len(self.classes)

	def init_datasets(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, target: int|str):
		if isinstance(target, int):
			target = data_train.columns[target]
		features = data_train.columns.difference([target])
		self.X_train = data_train.loc[:, features]
		self.Y_train = data_train.loc[:, target]
		self.X_valid = data_valid.loc[:, features]
		self.Y_valid = data_valid.loc[:, target]

	def one_hot(self, Y, m):
		one_hot_Y = np.zeros(m, self.n_classes)
		one_hot_Y[np.arange(m), Y] = 1
		one_hot_Y = one_hot_Y.T
		return one_hot_Y
	
	def compute_cost(self, Y_pred, Y_true, m):
		epsilon = 1e-10
		Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
		return - 1 / m * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))

	def train():
		return


class DenseLayer():
	def __init__(self, n_inputs: int, n_neurons: int,
			  activation_function: callable,
			  derivative_function: callable,
			  weights_initializer: callable):
		self.activation_function = activation_function
		self.derivative_function = derivative_function
		self.weights = weights_initializer(n_inputs, n_neurons)
		self.biases = np.zeros((n_neurons, 1))

	def forward_propagation(self, inputs):
		values = np.dot(self.weights, inputs) + self.biases
		self.outputs = self.activation_function(values)
		return self.outputs

	def backward_propagation(self, inputs, learning_rate):
		m = len(inputs)
		dW = np.dot(inputs, self.outputs.T) / m
		db = np.sum(inputs, axis=1, keepdims=True) /  m
		values = np.dot(self.weights.T, self.outputs)
		self.outputs = self.derivative_function(values)
		self.weights -= learning_rate * dW
		self.biases -= learning_rate * db
		return self.outputs