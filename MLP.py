import numpy as np
import pandas as pd
import DenseLayer as dl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

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
		self.adam_layers = copy.deepcopy(self.layers)

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

		loss_train_line, = ax1.plot([], [], label='Training Loss', color='blue')
		loss_valid_line, = ax1.plot([], [], label='Validation Loss', color='orange')
		loss_train_adam_line, = ax1.plot([], [], label='Training Loss ADAM', color='red')
		loss_valid_adam_line, = ax1.plot([], [], label='Validation Loss ADAM', color='purple')

		accuracy_train_line, = ax2.plot([], [], label='Training Accuracy', color='blue')
		accuracy_valid_line, = ax2.plot([], [], label='Validation Accuracy', color='orange')
		accuracy_train_adam_line, = ax2.plot([], [], label='Training Accuracy ADAM', color='red')
		accuracy_valid_adam_line, = ax2.plot([], [], label='Validation Accuracy ADAM', color='purple')

		ax1.legend()
		ax2.legend()
		ax1.grid()
		ax2.grid()

		def frame_update(epoch):
			loss_train_line.set_data(range(0, len(self.loss_train[:epoch+1])), self.loss_train[:epoch+1])
			loss_valid_line.set_data(range(0, len(self.loss_valid[:epoch+1])), self.loss_valid[:epoch+1])
			loss_train_adam_line.set_data(range(0, len(self.loss_train_adam[:epoch+1])), self.loss_train_adam[:epoch+1])
			loss_valid_adam_line.set_data(range(0, len(self.loss_valid_adam[:epoch+1])), self.loss_valid_adam[:epoch+1])
			
			accuracy_train_line.set_data(range(0, len(self.accuracy_train[:epoch+1])), self.accuracy_train[:epoch+1])
			accuracy_valid_line.set_data(range(0, len(self.accuracy_valid[:epoch+1])), self.accuracy_valid[:epoch+1])
			accuracy_train_adam_line.set_data(range(0, len(self.accuracy_train_adam[:epoch+1])), self.accuracy_train_adam[:epoch+1])
			accuracy_valid_adam_line.set_data(range(0, len(self.accuracy_valid_adam[:epoch+1])), self.accuracy_valid_adam[:epoch+1])

			ax1.set_xlim(0, len(self.loss_train))
			ax2.set_xlim(0, len(self.accuracy_train))
			
			if self.loss_train:
				ax1.set_ylim(0, max(max(self.loss_train), max(self.loss_valid), max(self.loss_train_adam), max(self.loss_valid_adam)))
			if self.accuracy_train:
				ax2.set_ylim(min(min(self.accuracy_train), min(self.accuracy_valid), min(self.accuracy_train_adam), min(self.accuracy_valid_adam)), 1)

			return loss_train_line, loss_valid_line, loss_train_adam_line, loss_valid_adam_line, accuracy_train_line, accuracy_valid_line, accuracy_train_adam_line, accuracy_valid_adam_line

		anim = animation.FuncAnimation(fig, frame_update, frames=len(self.loss_train), interval=1, repeat=False, blit=False)
		plt.show(block=True)
		return

	def train(self, epochs: int, learning_rate: float, batch_size: int, loss: str):
		self.init_layers()
		if batch_size == 0:
			batch_size = len(self.X_train)

		# Initialize metric storage
		self.loss_train = []
		self.loss_valid = []
		self.loss_train_adam = []
		self.loss_valid_adam = []
		self.accuracy_train = []
		self.accuracy_valid = []
		self.accuracy_train_adam = []
		self.accuracy_valid_adam = []

		for epoch in range(epochs):
			indexes = np.random.permutation(self.X_train.index)

			for i in range(0, len(self.X_train), batch_size):
				indexes_batch = indexes[i:i + batch_size]
				X_batch = self.X_train.loc[indexes_batch].T
				Y_batch = self.Y_train.loc[indexes_batch]

				self.layers = self.forward_propagation(X_batch, self.layers)
				self.adam_layers = self.forward_propagation(X_batch, self.adam_layers)

				self.layers = self.backward_propagation(X_batch, Y_batch, self.layers)
				self.adam_layers = self.backward_propagation(X_batch, Y_batch, self.adam_layers)

				for layer in self.layers:
					layer.update_parameters(learning_rate)
				for layer in self.adam_layers:
					layer.update_parameters_adam(learning_rate)


			bgd = self.forward_propagation(self.X_train.T, self.layers)[-1].forward_outputs
			cost_train = self.compute_cost(bgd, self.Y_train, len(self.Y_train), loss)
			Y_train_pred = np.argmax(bgd, axis=0)
			bgd_valid = self.forward_propagation(self.X_valid.T, self.layers)[-1].forward_outputs
			cost_valid = self.compute_cost(bgd_valid, self.Y_valid, len(self.Y_valid), loss)
			Y_val_pred = np.argmax(bgd_valid, axis=0)

			adam = self.forward_propagation(self.X_train.T, self.adam_layers)[-1].forward_outputs
			cost_train_adam = self.compute_cost(adam, self.Y_train, len(self.Y_train), loss)
			Y_train_pred_adam = np.argmax(adam, axis=0)
			adam_valid = self.forward_propagation(self.X_valid.T, self.adam_layers)[-1].forward_outputs
			cost_valid_adam = self.compute_cost(adam_valid, self.Y_valid, len(self.Y_valid), loss)
			Y_val_pred_adam = np.argmax(adam_valid, axis=0)

			# Store metrics
			self.loss_train.append(cost_train)
			self.loss_valid.append(cost_valid)
			self.loss_train_adam.append(cost_train_adam)
			self.loss_valid_adam.append(cost_valid_adam)
			self.accuracy_train.append(np.sum(Y_train_pred == self.Y_train)/len(Y_train_pred))
			self.accuracy_valid.append(np.sum(Y_val_pred == self.Y_valid)/len(Y_val_pred))
			self.accuracy_train_adam.append(np.sum(Y_train_pred_adam == self.Y_train)/len(Y_train_pred_adam))
			self.accuracy_valid_adam.append(np.sum(Y_val_pred_adam == self.Y_valid)/len(Y_val_pred_adam))

			if self.early_stopping_patience is not None:
				if not self.early_stopping_handler(self.Y_valid, self.adam_layers, loss):
					break

			print("Epoch {} - Training Cost {} - Validation Cost {} - Training Accuracy {} - Validation Accuracy {}".format(
				f"{epoch}".rjust(5),
				f"{cost_train:.5f}".rjust(5),
				f"{cost_valid:.5f}".rjust(5),
				f"{np.sum(Y_train_pred == self.Y_train)/len(Y_train_pred):.3f}",
				f"{np.sum(Y_val_pred == self.Y_valid)/len(Y_val_pred):.3f}"
			))
			print("Adamh {} - Training ADAM {} - Validation ADAM {} - Training ADAMracy {} - Validation ADAMracy {}".format(
				f"{epoch}".rjust(5),
				f"{cost_train_adam:.5f}".rjust(5),
				f"{cost_valid_adam:.5f}".rjust(5),
				f"{np.sum(Y_train_pred_adam == self.Y_train)/len(Y_train_pred_adam):.3f}",
				f"{np.sum(Y_val_pred_adam == self.Y_valid)/len(Y_val_pred_adam):.3f}"
			))

	def predict(self, X: pd.DataFrame, layers: dl.DenseLayer|None=None):
		if layers is None:
			if self.early_stopping_best_layers is not None:
				layers = self.early_stopping_best_layers
			else:
				layers = self.layers
		layers = self.forward_propagation(X.T, layers)
		predictions = np.argmax(layers[-1].forward_outputs, axis=0)
		df = pd.Series(predictions, name=self.target, index=X.index).map(self.classes)
		df.to_csv('datasets/predictions.csv')
		return df
	
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