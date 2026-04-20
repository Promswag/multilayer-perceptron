import numpy as np
import activation_functions as af
import weights_initializer as wi

AF = {
	'ReLU': [af.relu, af.relu_derivative],
	'sigmoid': [af.sigmoid, af.sigmoid_derivative],
	'softmax': [af.softmax, None]
}
WI = {
	'HeNormal': wi.he_normal,
	'HeUniform': wi.he_uniform,
	'GlorotNormal': wi.glorot_normal,
	'GlorotUniform': wi.glorot_uniform,
	'LecunNormal': wi.lecun_normal,
	'LecunUniform': wi.lecun_uniform,
}

class DenseLayer():
	def __init__(self, n_inputs: int, n_neurons: int, activation_function: str, weights_initializer: str):
		self.n_inputs = n_inputs
		self.n_neurons = n_neurons
		self.activation_function = AF[activation_function][0]
		self.derivative_function = AF[activation_function][1]
		self.weights_initializer = WI[weights_initializer]

		#Adam optimizer
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8
		self.t = 0


	def init_weights(self):
		self.weights = self.weights_initializer(self.n_inputs, self.n_neurons)
		self.biases = np.zeros((self.n_neurons, 1))

		#Adam optimizer
		self.mW = np.zeros_like(self.weights)
		self.vW = np.zeros_like(self.weights)
		self.mb = np.zeros_like(self.biases)
		self.vb = np.zeros_like(self.biases)

	def forward_propagation(self, inputs):
		values = np.dot(self.weights, inputs) + self.biases
		self.saved = values
		self.forward_outputs = self.activation_function(values)
		return self.forward_outputs

	def backward_propagation(self, inputs:None, one_hot:None=None, W:None=None, Z:None=None):
		m = len(inputs)

		if one_hot is not None:
			dZ = self.forward_outputs - one_hot
		else:
			dZ = np.dot(W.T, Z) * self.derivative_function(self.saved)

		dW = np.dot(dZ, inputs.T) / m
		db = np.sum(dZ, axis=1, keepdims=True) /  m

		self.dW = dW
		self.db = db
		self.dZ = dZ

	def update_parameters(self, learning_rate: float = 0.1):
		# Clip gradients to prevent explosion
		self.dW = np.clip(self.dW, -5, 5)
		self.db = np.clip(self.db, -5, 5)
		
		self.weights -= learning_rate * self.dW
		self.biases -= learning_rate * self.db
		
		# Clip weights to prevent divergence
		self.weights = np.clip(self.weights, -10, 10)
		self.biases = np.clip(self.biases, -10, 10)

	def update_parameters_adam(self, learning_rate: float = 0.1):
		self.t += 1
		self.mW = self.beta1 * self.mW + (1 - self.beta1) * self.dW
		self.vW = self.beta2 * self.vW + (1 - self.beta2) * (self.dW ** 2)
		self.mb = self.beta1 * self.mb + (1 - self.beta1) * self.db
		self.vb = self.beta2 * self.vb + (1 - self.beta2) * (self.db ** 2)

		mW_hat = self.mW / (1 - self.beta1 ** self.t)
		vW_hat = self.vW / (1 - self.beta2 ** self.t)
		mb_hat = self.mb / (1 - self.beta1 ** self.t)
		vb_hat = self.vb / (1 - self.beta2 ** self.t)

		self.weights -= learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
		self.biases -= learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
		
		# Clip weights to prevent divergence
		self.weights = np.clip(self.weights, -10, 10)
		self.biases = np.clip(self.biases, -10, 10)
