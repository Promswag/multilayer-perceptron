import numpy as np

def tanh(x):
	return np.tanh(x)

def tanh_derivative(x):
	return 1 - np.tanh(x) ** 2

def relu(x):
	return np.maximum(0, x)

def relu_derivative(x):
	return x > 0

def leaky_relu(x, alpha=0.01):
	return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
	return np.where(x > 0, 1.0, alpha)

def sigmoid(x):
	x = np.clip(x, -709, 709)
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	x = sigmoid(x)
	return x * (1 - x)

def silu(x):
	return x * sigmoid(x)

def silu_derivative(x):
	s = sigmoid(x)
	return s + x * s * (1 - s)

def gelu(x):
	x = np.asarray(x)
	c = np.sqrt(2 / np.pi)
	u = c * (x + 0.044715 * (x ** 3))
	return 0.5 * x * (1 + np.tanh(u))

def gelu_derivative(x):
	x = np.asarray(x)
	c = np.sqrt(2 / np.pi)
	u = c * (x + 0.044715 * (x ** 3))
	t = np.tanh(u)
	du = c * (1 + 0.134145 * (x ** 2))
	return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du

def softmax(x):
	x = np.exp(np.clip(x - np.max(x), -709, 709))
	return x / np.sum(x, axis=0, keepdims=True)

def softmax_derivative(x):
	x = softmax(x)
	return np.diagflat(x) - np.outer(x, x)