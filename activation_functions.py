import numpy as np

def tanh(x):
	return np.tanh(x)

def tanh_derivative(x):
	return 1 - np.tanh(x) ** 2

def relu(x):
	return np.maximum(0, x)

def relu_derivative(x):
	return x > 0

def sigmoid(x):
	x = np.clip(x, -709, 709)
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	x = sigmoid(x)
	return x * (1 - x)

def softmax(x):
	x = np.exp(np.clip(x - np.max(x), -709, 709))
	return x / np.sum(x, axis=0, keepdims=True)

def softmax_derivative(x):
	x = softmax(x)
	return np.diagflat(x) - np.outer(x, x)