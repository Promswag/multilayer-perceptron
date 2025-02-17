import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from StandardScaler import StandardScaler

def tanh(z):
	return np.tanh(z)

def tanh_derivative(z):
	return 1 - np.tanh(z) ** 2

def ReLU(z):
	return np.maximum(0, z)

def ReLU_derivative(z):
	return z > 0

def sigmoid(z):
	z = np.clip(z, -709, 709)
	return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
	z = np.exp(np.clip(z, -709, 709))
	return z / sum(z)

def forward_prop(X, W, b):
	A = []
	A.append(X.values.T)
	# print(A)
	# return
	for i in range(len(W)):
		z = np.dot(W[i], A[i]) + b[i]
		# print("WWWWWWWWWWW: ", W[i])
		# print("AAAAAAAAA: ", A[i])
		# print("BBBBBBBB: ", b[i])
		# print(z)
		A.append(sigmoid(z))
		# A.append(ReLU(z))
	return softmax(z), A

def compute_cost(pred, Y, m):
	pred = np.clip(pred, 1e-10, 1 - 1e-10)
	return -1 / m * np.sum(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))
	# return -1 / m * np.sum(Y * np.log(pred))

def backward_prop(A, Y, W, b, m):
	dW = []
	db = []
	delta = A[-1] - one_hot(Y)
	for i in range(len(W) - 1, -1, -1):
		dW.append(np.dot(delta, A[i].T) / m)
		db.append(np.sum(delta) / m)
		if i > 0:
			delta = np.dot(W[i].T, delta) * sigmoid_derivative(A[i])
			# delta = np.dot(W[i].T, delta) * ReLU_derivative(A[i])

	dW.reverse()
	db.reverse()
	return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    for i in range(len(W)):
        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]
    return W, b

def accuracy(A, Y):
    predictions = A[-1] > 0.5
    return np.mean(predictions == Y)

def one_hot(Y):
	oh = np.zeros((Y.size, Y.max() + 1))
	oh[np.arange(Y.size), Y] = 1
	return oh.T

def predict(X, W, b):
	A = []
	A.append(X.values.T)
	for i in range(len(W)):
		z = np.dot(W[i], A[i]) + b[i]
		A.append(sigmoid(z))
	return softmax(A[-1])

def main():
	data = pd.read_csv("ressources/data.csv", index_col=0, header=None)
	# data = pd.read_csv("ressources/test.csv", index_col=0, header=None)
	# data = load_digits(as_frame=True)
	# data = pd.DataFrame(data.frame)
	scaler = StandardScaler()
	data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

	m, n = data.iloc[:, 1:].shape
	Y = (data.iloc[:, 0] == "M").astype(int)
	X = data.iloc[:, 1:]

	# data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
	# # data.iloc[:, :-1] = data.iloc[:, :-1] / data.iloc[:, :-1].max()
	# data = data.fillna(0)
	# m, n = data.iloc[:, :-1].shape
	# Y = data.iloc[:, -1]
	# X = data.iloc[:, :-1]

	LAYERS = [10, 10, 10, 2]
	W = []
	b = []

	for layer in range(len(LAYERS)):
		if layer == 0:
			W.append(np.random.randn(LAYERS[layer], n))
			# W.append(np.random.uniform(-1, 1, size=[LAYERS[layer], n]))
		else:
			W.append(np.random.randn(LAYERS[layer], LAYERS[layer - 1]))
			# W.append(np.random.uniform(-1, 1, size=[LAYERS[layer], LAYERS[layer - 1]]))
		b.append(np.zeros((LAYERS[layer], 1)))

	for epoch in range(1000):
		output, A = forward_prop(X, W, b)
		# return
		cost = compute_cost(output, one_hot(Y), m)
		dW, db = backward_prop(A, Y, W, b, m)
		W, b = update_parameters(W, b, dW, db, 1)
		if epoch % 100 == 0:
			output = np.argmax(output.T, axis=1)
			print(f"Epoch {epoch} - Cost {cost:.5f} - Accuracy {(output == Y).value_counts().iloc[0]/len(output):.3f}")

	pred = np.argmax(predict(X, W, b).T, axis=1)
	print(f"{(pred == Y).value_counts().iloc[0]/len(pred):.3f}")
	pd.DataFrame(np.where(pred == 1, 'M', 'B'), index=data.index).to_csv("ressources/pred.csv", header=None)

if __name__ == "__main__":
	main()