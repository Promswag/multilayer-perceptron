import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from StandardScaler import StandardScaler
import activation_functions as af

def forward_prop(X, W, b):
	A = []
	A.append(X.values.T)
	for i in range(len(W)):
		z = np.dot(W[i], A[i]) + b[i]
		A.append(af.sigmoid(z))
	return af.softmax(z), A

def compute_cost(pred, Y, m):
	pred = np.clip(pred, 1e-10, 1 - 1e-10)
	return -1 / m * np.sum(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))

def backward_prop(A, Y, W, b, m):
	dW = []
	db = []
	delta = A[-1] - Y
	for i in range(len(W) - 1, -1, -1):
		dW.append(np.dot(delta, A[i].T) / m)
		db.append(np.sum(delta, axis=1, keepdims=True) / m)
		if i > 0:
			delta = np.dot(W[i].T, delta) * af.sigmoid_derivative(A[i])

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
		A.append(af.sigmoid(z))
	return af.softmax(z)

def main():
	data = pd.read_csv("ressources/test.csv", index_col=0, header=None)

	scaler = StandardScaler()
	data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

	m, n = data.iloc[:, 1:].shape
	Y = data.iloc[:, 0]
	X = data.iloc[:, 1:]

	output_classes = {i: c for i, c in enumerate(Y.unique())}
	Y = Y.map({v: k for k, v in output_classes.items()})

	LAYERS = [2, len(Y.unique())]
	W = []
	b = []

	for layer in range(len(LAYERS)):
		if layer == 0:
			W.append(0.1*np.random.randn(LAYERS[layer], n))
		else:
			W.append(0.1*np.random.randn(LAYERS[layer], LAYERS[layer - 1]))
		b.append(np.zeros((LAYERS[layer], 1)))

	# for arr in b:
	# 	print(arr.shape)
	# for arr in W:
	# 	print(arr.shape)

	one_hot_Y = one_hot(Y)

	for epoch in range(1000):
		output, A = forward_prop(X, W, b)
		# print(A)
		cost = compute_cost(output, one_hot_Y, m)
		dW, db = backward_prop(A, one_hot_Y, W, b, m)
		W, b = update_parameters(W, b, dW, db, 0.1)

		# print(output.T)
		# print(A)

		# print(W)
		# print(b)
		# print(output)
		if epoch % 100 == 0:
			output = np.argmax(output, axis=0)
			print(f"Epoch {epoch} - Cost {cost:.5f} - Accuracy {np.sum(output == Y)/len(output):.3f}")

	pred = np.argmax(predict(X, W, b).T, axis=1)
	print(np.sum(pred == Y)/len(pred))
	pd.Series(pred, index=data.index[:10]).map(output_classes).to_csv("ressources/pred.csv", header=None)
	Y.to_csv("ressources/mdr.csv", header=None)

if __name__ == "__main__":
	main()