import numpy as np
import pandas as pd
import activation_functions as af
from sklearn.datasets import load_digits
import StandardScaler

def forward_prop(X, W1, b1, W2, b2):
	Z1 = W1.dot(X) + b1
	A1 = af.ReLU(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = af.softmax(Z2)
	return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
	m = Y.size
	one_hot_Y = one_hot(Y)

	dZ2 = A2 - one_hot_Y
	dW2 = 1 / m * dZ2.dot(A1.T)
	db2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1,1)

	dZ1 = W2.T.dot(dZ2) * af.ReLU_derivative(Z1)
	dW1 = 1 / m * dZ1.dot(X.T)
	db1 = 1 / m * np.sum(dZ1, axis=1).reshape(-1,1) 
	return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
	W1 = W1 - dW1 * learning_rate
	b1 = b1 - db1 * learning_rate
	W2 = W2 - dW2 * learning_rate
	b2 = b2 - db2 * learning_rate
	return W1, b1, W2, b2

def get_predictions(A2):
	return np.argmax(A2, 0)

def accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def one_hot(Y):
	oh = np.zeros((Y.size, Y.max() + 1))
	oh[np.arange(Y.size), Y] = 1
	return oh.T

def main():
	# data = pd.read_csv("ressources/test.csv", index_col=0, header=None)
	# scaler = StandardScaler.StandardScaler()
	# data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

	# m, n = data.iloc[:, 1:].shape
	# Y_train = data.iloc[:, 0].T
	# X_train = data.iloc[:, 1:].T

	# output_classes = {i: c for i, c in enumerate(Y_train.unique())}
	# Y_train = Y_train.map({v: k for k, v in output_classes.items()})

	# W1 = np.random.rand(4, 3) - 0.5
	# b1 = np.random.rand(4, 1) - 0.5
	# W2 = np.random.rand(2, 4) - 0.5
	# b2 = np.random.rand(2, 1) - 0.5

	data = load_digits(as_frame=True)
	data = np.array(data.frame)
	m, n = data.shape
	np.random.shuffle(data)

	data_dev = data[:int(m * 0.25)].T
	Y_dev = data_dev[-1].astype(int)
	X_dev = data_dev[:-1]

	data_train = data[int(m * 0.25):].T
	Y_train = data_train[-1].astype(int)
	X_train = data_train[:-1]

	W1 = np.random.randn(10, 64)
	b1 = np.random.randn(10, 1)
	W2 = np.random.randn(10, 10)
	b2 = np.random.randn(10, 1)

	for epoch in range(1000):
		Z1, A1, Z2, A2 = forward_prop(X_train, W1, b1, W2, b2)
		dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X_train, Y_train)
		W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.1)
		if epoch % 100 == 0:
			print(f"Epoch {f'{epoch}'.rjust(6)} - Acc {accuracy(get_predictions(A2), Y_train):.5f}")

	print(get_predictions(A2))

if __name__ == "__main__":
	main()