import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from StandardScaler import StandardScaler
import activation_functions as af

AF = [
	[af.ReLU, af.ReLU_derivative],
	[af.sigmoid, af.sigmoid_derivative],
	[af.softmax, af.softmax_derivative],
]
ACTIVATION_FUNCTION = 0

def forward_prop(X, W, b):
	A = []
	A.append(X.values.T)
	for i in range(len(W)):
		z = np.dot(W[i], A[i]) + b[i]
		A.append(AF[ACTIVATION_FUNCTION][0](z))
	return af.softmax(z), A

def predict(X, W, b):
	A = []
	A.append(X.values.T)
	for i in range(len(W)):
		z = np.dot(W[i], A[i]) + b[i]
		A.append(AF[ACTIVATION_FUNCTION][0](z))
	return af.softmax(z), A

def compute_cost(pred, Y, m):
	pred = np.clip(pred, 1e-10, 1 - 1e-10)
	return -1 / m * np.sum(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))

def backward_prop(A, Y, W, m):
	dW = []
	db = []
	delta = A[-1] - Y
	for i in range(len(W) - 1, -1, -1):
		dW.append(np.dot(delta, A[i].T) / m)
		db.append(np.sum(delta, axis=1, keepdims=True) / m)
		if i > 0:
			delta = np.dot(W[i].T, delta) * AF[ACTIVATION_FUNCTION][1](A[i])

	dW.reverse()
	db.reverse()
	return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    for i in range(len(W)):
        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]
    return W, b

def one_hot(Y, n):
	oh = np.zeros((Y.size, n))
	oh[np.arange(Y.size), Y] = 1
	return oh.T


def main():
	try:
		# data = pd.read_csv("ressources/data.csv", index_col=0, header=None)
		data = pd.read_csv("ressources/test.csv", index_col=0, header=None)
		print(data)
		data = load_digits(as_frame=True)
		data = pd.DataFrame(data.frame)
		scaler = StandardScaler()

		targetT = 64

		Y = data.iloc[:, targetT]
		output_classes = {i: c for i, c in enumerate(Y.unique())}
		Y = Y.map({v: k for k, v in output_classes.items()})


		X = data.drop(data.columns[targetT], axis=1)
		X = scaler.fit_transform(X)
		X = X.fillna(0)

		print(Y)
		print(X)

		IDXS = np.random.permutation(X.index)
		SPLIT_IDX = int(Y.size * 0.25)
		X_dev = X.loc[IDXS[:SPLIT_IDX]]
		Y_dev = Y.loc[IDXS[:SPLIT_IDX]]
		X = X.loc[IDXS[SPLIT_IDX:]]
		Y = Y.loc[IDXS[SPLIT_IDX:]]


		m, n = X.shape
		batch_size = 100

		LAYERS = [16, 16, len(output_classes)]
		W = []
		b = []

		for layer in range(len(LAYERS)):
			if layer == 0:
				W.append(0.1*np.random.randn(LAYERS[layer], n))
			else:
				W.append(0.1*np.random.randn(LAYERS[layer], LAYERS[layer - 1]))
			b.append(np.zeros((LAYERS[layer], 1)))

		for epoch in range(100):
			indexes = np.random.permutation(X.index)
			for i in range(0, m, batch_size):
				indexes_batch = indexes[i:i + batch_size]
				X_batch = X.loc[indexes_batch]
				Y_batch = Y.loc[indexes_batch]
				one_hot_Y = one_hot(Y_batch, len(output_classes))
				mm = indexes_batch.size

				output, A = forward_prop(X_batch, W, b)
				cost = compute_cost(output, one_hot_Y, mm)
				dW, db = backward_prop(A, one_hot_Y, W, mm)
				W, b = update_parameters(W, b, dW, db, 1)

			if epoch % 10 == 0:
				output = np.argmax(output, axis=0)
				print(f"Epoch {epoch} - Cost {cost:.5f} - Accuracy {np.sum(output == Y_batch)/len(output):.3f}")

		output, rrr = forward_prop(X_dev, W, b)
		output = np.argmax(output, axis=0)
		print(f"{np.sum(output == Y_dev)/len(output):.3f}")
		pd.DataFrame(output, index=X_dev.index).to_csv("ressources/pred.csv", header=None)
		Y_dev.to_csv("ressources/mdr.csv", header=None)
		for idx, sample in enumerate(X_dev.iloc):
			plt.imshow(sample.values.reshape(8,8), cmap='gray')
			plt.title(Y_dev.iloc[idx])
			plt.show()
	except Exception as e:
		print(f"{type(e).__name__}: {e}")
		raise e

if __name__ == "__main__":
	main()