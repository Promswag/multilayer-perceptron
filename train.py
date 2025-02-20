import numpy as np
import pandas as pd
from train_test_split import train_test_split
from sklearn.datasets import load_digits
import activation_functions as af
import weights_initializer as wi
import StandardScaler
import MLP

def main():
	# df = pd.read_csv("ressources/data.csv", index_col=0, header=None)
	df = pd.DataFrame(load_digits(as_frame=True).frame)
	# print(df)

	target = "target"
	# target = 0
	target_col = target	
	if isinstance(target, int):
		target_col = df.columns[target]
	output_classes = {i: c for i, c in enumerate(df.loc[:, target_col].unique())}
	n_classes = len(output_classes)
	df.loc[:, target_col] = df.loc[:, target_col].map({v: k for k, v in output_classes.items()})

	numeric_features = df.select_dtypes(include='number').columns.difference([target])
	scaler = StandardScaler.StandardScaler()

	df.loc[:, numeric_features] = scaler.fit_transform(df.loc[:, numeric_features])
	df = df.fillna(0)
	m, n = df.shape[0], df.shape[1] - 1

	train, test = train_test_split(df, target, 0.2)

	model = MLP.MultiLayerPerceptron(train, test, target)
	layer1 = MLP.DenseLayer(n, 10, af.ReLU, af.ReLU_derivative, wi.he_uniform)
	layer2 = MLP.DenseLayer(layer1.n_neurons, 10, af.ReLU, af.ReLU_derivative, wi.he_uniform)
	l_out = MLP.DenseLayer(layer2.n_neurons, n_classes, af.softmax, af.softmax_derivative, wi.he_normal)

	features = train.columns.difference([target_col])
	X_train = train.loc[:, features]
	Y_train = train.loc[:, target_col]
	X_test = test.loc[:, features]
	Y_test = test.loc[:, target_col]

	batch_size = 20
	learning_rate = 0.1

	for epoch in range(100):
		indexes = np.random.permutation(X_train.index)
		for i in range(0, m, batch_size):
			indexes_batch = indexes[i:i + batch_size]
			mm = len(indexes_batch)
			X_batch = X_train.loc[indexes_batch]
			Y_batch = Y_train.loc[indexes_batch]
			layer1.forward_propagation(X_batch.T)
			layer2.forward_propagation(layer1.forward_outputs)
			l_out.forward_propagation(layer2.forward_outputs)

			cost = model.compute_cost(l_out.forward_outputs, Y_batch, mm)

			if epoch % 10 == 0:
				Y_pred = np.argmax(l_out.forward_outputs, axis=0)
				print(f"Epoch {epoch} - Cost {cost:.5f} - Accuracy {np.sum(Y_pred == Y_batch)/len(Y_pred):.3f}")

			delta = l_out.forward_outputs - model.one_hot(Y_batch, mm)
			l_out.backward_propagation(l_out.saved, learning_rate)
			layer2.backward_propagation(l_out.backward_outputs, learning_rate)
			layer1.backward_propagation(layer2.backward_outputs, learning_rate)

if __name__ == "__main__":
	main()