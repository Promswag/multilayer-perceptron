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

	numeric_features = df.select_dtypes(include='number').columns.difference([target])
	scaler = StandardScaler.StandardScaler()
	df.loc[:, numeric_features] = scaler.fit_transform(df.loc[:, numeric_features])
	df.loc[:, numeric_features] = df.loc[:, numeric_features].fillna(0)

	n_classes = len(output_classes)
	df.loc[:, target_col] = df.loc[:, target_col].map({v: k for k, v in output_classes.items()}).astype(int)
	m, n = df.shape[0], df.shape[1] - 1

	train, test = train_test_split(df, target, 0.2)

	model = MLP.MultiLayerPerceptron(train, test, target)
	layer1 = MLP.DenseLayer(n, 10, af.ReLU, af.ReLU_derivative, wi.he_uniform)
	layer2 = MLP.DenseLayer(layer1.n_neurons, 10, af.ReLU, af.ReLU_derivative, wi.he_uniform)
	l_out = MLP.DenseLayer(layer2.n_neurons, n_classes, af.softmax, af.softmax_derivative, wi.he_normal)

	features = train.columns.difference([target_col])
	X_train = train.loc[:, features]
	Y_train = train.loc[:, target_col].astype(int)
	X_test = test.loc[:, features]
	Y_test = test.loc[:, target_col].astype(int)

	batch_size = 20
	learning_rate = 0.1

	for epoch in range(100):
		indexes = np.random.permutation(X_train.index)
		for i in range(0, len(indexes), batch_size):
			indexes_batch = indexes[i:i + batch_size]
			mm = len(indexes_batch)
			X_batch = X_train.loc[indexes_batch].T
			Y_batch = Y_train.loc[indexes_batch]
			layer1.forward_propagation(X_batch)
			layer2.forward_propagation(layer1.forward_outputs)
			l_out.forward_propagation(layer2.forward_outputs)
			
			oh = model.one_hot(Y_batch, mm)

			l_out.backward_propagation(
				inputs=layer2.forward_outputs,
				one_hot=oh)
			layer2.backward_propagation(
				inputs=layer1.forward_outputs,
				W=l_out.weights,
				Z=l_out.dZ)
			layer1.backward_propagation(
				inputs=X_batch,
				W=layer2.weights,
				Z=layer2.dZ)
			
			layer1.update_parameters(learning_rate)
			layer2.update_parameters(learning_rate)
			l_out.update_parameters(learning_rate)
			
		cost = model.compute_cost(l_out.forward_outputs, Y_batch, mm)
		if epoch % 10 == 0:
			Y_pred = np.argmax(l_out.forward_outputs, axis=0)
			print(f"Epoch {epoch} - Cost {cost:.5f} - Accuracy {np.sum(Y_pred == Y_batch)/len(Y_pred):.3f}")

	layer1.forward_propagation(X_test.T)
	layer2.forward_propagation(layer1.forward_outputs)
	l_out.forward_propagation(layer2.forward_outputs)
	Y_pred = np.argmax(l_out.forward_outputs, axis=0)
	print(f"Accuracy {np.sum(Y_pred == Y_test)/len(Y_pred):.3f}")
	pd.DataFrame(Y_pred, index=X_test.index).to_csv("ressources/pred.csv", header=None)
	Y_test.to_csv("ressources/mdr.csv", header=None)
if __name__ == "__main__":
	main()