import numpy as np
import pandas as pd
from train_test_split import train_test_split
from sklearn.datasets import load_digits
import StandardScaler
import MLP

def main():
	df = pd.read_csv("ressources/data.csv", index_col=0, header=None)
	# df = pd.DataFrame(load_digits(as_frame=True).frame)
	print(df)
	# target = "target"
	target = 0
	numeric_features = df.select_dtypes(include='number').columns.difference([target])
	scaler = StandardScaler.StandardScaler()

	df.loc[:, numeric_features] = scaler.fit_transform(df.loc[:, numeric_features])
	df = df.fillna(0)

	train, test = train_test_split(df, target, 0.2)

	model = MLP.MultiLayerPerceptron(train, test, target)
	print(model.X_train)
	print(model.Y_train)
	print(model.X_valid)
	print(model.Y_valid)

if __name__ == "__main__":
	main()