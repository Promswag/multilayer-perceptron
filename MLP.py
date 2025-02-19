import numpy as np
import pandas as pd

class MultiLayerPerceptron():
	def __init__(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, layers:int=None):
		self.X_train = data_train.iloc[:, 1:]
		self.Y_train = data_train.iloc[:, 0]
		self.X_valid = data_valid.iloc[:, 1:]
		self.Y_valid = data_valid.iloc[:, 0]
		# self.classes_names = np.unique(Y)
		# self.n_classes = len(self.classes_names)
		# self.__split_data__(frac)
		pass

def main():
	df = pd.read_csv("ressources/data.csv", index_col=0, header=None)
	indices = np.array(df.index)
	# df = np.array(df)

	# target_idx = 0
	# X = np.delete(df, target_idx, axis=1)
	# Y = df[:, 0]
	MLP = MultiLayerPerceptron(data_train=df, data_validation=df, indices=indices)

if __name__ == "__main__":
	main()