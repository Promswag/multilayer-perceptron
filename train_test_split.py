import pandas as pd

def train_test_split(dataset:pd.DataFrame, target: int|str, frac:float, random_state: int=None):

	dataset = dataset.sample(frac=1, random_state=random_state)

	if isinstance(target, int):
		indices = dataset.groupby(dataset.columns[target])
	elif isinstance(target, str):
		indices = dataset.groupby(target)
	else:
		raise ValueError(f"Target must be an index or a label.")
	
	if (round(int(dataset.shape[0] * (1 - frac))) < len(indices)):
		raise ValueError("Not enough samples to guarantee representation of all classes in training batch")

	train = []
	test = []

	for _, class_indices in indices:
		threshold = int(round(len(class_indices) * (1 - frac)))
		train.append(dataset.loc[class_indices.index[:threshold]])
		test.append(dataset.loc[class_indices.index[threshold:]])

	train = pd.concat(train).sample(frac=1, random_state=random_state)
	test = pd.concat(test).sample(frac=1, random_state=random_state)

	return train, test