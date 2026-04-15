import pandas as pd
from train_test_split import train_test_split
import sys
import weights_initializer as wi
import matplotlib.pyplot as plt

def main():
	fig, axes = plt.subplots(3, 2, figsize=(10, 15))
	axes = axes.flatten()
	axes[0].hist(wi.he_normal(10, 100).flatten(), bins=100)
	axes[1].hist(wi.he_uniform(10, 100).flatten(), bins=100)
	axes[2].hist(wi.glorot_normal(10, 100).flatten(), bins=100)
	axes[3].hist(wi.glorot_uniform(10, 100).flatten(), bins=100)
	axes[4].hist(wi.lecun_normal(10, 100).flatten(), bins=100)
	axes[5].hist(wi.lecun_uniform(10, 100).flatten(), bins=100)
	plt.show()
	return
	if len(sys.argv) != 4:
		print("Usage: python split.py <Dataset> <Target label|index> <Validation split>")
		return 0
	try:
		try:
			target = int(sys.argv[2])
		except:
			target = sys.argv[2]
		df = pd.read_csv(f"datasets/{sys.argv[1]}", index_col=0, header=None)
		train, test = train_test_split(df, target, float(sys.argv[3]))
		train.to_csv(f"datasets/{sys.argv[1]}"[:-4] + '_train.csv')
		test.to_csv(f"{sys.argv[1]}"[:-4] + '_test.csv')
	except Exception as e:
		print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
	main()