import matplotlib.pyplot as plt
import pandas as pd


def excel_label(index):
	label = ''
	index += 1
	while index > 0:
		index, remainder = divmod(index - 1, 26)
		label = chr(ord('A') + remainder) + label
	return label


def main():
	try:
		data = pd.read_csv('datasets/data.csv', header=None, index_col=0)
		data.columns = ['Target'] + [excel_label(i) for i in range(data.shape[1] - 1)]
		data['Target'] = data['Target'].map({'M': 1, 'B': 0})
		corr_cols = ['Target'] + [c for c in data.columns if c != 'Target']
		corr = data[corr_cols].corr()
		features = corr.columns.tolist()

		fig, ax = plt.subplots(figsize=(14, 12))
		im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

		ax.set_xticks(range(len(features)))
		ax.set_yticks(range(len(features)))
		ax.set_xticklabels(features, rotation=90, fontsize=8)
		ax.set_yticklabels(features, fontsize=8)
		ax.set_title('Correlation heatmap (Target + features)', fontsize=18)

		for i in range(len(features)):
			for j in range(len(features)):
				value = corr.iloc[i, j]
				color = 'white' if abs(value) > 0.5 else 'black'
				ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6, color=color)

		plt.tight_layout()
		plt.savefig('graphs/feature_heatmap.png', dpi=150, bbox_inches='tight')
		plt.show()

	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == '__main__':
	main()
