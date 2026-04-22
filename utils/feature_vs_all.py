import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
from textwrap import wrap


def excel_label(index):
	label = ''
	index += 1
	while index > 0:
		index, remainder = divmod(index - 1, 26)
		label = chr(ord('A') + remainder) + label
	return label


def feature_index_to_name(feature, features):
	requested = str(feature).strip().upper()
	lookup = {name.upper(): name for name in features}
	if requested not in lookup:
		raise ValueError(f"Feature '{feature}' not found. Available features: {', '.join(features)}")
	return lookup[requested]


def build_parser():
	parser = argparse.ArgumentParser(description='Plot one feature against all the others.')
	parser.add_argument('feature', help='Feature name to compare against all the others (A, B, ..., AA, AB, ...).')
	parser.add_argument('--data', default='datasets/data.csv', help='Path to the CSV dataset.')
	parser.add_argument('--output', default='graphs/feature_vs_all.png', help='Output image path.')
	return parser


def main():
	try:
		args = build_parser().parse_args()
		data = pd.read_csv(args.data, header=None, index_col=0)
		data.columns = ['Target'] + [excel_label(i) for i in range(data.shape[1] - 1)]
		raw_target = data['Target'].astype(str)
		data['Target'] = raw_target.map({'M': 1, 'B': 0})
		features = ['Target'] + [c for c in data.columns if c != 'Target']
		feature = feature_index_to_name(args.feature, features)

		color_map = {'M': 'r', 'B': 'b'}
		data['Colors'] = raw_target.map(color_map)

		other_features = [name for name in features if name != feature]
		n_plots = len(other_features)
		n_cols = max(1, min(6, math.ceil(math.sqrt(n_plots))))
		n_rows = math.ceil(n_plots / n_cols)
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows), squeeze=False)
		fig.suptitle(f'Feature {feature} against all others', fontsize=18)

		for i, other in enumerate(other_features):
			row, col = divmod(i, n_cols)
			ax = axes[row, col]
			ax.scatter(data[feature], data[other], c=data['Colors'], alpha=0.35, s=10)
			ax.set_xlabel('\n'.join(wrap(feature, 10)), fontsize=9)
			ax.set_ylabel('\n'.join(wrap(other, 10)), fontsize=9)
			ax.set_title(other, fontsize=10)
			ax.tick_params(labelsize=8)

		for i in range(n_plots, n_rows * n_cols):
			row, col = divmod(i, n_cols)
			axes[row, col].axis('off')

		handles = [
			plt.Line2D([0], [0], marker='o', linestyle='', color=color, label=target)
			for target, color in color_map.items()
		]
		fig.legend(handles=handles, loc='upper right')

		plt.tight_layout(rect=[0, 0, 0.98, 0.95])
		plt.savefig(args.output, dpi=150, bbox_inches='tight')
		plt.show()

	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == '__main__':
	main()