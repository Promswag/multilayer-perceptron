import pandas as pd

INPUT_CSV = "datasets/digits_valid.csv"
OUTPUT_CSV = "datasets/digits_valid_compact.csv"
TARGET_COLUMN = "target"


def main() -> None:
	df = pd.read_csv(INPUT_CSV, index_col=0)

	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {INPUT_CSV}.")

	compact = df.loc[:, [TARGET_COLUMN]]
	compact.to_csv(OUTPUT_CSV)
	print(f"Saved {OUTPUT_CSV} with index + '{TARGET_COLUMN}' only ({compact.shape[0]} rows).")


if __name__ == "__main__":
	main()
