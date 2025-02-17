import pandas as pd

class StandardScaler():
	def __init__(self):
		self.features = []
		self.mean = {}
		self.std = {}

	def from_file(self, path: str = 'datasets/scaler.csv') -> "StandardScaler":
		try:
			df = pd.read_csv(path)
			self.features = df['Features'].values
			self.mean = {f: df['Mean'][i] for i, f in enumerate(self.features)}
			self.std = {f: df['Std'][i] for i, f in enumerate(self.features)}
		except Exception as e:
			print(f"{type(e).__name__} : {e}")
		return self

	def fit(self, df: pd.DataFrame):
		self.features = df.select_dtypes(include='number').columns.values
		for f in self.features:
			lst = df[f].dropna()
			self.mean[f] = sum(lst) / len(lst)
			self.std[f] = (sum(abs(lst - self.mean[f]) ** 2) / (len(lst) - 1)) ** 0.5

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		if len(self.features) == 0:
			return df
		for f in self.features:
			df.loc[:,f] = (df.loc[:,f] - self.mean[f]) / self.std[f]
		return df

	def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
		self.fit(df)
		return self.transform(df)
	
	def save_to_file(self, path: str = 'datasets/scaler.csv'):
		try:
			scaler = pd.DataFrame()
			scaler['Features'] = pd.Series(self.features)
			scaler['Mean'] = pd.Series([v for k, v in self.mean.items()])
			scaler['Std'] = pd.Series([v for k, v in self.std.items()])
			scaler.to_csv(path, index=False)
			print(f"Scaler have been saved in {path}")
		except Exception as e:
			print(f'{type(e).__name__}: {e}')

