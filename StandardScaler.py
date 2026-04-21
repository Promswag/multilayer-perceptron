import pandas as pd

class StandardScaler():
	def __init__(self):
		self.features = []
		self.mean = {}
		self.std = {}
		self.zero_variance_features = []

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
		self.zero_variance_features = []
		for f in self.features:
			lst = df[f].dropna()
			if len(lst) == 0:
				self.mean[f] = 0.0
				self.std[f] = 0.0
				self.zero_variance_features.append(f)
				continue

			self.mean[f] = sum(lst) / len(lst)
			if len(lst) < 2:
				self.std[f] = 0.0
			else:
				self.std[f] = (sum(abs(lst - self.mean[f]) ** 2) / (len(lst) - 1)) ** 0.5

			if self.std[f] == 0 or pd.isna(self.std[f]):
				self.std[f] = 0.0
				self.zero_variance_features.append(f)

		if len(self.zero_variance_features) > 0:
			print(f"Warning: zero-variance features detected ({len(self.zero_variance_features)}): {list(self.zero_variance_features)}")

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		if len(self.features) == 0:
			return df
		for f in self.features:
			std = self.std.get(f, 0)
			if std == 0 or pd.isna(std):
				# Keep zero-variance features neutral after scaling.
				df.loc[:, f] = 0
				continue

			scaled = (df.loc[:, f] - self.mean[f]) / std
			scaled = scaled.replace([float('inf'), float('-inf')], 0).fillna(0)
			df.loc[:, f] = scaled
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

