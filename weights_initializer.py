import numpy as np

def he_normal(n_inputs, n_outputs):
	std_dev = (2 / n_inputs) ** 0.5
	return np.random.normal(0, std_dev, (n_outputs, n_inputs))

def he_uniform(n_inputs, n_outputs):
	limit = (6 / n_inputs) ** 0.5
	return np.random.uniform(-limit, limit, (n_outputs, n_inputs))

def glorot_normal(n_inputs, n_outputs):
	std_dev = (2 / (n_inputs + n_outputs)) ** 0.5
	return np.random.normal(0, std_dev, (n_outputs, n_inputs))

def glorot_uniform(n_inputs, n_outputs):
	limit = (6 / (n_inputs + n_outputs)) ** 0.5
	return np.random.uniform(-limit, limit, (n_outputs, n_inputs))

def lecun_normal(n_inputs, n_outputs):
	std_dev = (1 / n_inputs) ** 0.5
	return np.random.normal(0, std_dev, (n_outputs, n_inputs))

def lecun_uniform(n_inputs, n_outputs):
	limit = (3 / n_inputs) ** 0.5
	return np.random.uniform(-limit, limit, (n_outputs, n_inputs))