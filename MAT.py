import matplotlib.pyplot as plt
import numpy as np
import activation_functions as af

def main():
	val = np.linspace(-10, 10, 200)
	plt.ylim(-5, 5)
	plt.grid(visible=True)
	# plt.plot(val, af.sigmoid(val))
	# plt.plot(val, 1 / (1 + np.exp(-(val - min(val)))), c='r', alpha=0.4)
	# plt.plot(val, 1 / (1 + np.exp(-(val + min(val)))), c='g', alpha=0.4)
	# plt.plot(val, 2 / (1 + np.exp(-(val - max(val)))), c='b', alpha=0.4)
	# plt.plot(val, 2 / (1 + np.exp(-(val + max(val)))), c='y', alpha=0.4)
	# plt.plot(val, np.exp(val), c='orange')
	# plt.plot(val, np.exp(-val), c='grey')

	plt.plot(val, np.exp(val + max(val)), c='turquoise')
	# plt.plot(val, np.exp(val - min(val)), c='blue')

	plt.plot(val, np.exp(val - max(val)), c='blue')
	# plt.plot(val, np.exp(val + min(val)), c='turquoise')

	plt.plot(val, np.exp(-val + max(val)), c='yellow')
	# plt.plot(val, np.exp(-val - min(val)), c='purple')

	plt.plot(val, np.exp(-val - max(val)), c='purple')
	# plt.plot(val, np.exp(-val + min(val)), c='yellow')
	plt.show()

if __name__ == "__main__":
	main()