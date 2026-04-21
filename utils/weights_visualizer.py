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

if __name__ == "__main__":
	main()