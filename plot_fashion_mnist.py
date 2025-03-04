import wandb
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="fashion-mnist-classification", name="data-exploration")

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create sample images for each class
sample_images = []
for c in range(10):
    class_indices = np.where(y_train == c)
    sample_index = class_indices[0][0]  # Get the first image for each class
    sample_images.append(x_train[sample_index])

# Simulate 50 steps for the "Index" bar
num_indices = 50  # The horizontal bar should go from 0 to 50

for index in range(num_indices):
    # Create a figure to display all 10 images in one grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i], cmap='gray')
        ax.set_title(class_names[i])
        ax.axis('off')
    plt.suptitle(f"Index {index} - Sample Fashion MNIST Images", fontsize=16)

    # Log the entire grid as a single image in WandB
    wandb.log({"fashion_mnist_grid": wandb.Image(fig), "Index": index})  # No step= here
    plt.close(fig)  # Close figure to prevent memory leaks

wandb.finish()
