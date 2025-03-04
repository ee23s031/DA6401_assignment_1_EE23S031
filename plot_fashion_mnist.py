# ===============
# Question - 1 
# ===============


import wandb
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

# Initialize wandb for the plotting 
wandb.init(project="fashion-mnist-classification", name="data-exploration")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sample_images = []

for i in range(10):
    class_indices = np.where(y_train == i)
    sample_index = class_indices[0][0] 
    sample_images.append(x_train[sample_index])

# For having a control over 50 set of images
num_indices = 50  # The bar can move from 0 to 50 ( can chnage here to adjust how much is needed)

for i in range(num_indices):
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i], cmap='gray')
        ax.set_title(class_names[i])
        ax.axis('off')
    plt.suptitle(f"Index {i} - Sample Fashion MNIST Images", fontsize=16)

    # Log the entire grid as a single image in WandB
    wandb.log({"fashion_mnist_grid": wandb.Image(fig), "Index": i}) 
    plt.close(fig)  

wandb.finish()
