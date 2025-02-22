import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import wandb

# Initialize WandB
wandb.init(project="fashion-mnist-visualization", name="sample-images")

# Load Fashion-MNIST dataset
fashion_mnist = datasets.FashionMNIST(root="./data", train=True, download=True)
images, labels = fashion_mnist.data.numpy(), fashion_mnist.targets.numpy()

# Class names in Fashion-MNIST dataset
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Log 5 batches (each with 7 images)
for index in range(5):
    fig, axes = plt.subplots(1, 7, figsize=(12, 3))
    for i in range(7):
        img_idx = index * 7 + i
        axes[i].imshow(images[img_idx], cmap="gray")
        axes[i].set_title(class_names[labels[img_idx]])
        axes[i].axis("off")
    
    # Log the batch of images to WandB
    wandb.log({f"Batch {index}": wandb.Image(fig)})
    plt.close(fig)

wandb.finish()
