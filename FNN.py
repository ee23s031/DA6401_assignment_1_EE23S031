import numpy as np
import wandb
from keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="fashion-mnist-nn")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize input data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# One-hot encoding of labels
def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Initialize neural network parameters
def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters

# Activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Forward propagation
def forward_propagation(X, parameters, hidden_activation=sigmoid):
    cache = {"A0": X.T}
    L = len(parameters) // 2
    for l in range(1, L):
        cache[f"Z{l}"] = np.dot(parameters[f"W{l}"], cache[f"A{l-1}"]) + parameters[f"b{l}"]
        cache[f"A{l}"] = hidden_activation(cache[f"Z{l}"])
    cache[f"Z{L}"] = np.dot(parameters[f"W{L}"], cache[f"A{L-1}"]) + parameters[f"b{L}"]
    cache[f"A{L}"] = softmax(cache[f"Z{L}"])
    return cache

# Compute loss
def compute_loss(Y, A):
    m = Y.shape[0]
    return -np.sum(Y.T * np.log(A + 1e-9)) / m

# Backpropagation
def backward_propagation(Y, cache, parameters, hidden_activation_derivative):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[0]
    dZ = cache[f"A{L}"] - Y.T
    for l in reversed(range(1, L + 1)):
        grads[f"dW{l}"] = np.dot(dZ, cache[f"A{l-1}"].T) / m
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
        if l > 1:
            dZ = np.dot(parameters[f"W{l}"].T, dZ) * hidden_activation_derivative(cache[f"Z{l-1}"])
    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    for key in parameters.keys():
        parameters[key] -= learning_rate * grads[f"d{key}"]
    return parameters

# Sigmoid derivative
def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

# Training loop
def train_nn(X_train, Y_train, layer_dims, epochs=50, learning_rate=0.1, hidden_activation=sigmoid, hidden_activation_derivative=sigmoid_derivative):
    parameters = initialize_parameters(layer_dims)
    for epoch in range(epochs):
        cache = forward_propagation(X_train, parameters, hidden_activation)
        loss = compute_loss(Y_train, cache[f"A{len(layer_dims)-1}"])
        grads = backward_propagation(Y_train, cache, parameters, hidden_activation_derivative)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Log loss to wandb
        wandb.log({"Epoch": epoch, "Loss": loss})
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    return parameters

# Set up and train the network
layer_dims = [784, 128, 64, 10]  # Example: 2 hidden layers with 128 and 64 neurons
trained_parameters = train_nn(x_train, y_train, layer_dims, epochs=50, learning_rate=0.1)
