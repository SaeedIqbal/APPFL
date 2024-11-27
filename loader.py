import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image_dataset_from_directory
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

# --- Load MNIST and FashionMNIST dataset (using Keras for easy loading) ---
def load_mnist():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    return (train_images, train_labels), (test_images, test_labels)

def load_fashion_mnist():
    # Load FashionMNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    return (train_images, train_labels), (test_images, test_labels)

# --- Load MedicalMNIST dataset (using torch datasets and transforms) ---
def load_medical_mnist():
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    # Assuming dataset is available in a directory
    train_data = datasets.ImageFolder('path_to_medical_mnist/train', transform=transform)
    test_data = datasets.ImageFolder('path_to_medical_mnist/test', transform=transform)
    return train_data, test_data

# --- Load MVTec AD dataset ---
def load_mvtec_ad():
    # MVTec AD (Anomaly Detection Dataset) requires manual loading and image processing
    # Here we assume the dataset is available in a directory
    train_dataset = image_dataset_from_directory(
        'path_to_mvtec/train',
        image_size=(224, 224), batch_size=32, label_mode='int'
    )
    test_dataset = image_dataset_from_directory(
        'path_to_mvtec/test',
        image_size=(224, 224), batch_size=32, label_mode='int'
    )
    return train_dataset, test_dataset

# --- Federated Learning Setup ---
def create_federated_data(train_data, num_clients=10):
    """
    Creates federated data, simulating clients. Each client gets a partition of the dataset.
    """
    # Split the dataset into `num_clients` parts for federated learning.
    data_split = np.array_split(train_data, num_clients)
    
    federated_data = []
    for data in data_split:
        federated_data.append(tf.data.Dataset.from_tensor_slices(data))
    
    return federated_data

# --- Preprocessing data for federated learning ---
def preprocess_data_for_federated_learning(dataset_name):
    if dataset_name == 'mnist':
        (train_images, train_labels), _ = load_mnist()
        train_data = list(zip(train_images, train_labels))
        federated_data = create_federated_data(train_data)
    elif dataset_name == 'fashion_mnist':
        (train_images, train_labels), _ = load_fashion_mnist()
        train_data = list(zip(train_images, train_labels))
        federated_data = create_federated_data(train_data)
    elif dataset_name == 'medical_mnist':
        train_data, _ = load_medical_mnist()
        federated_data = create_federated_data(train_data, num_clients=10)
    elif dataset_name == 'mvtec_ad':
        train_data, _ = load_mvtec_ad()
        federated_data = create_federated_data(train_data, num_clients=10)
    else:
        raise ValueError("Unsupported dataset name: {}".format(dataset_name))
    
    return federated_data

# --- Define Federated Learning Model (Example using simple CNN) ---
def create_federated_model():
    # Simple CNN model for federated learning example
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for MNIST/FashionMNIST
    ])
    return model

# --- Training function for Federated Learning ---
def federated_training(federated_data, model):
    # Define federated learning model in TensorFlow Federated
    def model_fn():
        # Convert Keras model to TFF model
        return tff.learning.from_keras_model(model, 
                                             input_spec=tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
                                             loss=tf.keras.losses.SparseCategoricalCrossentropy())

    # Federated learning algorithm
    federated_averaging = tff.learning.build_federated_averaging_process(model_fn)
    federated_process = federated_averaging
    
    state = federated_process.initialize()
    
    for round_num in range(1, 11):  # Run 10 rounds of federated learning
        state, metrics = federated_process.next(state, federated_data)
        print(f"Round {round_num}, Metrics: {metrics}")
'''
# --- Main function ---
if __name__ == "__main__":
    # Choose a dataset to run federated learning
    dataset_name = 'mnist'  # Replace with 'fashion_mnist', 'medical_mnist', or 'mvtec_ad'
    
    federated_data = preprocess_data_for_federated_learning(dataset_name)
    
    model = create_federated_model()
    
    federated_training(federated_data, model)
'''