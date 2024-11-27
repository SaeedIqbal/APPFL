import numpy as np
from typing import List
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Global variables for reusability
K = 5  # Number of clients
rho = 0.1  # Trimming factor
p = 2  # Lp-norm (Euclidean distance)
key_size = 2048  # RSA key size

class Client:
    def __init__(self, id: int, data_size: int, model_weights: np.ndarray):
        self.id = id
        self.data_size = data_size
        self.model_weights = model_weights
        self.public_key, self.private_key = self.generate_keys()

    def generate_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        public_key = private_key.public_key()
        return public_key, private_key

    def encrypt_model(self, model_weights: np.ndarray):
        # Homomorphic encryption of model weights using RSA
        encrypted_model = self.public_key.encrypt(
            model_weights.tobytes(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_model

    def decrypt_model(self, encrypted_model: bytes):
        # Decrypting the model using the private key
        decrypted_model = self.private_key.decrypt(
            encrypted_model,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return np.frombuffer(decrypted_model, dtype=np.float64)  # assuming model weights are in float64

    def compute_gradient_similarity(self, global_gradient: np.ndarray):
        # Cosine similarity between client gradient and global gradient
        similarity = np.dot(self.model_weights, global_gradient) / (np.linalg.norm(self.model_weights) * np.linalg.norm(global_gradient))
        return similarity

    def compute_loss_reduction(self, previous_loss: float, current_loss: float):
        # Loss reduction = previous loss - current loss
        return previous_loss - current_loss

    def trim_model(self, models: List[np.ndarray], trimming_factor: float):
        # Trimming function to select top models based on Lp-norm
        num_models_to_select = int(len(models) * (1 - trimming_factor))
        trimmed_models = sorted(models, key=lambda m: np.linalg.norm(m))[:num_models_to_select]
        return np.mean(trimmed_models, axis=0)


class SecureAggregation:
    def __init__(self, clients: List[Client], rho: float, p: int):
        self.clients = clients
        self.rho = rho
        self.p = p

    def aggregate(self):
        encrypted_models = []
        for client in self.clients:
            encrypted_models.append(client.encrypt_model(client.model_weights))

        # Step 1: Secure Aggregation via Homomorphic Encryption
        aggregated_encrypted_model = np.sum(encrypted_models, axis=0)  # Aggregating encrypted models

        # Decrypting to compute global model
        global_model = self.clients[0].decrypt_model(aggregated_encrypted_model)

        # Step 2: Robust Aggregation via Trimmed Mean
        models = [client.model_weights for client in self.clients]
        robust_model = self.clients[0].trim_model(models, self.rho)

        # Step 3: Weighted Aggregation
        total_data_size = sum(client.data_size for client in self.clients)
        weights = [client.data_size / total_data_size for client in self.clients]

        # Calculate similarity and loss reduction, and refine weights
        for i, client in enumerate(self.clients):
            gradient_similarity = client.compute_gradient_similarity(global_model)
            loss_reduction = client.compute_loss_reduction(0.5, 0.4)  # Just example values, you would calculate actual loss
            weights[i] = (client.data_size * gradient_similarity * loss_reduction) / (total_data_size * sum(weights))

        # Step 4: Combined Secure, Robust, and Weighted Aggregation
        final_weighted_model = np.sum([weight * robust_model for weight in weights], axis=0)
        final_encrypted_model = np.sum([client.encrypt_model(final_weighted_model) for client in self.clients], axis=0)

        # Decrypt the final aggregated model
        final_model = self.clients[0].decrypt_model(final_encrypted_model)
        return final_model


# Example usage:
# Initialize clients with random model weights and data sizes
clients = [Client(id=i, data_size=np.random.randint(100, 1000), model_weights=np.random.randn(10)) for i in range(K)]

# Initialize Secure Aggregation instance
aggregation = SecureAggregation(clients=clients, rho=rho, p=p)

# Perform secure and robust aggregation
final_model = aggregation.aggregate()
print("Final Aggregated Model Weights:", final_model)
