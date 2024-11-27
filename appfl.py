import numpy as np

# Global variables
NUM_CLIENTS = 5  # Number of clients
T = 10  # Number of iterations
eta = 0.01  # Learning rate
lambda_1 = 0.1  # Regularization parameter for consistency
lambda_2 = 0.1  # Regularization parameter for client-specific stability
gamma = 0.5  # Scaling factor for aggregation

# Global Model Class for handling aggregation
class GlobalModel:
    def __init__(self, initial_model):
        """
        Initialize the global model with the initial model.
        
        Parameters:
        - initial_model: The initial global model to start federated learning.
        """
        self.model = initial_model
    
    def aggregate_models(self, client_updates, client_weights):
        """
        Aggregate client models based on computed weights.
        
        Parameters:
        - client_updates: List of updated models from clients.
        - client_weights: Corresponding weights for aggregation.
        
        Returns:
        - Aggregated global model.
        """
        weighted_sum = sum(w * model for w, model in zip(client_weights, client_updates))
        total_weight = sum(client_weights)
        return weighted_sum / total_weight

    def global_update(self, client_updates, client_losses):
        """
        Update the global model with aggregation and regularization.
        """
        # Compute weights for aggregation based on the client updates and losses
        client_weights = self.compute_aggregation_weights(client_losses)
        
        # Aggregate the client models
        self.model = self.aggregate_models(client_updates, client_weights)
    
    def compute_aggregation_weights(self, client_losses):
        """
        Compute aggregation weights for each client based on loss and dataset size.
        """
        total_loss = sum(client_losses)
        client_weights = [np.exp(-gamma * loss) for loss in client_losses]
        normalization_factor = sum(client_weights)
        return [w / normalization_factor for w in client_weights]


# Client Class for handling local updates and gradients
class Client:
    def __init__(self, client_id, dataset, global_model):
        """
        Initialize a client in the federated learning system.
        
        Parameters:
        - client_id: Unique identifier for the client
        - dataset: Client's dataset
        - global_model: The initial global model for the federation
        """
        self.client_id = client_id
        self.dataset = dataset
        self.model = global_model.copy()  # Make a copy of the global model
    
    def local_update(self, algorithm=1):
        """
        Perform local update of the model using gradient descent with regularization
        to minimize the loss function along with the regularization term.
        This can be customized to use different algorithms.
        """
        if algorithm == 1:
            return self.local_update_algorithm_1() % for anaPrivacyB.py
        elif algorithm == 2:
            return self.local_update_algorithm_2() % for sra.py
        elif algorithm == 3:
            return self.local_update_algorithm_3()% for PC.py
        else:
            raise ValueError("Invalid algorithm number.")
    
    def local_update_algorithm_1(self):
        """
        Local update following Algorithm 1 (APPFL). % for anaPrivacyB.py
        """
        loss = self.compute_loss(self.model)
        gradient = self.compute_gradient(self.model)
        
        # Regularization term: squared difference between global and local model
        regularization = lambda_1 * np.linalg.norm(self.model - self.global_model)
        
        # Local model update step using gradient descent with regularization
        self.model -= eta * (gradient + regularization)
        
        return self.model, loss
    
    def local_update_algorithm_2(self):
        """
        Local update following Algorithm 2 (alternative method). % for sra.py
        """
        # Implement alternative client update logic here (e.g., a different regularization strategy)
        gradient = self.compute_gradient(self.model)
        self.model -= eta * gradient  # Example of simple gradient descent update without regularization
        loss = self.compute_loss(self.model)
        return self.model, loss
    
    def local_update_algorithm_3(self):
        """
        Local update following Algorithm 3 (privacy-preserving method). % for PC.py
        """
        # Example: Add noise for privacy preservation
        gradient = self.compute_gradient(self.model)
        noise = np.random.normal(0, 0.1, size=gradient.shape)  # Add Gaussian noise
        self.model -= eta * (gradient + noise)
        loss = self.compute_loss(self.model)
        return self.model, loss
    
    def compute_loss(self, model):
        """
        Mock loss function, should be replaced with actual loss calculation.
        """
        # Example: L2 loss (placeholder)
        return np.sum((model - self.dataset)**2)
    
    def compute_gradient(self, model):
        """
        Mock gradient computation for the loss function.
        """
        # Example gradient (placeholder for actual loss gradient computation)
        return 2 * (model - self.dataset)


# Federated Learning System Class
class FederatedLearning:
    def __init__(self, num_clients, global_model, clients_data):
        """
        Initialize the Federated Learning framework.
        
        Parameters:
        - num_clients: Number of clients participating in federated learning.
        - global_model: The initial global model.
        - clients_data: Dataset for each client.
        """
        self.num_clients = num_clients
        self.global_model = global_model
        self.clients = [Client(i, data, global_model.model) for i, data in enumerate(clients_data)]
        self.global_model = GlobalModel(global_model)
    
    def train(self, num_iterations=T, algorithm=1):
        """
        Train the global model through federated learning using the specified algorithm.
        
        Parameters:
        - num_iterations: Number of iterations for federated learning.
        - algorithm: The algorithm to use for local updates (1, 2, or 3).
        """
        for t in range(num_iterations):
            client_updates = []
            client_losses = []
            
            # Perform local updates for each client
            for client in self.clients:
                updated_model, loss = client.local_update(algorithm)
                client_updates.append(updated_model)
                client_losses.append(loss)
            
            # Update global model
            self.global_model.global_update(client_updates, client_losses)
            
            print(f"Iteration {t+1}/{num_iterations}, Global Model Updated (Algorithm {algorithm})")
    
    def get_global_model(self):
        """
        Return the current global model.
        """
        return self.global_model.model


# Initialize a random global model and datasets for clients
initial_global_model = np.random.rand(100)  # Example: 100 features
clients_data = [np.random.rand(100) for _ in range(NUM_CLIENTS)]  # Example datasets for clients

# Create Federated Learning system
fl_system = FederatedLearning(NUM_CLIENTS, GlobalModel(initial_global_model), clients_data)

# Train the model through federated learning using Algorithm 1 (APPFL)
fl_system.train(algorithm=1)

# Get the final global model after training
final_global_model = fl_system.get_global_model()
print("Final Global Model (Algorithm 1):", final_global_model)

# Train the model through federated learning using Algorithm 2 (Alternative Method)
fl_system.train(algorithm=2)

# Get the final global model after training
final_global_model = fl_system.get_global_model()
print("Final Global Model (Algorithm 2):", final_global_model)

# Train the model through federated learning using Algorithm 3 (Privacy-Preserving Method)
fl_system.train(algorithm=3)

# Get the final global model after training
final_global_model = fl_system.get_global_model()
print("Final Global Model (Algorithm 3):", final_global_model)
