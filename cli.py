import tensorflow as tf
import tensorflow_federated as tff
import random

# Global Variables
USERS_PER_ROUND = 15
comms_round = 300
clients_datasets_dict = {}  # Should be populated with client datasets
central_emnist_test = None  # Should be set with test data
NUM_CLIENTS = 100
epsilon = [1.0, 5.0, 15.0, 25.0, 35.0, 45.0]  # Placeholder for epsilon
c = [1.0, 5.0, 15.0, 25.0, 35.0, 45.0]  # Placeholder for constant c
clipping_norm = 1.0  # Placeholder for clipping norm


class KerasModel:
    def __init__(self):
        self.initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.model = self.create_keras_model()

    def create_keras_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=self.initializer),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def get_model(self):
        return self.model


class FederatedLearning:
    def __init__(self):
        self.keras_model = KerasModel().get_model()
        self.input_spec = (
            tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        )
        self.client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    def model_fn(self):
        return tff.learning.models.from_keras_model(
            self.keras_model,
            input_spec=self.input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    def client_update(self, model, dataset, server_weights):
        """Performs training (using the server model weights) on the client's dataset."""
        client_weights = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

        for batch in dataset:
            with tf.GradientTape() as tape:
                outputs = model(batch)
            grads = tape.gradient(outputs.loss, client_weights)
            grads_and_vars = zip(grads, client_weights)
            self.client_optimizer.apply_gradients(grads_and_vars)

        client_weights = tf.nest.map_structure(lambda x: x.assign(clip_weights(x, clipping_norm)), client_weights)
        sensitivity = 2 * clipping_norm / (len(dataset) / NUM_CLIENTS)
        stddev = (c * sensitivity) / epsilon
        for var in client_weights:
            stddev = tf.cast(stddev, var.dtype)
            noise = tf.random.normal(shape=tf.shape(var), mean=0.0, stddev=stddev)
            var.assign_add(noise)

        return client_weights

    def server_update(self, model, mean_client_weights):
        """Updates the server model weights as the average of the client model weights."""
        model_weights = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, mean_client_weights)

        stddev = get_global_stddev()  # Assume this is implemented elsewhere
        for var in model_weights:
            stddev = tf.cast(stddev, var.dtype)
            noise = tf.random.normal(shape=tf.shape(var), mean=0.0, stddev=stddev)
            var.assign_add(noise)

        return model_weights


class FederatedComputation:
    def __init__(self):
        self.federated_learning = FederatedLearning()

    def initialize_fn(self):
        model = self.federated_learning.model_fn()
        return model.trainable_variables

    @tff.tf_computation
    def server_init(self):
        return self.federated_learning.initialize_fn()

    @tff.federated_computation
    def initialize(self):
        return tff.federated_value(self.server_init(), tff.SERVER)

    @tff.tf_computation
    def client_update_fn(self, tf_dataset, server_weights):
        model = self.federated_learning.model_fn()
        return self.federated_learning.client_update(model, tf_dataset, server_weights)

    @tff.tf_computation
    def server_update_fn(self, mean_client_weights):
        model = self.federated_learning.model_fn()
        return self.federated_learning.server_update(model, mean_client_weights)

    @tff.federated_computation
    def next_fn(self, server_weights, federated_dataset):
        server_weights_at_client = tff.federated_broadcast(server_weights)
        client_weights = tff.federated_map(self.client_update_fn, (federated_dataset, server_weights_at_client))
        mean_client_weights = tff.federated_mean(client_weights)
        server_weights = tff.federated_map(self.server_update_fn, mean_client_weights)
        return server_weights


def evaluate(server_state):
    keras_model = KerasModel().get_model()
    keras_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    keras_model.set_weights(server_state)
    history = keras_model.evaluate(central_emnist_test)
    return history


# Main Federated Learning Loop
def train_federated_algorithm():
    federated_computation = FederatedComputation()
    federated_algorithm = tff.templates.IterativeProcess(
        initialize_fn=federated_computation.initialize,
        next_fn=federated_computation.next_fn
    )

    server_state = federated_algorithm.initialize()
    history = {'loss': [], 'accuracy': []}
    clients_ids = [i for i in range(50)]

    for round in range(comms_round):
        current_epoch_num = round + 1
        print(f'Round: {current_epoch_num}')
        selected_clients = random.sample(clients_ids, USERS_PER_ROUND)
        current_federated_train_data = [clients_datasets_dict[client_id] for client_id in selected_clients]
        server_state = federated_algorithm.next(server_state, current_federated_train_data)
        loss, accuracy = evaluate(server_state)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)

    return history

# Running the training process
history = train_federated_algorithm()
