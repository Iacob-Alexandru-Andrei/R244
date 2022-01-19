# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Flower Quickstart (Simulation with TensorFlow/Keras)
# 
# Welcome to Flower, a friendly federated learning framework!
# 
# In this notebook, we'll simulate a federated learning system with 100 clients. The clients will use TensorFlow/Keras to define model training and evaluation. Let's start by installing Flower Nightly, published as `flwr-nightly` on PyPI:

# %%
# !pip install git+https://github.com/adap/flower.git@release/0.17#egg=flwr["simulation"]  # For a specific branch (release/0.17) w/ extra ("simulation")
# # !pip install -U flwr["simulation"]  # Once 0.17.1 is released

# %% [markdown]
# Next, we import the required dependencies. The most important imports are Flower (`flwr`) and TensorFlow:

# %%
import os

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import tensorflow.compat.v1 as tf
import flwr as fl

from CBO_Flower_Simulation.simulation import start_simulation

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from shared.shared import*

from sklearn.model_selection import train_test_split


# %% [markdown]
# With that out of the way, let's move on to the interesting bits. Federated learning systems consist of a server and multiple clients. In Flower, we create clients by implementing subclasses of `flwr.client.Client` or `flwr.client.NumPyClient`. We use `NumPyClient` in this tutorial because it is easier to implement and requires us to write less boilerplate.
# 
# To implement the Flower client, we create a subclass of `flwr.client.NumPyClient` and implement the three methods `get_parameters`, `fit`, and `evaluate`:
# 
# - `get_parameters`: Return the current local model parameters
# - `fit`: Receive model parameters from the server, train the model parameters on the local data, and return the (updated) model parameters to the server 
# - `evaluate`: Received model parameters from the server, evaluate the model parameters on the local data, and return the evaluation result to the server
# 
# We mentioned that our clients will use TensorFlow/Keras for the model training and evaluation. Keras models provide methods that make the implementation staightforward: we can update the local model with server-provides parameters through `model.set_weights`, we can train/evaluate the model through `fit/evaluate`, and we can get the updated model parameters through `model.get_weights`.
# 
# Let's see a simple implementation:

# %%
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test) -> None:
        # self.model = model
        self.X_train, self.Y_train = x_train, y_train
        self.X_test, self.Y_test = x_test, y_test
    def get_parameters(self):
        return np.zeros(1)

    def fit(self, parameters, config):
        # self.model.set_weights(parameters)
        # self.model.fit(self.x_train, self.y_train, epochs=1, verbose=2)
        return np.zeros(1), 0, {}

    def evaluate(self, parameters, config):
        learning_rate = torch.exp(config['h0']).item()
        learning_rate_decay = torch.exp(config['h1']).item()
        l2_regular = torch.exp(config['h2']).item()
        s = int(config['seed'])

        config = tf.ConfigProto()
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        tf.set_random_seed(s)
        np.random.seed(s)
    
        epochs = 20
        batch_size = 32
        
        # build the CNN model using Keras
        model = get_compiled_original_CBO_CNN(self.X_train.shape[1:], learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, l2_regular=l2_regular)
        #
        history = model.fit(self.X_train, self.Y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(self.X_test, self.Y_test),
                      shuffle=True, verbose=0)
        val_acc = max(history.history['val_acc'])  
      
        return 0.0, s, {"accuracy": val_acc}

# %% [markdown]
# Our class `FlowerClient` defines how local training/evaluation will be performed and allows Flower to call the local training/evaluation through `fit` and `evaluate`. Each instance of `FlowerClient` represents a *single client* in our federated learning system. Federated learning systems have multiple clients (otherwise there's not much to federate, is there?), so each client will be represented by its own instance of `FlowerClient`. If we have, for example, three clients in our workload, we'd have three instances of `FlowerClient`. Flower calls `FlowerClient.fit` on the respective instance when the server selects a particular client for training (and `FlowerClient.evaluate` for evaluation).
# 
# In this notebook, we want to simulate a federated learning system with 100 clients on a single machine. This means that the server and all 100 clients will live on a single machine and share resources such as CPU, GPU, and memory. Having 100 clients would mean having 100 instances of `FlowerClient` im memory. Doing this on a single machine can quickly exhaust the available memory resources, even if only a subset of these clients participates in a single round of federated learning.
# 
# In addition to the regular capabilities where server and clients run on multiple machines, Flower therefore provides special simulation capabilities that create `FlowerClient` instances only when they are actually necessary for training or evaluation. To enable the Flower framework to create clients when necessary, we need to implement a function called `client_fn` that creates a `FlowerClient` instance on demand. Flower calls `client_fn` whenever it needs an instance of one particular client to call `fit` or `evaluate` (those instances are usually discarded after use). Clients are identified by a client ID, or short `cid`. The `cid` can be used, for example, to load different local data partitions for each client:

# %%
def get_client_fn(data_generator = load_partitioned_FEMINST_data):
    X_trains, X_tests, Y_trains, Y_tests = data_generator()
    def client_fn(cid: str) -> fl.client.Client:
        nonlocal X_trains
        nonlocal Y_trains
        nonlocal X_tests
        nonlocal Y_tests
        s = int(cid) % 10 
        return FlowerClient(0, X_trains[s], Y_trains[s], X_tests[s], Y_tests[s])
    return client_fn


# %%
client_fn = get_client_fn()
NUM_CLIENTS=10


client_fn("123").evaluate(None,{
    "h0":torch.Tensor([np.log(0.001)]),
    "h1":torch.Tensor([np.log(0.001)]),
    "h2":torch.Tensor([np.log(0.01)]), 
    "seed":1,
})

# %% [markdown]
# We now have `FlowerClient` which defines client-side training and evaluation and `client_fn` which allows Flower to create `FlowerClient` instances whenever it needs to call `fit` or `evaluate` on one particular client. The last step is to start the actual simulation using `flwr.simulation.start_simulation`. 
# 
# The function `start_simulation` accepts a number of arguments, amongst them the `client_fn` used to create `FlowerClient` instances, the number of clients to simulate `num_clients`, the number of rounds `num_rounds`, and the strategy. The strategy encapsulates the federated learning approach/algorithm, for example, *Federated Averaging* (FedAvg).
# 
# Flower comes with a number of built-in strategies, but we can also use our own strategy implementations to customize nearly all aspects of the federated learning approach. For this example, we use the built-in `FedAvg` implementation and customize it using a few basic parameters. The last step is the actual call to `start_simulation` which - you guessed it - actually starts the simulation.

# %%
client_fn = get_client_fn()
# Create FedAvg strategy
strategy=fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 10% of available clients for training
        fraction_eval=1.0,  # Sample 5% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
        min_eval_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
        min_available_clients=NUM_CLIENTS,  # Wait until at least 75 clients are available
)

# Start simulation
start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    num_rounds=1,
    strategy=strategy,
)

# %% [markdown]
# Congratulations! With that, you built a Flower client, customized it's instantiation through the `client_fn`, customized the server-side execution through a `FedAvg` strategy configured for this workload, and started a simulation with 100 clients (each holding their own individual partition of the MNIST dataset).
# 
# Next, you can continue to explore more advanced Flower topics:
# 
# - Deploy server and clients on different machines using `start_server` and `start_client`
# - Customize the server-side execution through custom strategies
# - Customize the client-side exectution through `config` dictionaries

