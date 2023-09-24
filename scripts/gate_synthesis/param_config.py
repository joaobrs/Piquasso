#
# Copyright 2021-2023 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

## Some quality of life configuartions depending of the platform on which the main script is executed
sys.path.append("/home/kolarovszki.zoltan/piquasso/")
sys.path.append("/home/vhenrik/PyProjects/piquasso/")
tf.debugging.enable_traceback_filtering()
tf.get_logger().setLevel('INFO')
np.set_printoptions(suppress=True, linewidth=200, threshold=sys.maxsize)

## Overall hyperparamteres ##
is_on_cluster = False
number_of_modes = 1
cutoff = 12
dtype = np.complex128

## Unitary ##
degree = 3 # of Hamiltonian
gate_cutoff = 4 # Size of matrices to be approximated
seed = None # For debugging purposes

# Random generator #
number_of_unitaries = 7500 # To be generated

# Datacube #
lower_coefficient_bound = 0.2 # Of linspace
upper_coefficient_bound = 0.8 # Of linspace
number_of_inner_points = 3 # Of linspace
coefficient_increment = (upper_coefficient_bound - lower_coefficient_bound)/number_of_inner_points # If needed

## CVNN Parameters ##
isProfiling = False # Debugging purposes
number_of_datapacks = 1 # shape: (amount of unitaries, (amount_of_unitaries, number_of_coefficients))
number_of_cvnn_layers = 10 # To be trained
number_of_layer_parameters = 7 # How many parameters are needed in a single layer (different for more than 1 mode)
number_of_cvnn_steps = 4000 # Training steps
cvnn_tolerance = 0.008 # Threshold under which training should be stopped for the given unitary
cvnn_learning_rate = 0.05 # Optimizer hyperparam
cvnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cvnn_learning_rate) # Adam worked fastest in terms of convergence
passive_sd = 0.1 # For random generation
active_sd = 0.001 # For random generation

## Classical neural network ##
number_of_nn_epochs = 5000 # Training steps
nn_batch_size = 8 # Batching, for different platforms
nn_learning_rate = 0.01 # Optimizer hyperparam
nn_optimizer = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate) # Adam worked fastest in terms of convergence
nn_loss = "mse" # If not custom is used
nn_validation_split = 0.6
load_model = True

## Persistence ##
cvnn_path = "scripts/gate_synthesis/cvnn_approximations/gc4hd3/" # Place to load and save CVNN data
nn_path = "scripts/gate_synthesis/neural_network_checkpoints/" # Place to load and save NN checkpoints
plot_path = "scripts/gate_synthesis/neural_network_checkpoints/plots/" # Place to save the histories of the learning
# Dictionaries to save for reproduction purposes
general_cvnn_info = {
    "cutoff": cutoff,
    "mode_amount": number_of_modes,
    "gate_cutoff": gate_cutoff,
    "number_of_layers": number_of_cvnn_layers,
    "number_of_steps": number_of_cvnn_steps,
    "number_of_unitaries": number_of_unitaries,
    "learnin_rate": cvnn_learning_rate,
    "active_sd": active_sd,
    "passive_sd": passive_sd,
    "seed": seed,
    "hamiltonian_degree": degree,
    "optimizer": cvnn_optimizer.__str__(),
    "tolerance": cvnn_tolerance,
}
general_nn_info = {
    "number_of_steps": number_of_nn_epochs,
    "learnin_rate": nn_learning_rate,
    "optimizer": nn_optimizer.__str__(),
}