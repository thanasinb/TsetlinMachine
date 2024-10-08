#!/usr/bin/python

import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import MultiClassTsetlinMachine
import vteam_params

# Parameters for the Tsetlin Machine
T = 15 
s = 3.9
number_of_clauses = 20
states = 100 

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_classes = 2

# Training configuration
epochs = 200

init_memristor_state = 0.5
voltage = 1.2

selected_params = vteam_params.get_vteam_params("Linear12")
alpha_off = selected_params["alpha_off"]
alpha_on = selected_params["alpha_on"]
v_off = selected_params["v_off"]
v_on = selected_params["v_on"]
k_off = selected_params["k_off"]
k_on = selected_params["k_on"]
d = selected_params["d"]
dt_off = d / (k_off * (((voltage / v_off) - 1) ** alpha_off))
dt_on = d / (k_on * (((-voltage / v_on) - 1) ** alpha_on))
dt = max(dt_off, -dt_on)/states

print(f"dt_off = {dt_off}")
print(f"dt_on = {dt_on}")
print(f"dt = {dt}\n")

# Loading of training and test data
training_data = np.loadtxt("Corrected_NoisyXORTrainingData.txt").astype(dtype=np.int32)
test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

X_training = training_data[:,0:12] # Input features
y_training = training_data[:,12] # Target value

X_test = test_data[:,0:12] # Input features
y_test = test_data[:,12] # Target value

# This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T,
                                                                    init_memristor_state,
                                                                    alpha_off, alpha_on, v_off, v_on,
                                                                    selected_params["r_off"],
                                                                    selected_params["r_on"],
                                                                    k_off, k_on, d, voltage, dt, dt)

# Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

# Some performance statistics

# print("Accuracy on test data (no noise):", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
# print("Accuracy on training data (40% noise):", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))
print("Accuracy on test data:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
print("Accuracy on training data:", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))
print()
print("Prediction: x1 = 1, x2 = 0, ... -> y = ",
      tsetlin_machine.predict(np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.int32)))
print("Prediction: x1 = 0, x2 = 1, ... -> y = ",
      tsetlin_machine.predict(np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.int32)))
print("Prediction: x1 = 0, x2 = 0, ... -> y = ",
      tsetlin_machine.predict(np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.int32)))
print("Prediction: x1 = 1, x2 = 1, ... -> y = ",
      tsetlin_machine.predict(np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=np.int32)))


