# Imports
import tensorflow as tf
import numpy as np

# Inputs
fahrenheit = np.array([-40, -14, 32, 46, 59, 72, 100], dtype=float)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# Define the layers
hidden1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
hidden2 = tf.keras.layers.Dense(units = 3)
output = tf.keras.layers.Dense(units = 1)

# Define the model
model2 = tf.keras.Sequential([hidden1, hidden2, output])

# Compile the model
model2.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

# Train the model
print("Starting training...")
history2 = model2.fit(fahrenheit, celsius, epochs = 1000, verbose = False)
print("Model trained!")

# Test the model
result = model2.predict([120])
print("The result is " + str(result) + " celsius!")