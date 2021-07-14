# Imports
import tensorflow as tf
import numpy as np

# Inputs
fahrenheit = np.array([-40, -14, 32, 46, 59, 72, 100], dtype=float)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# Define the layer
layer = tf.keras.layers.Dense(units = 1, input_shape = [1])

# Define the model
model = tf.keras.Sequential([layer])

# Compile the model
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

# Train the model
print("Starting training...")
history = model.fit(fahrenheit, celsius, epochs = 1000, verbose = False)
print("Model trained!")

# Test the model
print("Let's do our first prediction!")
result = model.predict([120])
print("The result is " + str(result) + " celsius!")