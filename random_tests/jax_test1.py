import jax
import jax.numpy as jnp
from jax import grad, jit

# Define a simple neural network
def init_params():
    w = jax.random.normal(jax.random.PRNGKey(0), (10, 1))  # Input size: 10, Output size: 1
    return w

def forward(params, x):
    return jnp.dot(x, params)

# Example input
input_data = jax.random.normal(jax.random.PRNGKey(1), (5, 10))  # Batch size: 5
target_data = jax.random.normal(jax.random.PRNGKey(2), (5, 1))

# Initialize parameters
params = init_params()

# Forward pass
output = forward(params, input_data)
loss = jnp.mean((output - target_data) ** 2)  # MSE loss

print("input:", input_data)
print("Output:", output)
print("Loss:", loss)
