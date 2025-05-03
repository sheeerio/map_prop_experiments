import numpy as np
import jax.numpy as jnp
import jax
import time
from util_jax import *
from util import *

# Create a large array
size = 10_000  # 10 million elements
x_np = np.random.randn(size)
x_jax = jnp.array(x_np)  # Convert to JAX array
key = jax.random.PRNGKey(42)  # Random key for JAX


# Define a sample function (ReLU)
def relu_np(x):
    return np.where(x < 0, 0, x)


def relu_jax(x):
    return jnp.where(x < 0, 0, x)


@jax.jit
def relu_jax_jit(x):
    return jnp.where(x < 0, 0, x)


# Benchmark function
def benchmark(func, *args):
    start = time.time()
    result = func(*args)
    if isinstance(result, jnp.ndarray):  # Ensure JAX computations complete
        jax.block_until_ready(result)
    return time.time() - start


# Measure time
np_time = benchmark(to_one_hot, x_np, size)
jax_time = benchmark(jax_to_one_hot, x_jax, size)  # Without JIT
jax_jit_time = benchmark(jax_to_one_hot, x_jax, size)  # With JIT

print(f"NumPy (CPU) Time: {np_time:.6f} sec")
print(f"JAX (MPS, No JIT) Time: {jax_time:.6f} sec")
print(f"JAX (MPS, JIT) Time: {jax_jit_time:.6f} sec")

for i in range(20):
    jax_jit_time = benchmark(relu_jax_jit, x_jax)
    print(f"Run {i+1}: JAX (JIT) Time: {jax_jit_time:.6f} sec")
