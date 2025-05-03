import jax
from jax.lib import xla_bridge
from jax import numpy as jnp


def f(x):
    return jnp.log(x == 1)


args = [jnp.arange(2), jnp.ones(2)]
print(f"running on {xla_bridge.get_backend().platform} ...")
fns = {"f": f, "jitted_f": jax.jit(f)}
for key, fn in fns.items():
    for x in args:
        print(f"{key}({x}) -> ...", end=" ", flush=True)
        print(fn(x), flush=True)
print("done")
