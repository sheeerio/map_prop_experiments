import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax


# --- helper function to pad an array to a target shape along its trailing dimensions
def pad_to(x, target_shape):
    # Assumes x.shape and target_shape have the same number of dimensions.
    pad_width = [(0, t - s) for s, t in zip(x.shape, target_shape)]
    return jnp.pad(x, pad_width, mode="constant")


# --- Compute maximum number of units across layers.
def compute_max_units(state_n, hidden, action_n):
    return max([state_n] + hidden + [action_n])


# --- Modified init_layer: note the extra parameter max_units.
def init_layer(key, input_size, output_size, var, temp, l_type, max_units):
    # For discrete layers, use the given temp; otherwise, force temp=1.
    temp_val = temp if l_type == L_DISCRETE else 1.0
    lim = jnp.sqrt(2.0 / (input_size + output_size))
    key, subkey = jrandom.split(key)
    # Create the “true” parameters...
    _w_real = jrandom.uniform(
        subkey, shape=(input_size, output_size), minval=-lim, maxval=lim
    )
    _b_real = jnp.zeros((output_size,))
    _inv_var_real = jnp.full((output_size,), 1 / var)
    # Pad them up to (max_units, max_units) or (max_units,) as needed.
    _w = pad_to(_w_real, (max_units, max_units))
    _b = pad_to(_b_real, (max_units,))
    _inv_var = pad_to(_inv_var_real, (max_units,))
    # For the dynamic arrays, we want them to have shape (1, max_units).
    values = pad_to(jnp.zeros((1, output_size)), (1, max_units))
    new_values = pad_to(jnp.zeros((1, output_size)), (1, max_units))
    # (For the traces, we similarly pad; adjust if needed.)
    w_trace = pad_to(jnp.zeros((1, input_size, output_size)), (1, max_units, max_units))
    b_trace = pad_to(jnp.zeros((1, output_size)), (1, max_units))
    layer = {
        "input_size": input_size,
        "output_size": output_size,
        "l_type": l_type,
        "temp": temp_val,
        "max_units": max_units,
        "_w": _w,
        "_b": _b,
        "_inv_var": _inv_var,
        # Dynamic state fields (initially None or zero-padded)
        "inputs": None,
        "pot": None,
        "mean": None,
        "values": values,
        "new_values": new_values,
        "w_trace": w_trace,
        "b_trace": b_trace,
    }
    return layer, key


# --- Modified init_network: build each layer using the padded sizes.
def init_network(
    key,
    state_n,
    action_n,
    hidden,
    var,
    temp,
    hidden_l_type,
    output_l_type,
    optimizer_factory,
):
    max_units = compute_max_units(state_n, hidden, action_n)
    layers = []
    optimizers = []  # non-jittable objects
    in_size = state_n
    for d, n in enumerate(hidden + [action_n]):
        l_type = output_l_type if d == len(hidden) else hidden_l_type
        layer, key = init_layer(key, in_size, n, var[d], temp, l_type, max_units)
        layers.append(layer)
        optimizers.append(optimizer_factory())
        in_size = n  # next layer’s input size is the current layer’s true output size
    network = {"layers": tuple(layers), "max_units": max_units}
    return network, optimizers, key


# --- Sample activation functions (unchanged, for example)
@jit
def jax_relu(x):
    return jnp.where(x < 0, 0, x)


@jit
def jax_softplus(x):
    return jnp.where(x > 30, x, jnp.log1p(jnp.exp(x)))


@jit
def jax_sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@jit
def jax_softmax(x, axis=-1):
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


# --- Compute mean using the appropriate activation.
def compute_mean(pot, l_type, temp):
    branches = [
        lambda x: jax_softplus(x),
        lambda x: jax_relu(x),
        lambda x: x,
        lambda x: jax_sigmoid(x),
        lambda x: jax_softmax(x / temp, axis=-1),
    ]
    return lax.switch(l_type, branches, pot)


# --- We now modify the forward function.
@jit
def forward_network(key, network, state):
    # We assume that 'state' is already padded to shape (batch_size, max_units)
    def layer_forward(carry, layer):
        x, key = carry
        # Grab the “true” dimensions for this layer.
        in_size = layer["input_size"]
        out_size = layer["output_size"]
        max_units = layer["max_units"]
        # For the dot product, slice x to the actual input size.
        x_actual = x[:, :in_size]
        _w = layer["_w"][:in_size, :out_size]
        _b = layer["_b"][:out_size]
        _inv_var = layer["_inv_var"][:out_size]
        # Compute pre-activation and mean.
        pot = jnp.dot(x_actual, _w) + _b  # shape: (batch_size, out_size)
        mean = compute_mean(pot, layer["l_type"], layer["temp"])
        key, subkey = jrandom.split(key)

        # Use the same operand structure in both branches.
        def sample_continuous(args):
            m, pot, inv_var, key = args
            sigma = jnp.sqrt(1.0 / inv_var)
            return m + sigma * jrandom.normal(key, shape=pot.shape)

        def sample_discrete(args):
            m, pot, inv_var, key = args
            return jax_multinomial_rvs(key, n=1, p=m)

        values = lax.cond(
            layer["l_type"] != L_DISCRETE,
            lambda args: sample_continuous(args),
            lambda args: sample_discrete(args),
            operand=(mean, pot, _inv_var, subkey),
        )
        # Now pad the computed arrays up to (batch_size, max_units)
        pot_padded = pad_to(pot, (pot.shape[0], max_units))
        mean_padded = pad_to(mean, (mean.shape[0], max_units))
        values_padded = pad_to(values, (values.shape[0], max_units))
        # (You could also update trace fields here if needed.)
        # Update the layer’s dynamic state.
        updated_layer = {
            **layer,
            "inputs": x,
            "pot": pot_padded,
            "mean": mean_padded,
            "values": values_padded,
        }
        # The carry for the next layer is the padded output.
        return (values_padded, key), updated_layer

    (final_x, key), new_layers = lax.scan(
        layer_forward, (state, key), network["layers"]
    )
    # Rebuild the network state with the updated (padded) dynamic fields.
    new_network = {**network, "layers": new_layers}
    return new_network, final_x, key


# --- Dummy multinomial sampler (unchanged)
@jit
def jax_multinomial_rvs(key, n, p):
    def sample_one(key, p_row):
        idx = jrandom.categorical(key, jnp.log(p_row))
        return jax.nn.one_hot(idx, p_row.shape[0])

    keys = jrandom.split(key, p.shape[0])
    return jax.vmap(sample_one)(keys, p)


# --- Global layer type constants.
L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4

# -----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # (For example purposes, assume some dummy values.)
    key = jrandom.PRNGKey(0)
    state_n = 8
    action_n = 1
    hidden = [64, 32]
    var = [0.3, 1, 1]
    temp = 1.0
    hidden_l_type = L_RELU
    output_l_type = L_LINEAR

    # Define a dummy optimizer factory.
    def optimizer_factory():
        class DummyOpt:
            def delta(self, grads, name, learning_rate=None):
                lr = 0.01 if learning_rate is None else learning_rate
                return [lr * g for g in grads]

        return DummyOpt()

    # Initialize network.
    network, optimizers, key = init_network(
        key,
        state_n,
        action_n,
        hidden,
        var,
        temp,
        hidden_l_type,
        output_l_type,
        optimizer_factory,
    )
    # Create a dummy input state.
    batch_size = 32
    # Note: We now need to pad the input state to (batch_size, max_units).
    max_units = network["max_units"]
    state_real = jrandom.normal(key, shape=(batch_size, state_n))
    state = pad_to(state_real, (batch_size, max_units))

    # Forward pass.
    network, final_x, key = forward_network(key, network, state)
    print("Final output shape:", final_x.shape)
