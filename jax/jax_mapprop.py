import jax.numpy as jnp
from util import *
import jax.random as jrandom
from dataclasses import dataclass, replace
from util_jax import *
from typing import Optional, List

L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4
LS_REAL = [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID]

ACT_F = {
    L_SOFTPLUS: softplus,
    L_RELU: relu,
    L_SIGMOID: sigmoid,
    L_LINEAR: lambda x: x,
}

ACT_D_F = {
    L_SOFTPLUS: sigmoid,
    L_RELU: relu_d,
    L_SIGMOID: sigmoid_d,
    L_LINEAR: lambda x: 1,
}


@dataclass(frozen=True)
class EqPropLayer:
    name: str
    input_size: int
    output_size: int
    optimizer: any  # e.g. an instance of jax_adam_optimizer or jax_simple_grad_optimizer
    l_type: int
    temp: float

    _w: jnp.ndarray   # shape: (input_size, output_size)
    _b: jnp.ndarray   # shape: (output_size,)
    _inv_var: jnp.ndarray  # shape: (output_size,)

    inputs: Optional[jnp.ndarray] = None   # latest inputs (e.g. shape (batch, input_size))
    pot: Optional[jnp.ndarray] = None      # potential (pre-activation)
    mean: Optional[jnp.ndarray] = None     # post-activation mean (or softmax probabilities)
    values: Optional[jnp.ndarray] = None   # current sampled value
    new_values: Optional[jnp.ndarray] = None  # new values (after an update step)
    w_trace: Optional[jnp.ndarray] = None  # shape: (batch, input_size, output_size)
    b_trace: Optional[jnp.ndarray] = None  # shape: (batch, output_size)

def init_eq_prop_layer(key, name, input_size, output_size, optimizer, var, temp, l_type):
    if l_type not in [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID, L_DISCRETE]:
        raise Exception(f"l_type ({l_type}) not implemented")
    temp = temp if l_type == L_DISCRETE else 1.0
    lim = jnp.sqrt(2.0 / (input_size + output_size))
    key, subkey = jrandom.split(key)
    _w = jrandom.uniform(subkey, shape=(input_size, output_size), minval=-lim, maxval=lim)
    _b = jnp.zeros((output_size,))
    _inv_var = jnp.full((output_size,), 1/var)
    values = jnp.zeros((1, output_size))
    new_values = jnp.zeros((1, output_size))
    w_trace = jnp.zeros((1, input_size, output_size))
    b_trace = jnp.zeros((1, output_size))
    return EqPropLayer(name, input_size, output_size, optimizer, l_type, temp,
                       _w, _b, _inv_var,
                       inputs=None, pot=None, mean=None,
                       values=values, new_values=new_values,
                       w_trace=w_trace, b_trace=b_trace), key

@jit
def compute_pot_mean(layer: EqPropLayer, inputs: jnp.ndarray):
    """
    Given inputs, compute the pre-activation (pot) and the mean.
    Also return an updated layer that “remembers” the inputs, pot, and mean.
    """
    pot = jnp.dot(inputs, layer._w) + layer._b
    if layer.l_type in LS_REAL:
        mean = ACT_F[layer.l_type](pot)
    else:
        mean = jax_softmax(pot / layer.temp, axis=-1)
    return pot, mean

@jit
def sample_layer(key, layer: EqPropLayer, inputs: jnp.ndarray):
    """
    Compute the potential and mean, then sample new values from the layer.
    Return an updated layer and a new key.
    """
    pot, mean = compute_pot_mean(layer, inputs)
    if layer.l_type in LS_REAL:
        sigma = jnp.sqrt(1.0 / layer._inv_var)
        key, subkey = jrandom.split(key)
        values = mean + sigma * jrandom.normal(subkey, shape=pot.shape)
    else:  # discrete
        key, subkey = jrandom.split(key)
        values = jax_multinomial_rvs(subkey, n=1, p=mean)
    new_layer = replace(layer, inputs=inputs, pot=pot, mean=mean, values=values)
    return new_layer, key

@jit
def refresh_layer(layer: EqPropLayer, freeze_value: bool):
    """
    Recompute the layer’s pot and mean using the stored inputs.
    If freeze_value is False, update the current values from new_values.
    """
    if layer.inputs is None:
        raise ValueError("Layer inputs are not set. Run a forward pass first.")
    pot, mean = compute_pot_mean(layer, layer.inputs)
    values = layer.values if freeze_value else layer.new_values
    new_layer = replace(layer, pot=pot, mean=mean, values=values)
    return new_layer

@jit
def update_layer(key, layer: EqPropLayer, update_size: float, next_layer: Optional[EqPropLayer] = None):
    """
    Perform one update step (MAP gradient ascent) on the layer.
    If next_layer is provided, use feedback from it.
    Return the updated layer and a new key.
    """
    if next_layer is None:
        if layer.l_type in LS_REAL:
            sigma = jnp.sqrt(1.0 / layer._inv_var)
            key, subkey = jrandom.split(key)
            new_values = layer.mean + sigma * jrandom.normal(subkey, shape=layer.pot.shape)
        else:
            key, subkey = jrandom.split(key)
            new_values = jax_multinomial_rvs(subkey, n=1, p=layer.mean)
    else:
        if layer.l_type in LS_REAL:
            lower_pot = (layer.mean - layer.values) * layer._inv_var
            if next_layer.l_type in LS_REAL:
                upper_term = (next_layer.values - next_layer.mean)
                upper_term = upper_term * ACT_D_F[next_layer.l_type](next_layer.pot) * next_layer._inv_var
                fb_w = jnp.transpose(next_layer._w)
                upper_pot = jnp.dot(upper_term, fb_w)
            else:
                fb_w = jnp.transpose(next_layer._w)
                upper_pot = jnp.dot((next_layer.values - next_layer.mean), fb_w) / next_layer.temp
            update_pot = lower_pot + upper_pot
            new_values = layer.values + update_size * update_pot
        else:
            key, subkey = jrandom.split(key)
            new_values = jax_multinomial_rvs(subkey, n=1, p=layer.mean)
    new_layer = replace(layer, new_values=new_values)
    return new_layer, key

@jit
def record_trace_layer(layer: EqPropLayer, gate: Optional[jnp.ndarray] = None, lambda_: float = 0.0):
    """
    Record the change in activations as traces.
    (Here we assume that layer.inputs is set from a previous forward pass.)
    """
    if layer.inputs is None or layer.pot is None or layer.mean is None:
        raise ValueError("Layer state is incomplete. Make sure to run sample/compute before trace.")
    if layer.l_type in LS_REAL:
        v_ch = (layer.values - layer.mean) * ACT_D_F[layer.l_type](layer.pot) * layer._inv_var
    else:
        v_ch = (layer.values - layer.mean) / layer.temp
    if gate is not None:
        v_ch = v_ch * gate[:, None]
    new_w_trace = lambda_ * layer.w_trace + jnp.einsum('bi,bj->bij', layer.inputs, v_ch)
    new_b_trace = lambda_ * layer.b_trace + v_ch
    new_layer = replace(layer, w_trace=new_w_trace, b_trace=new_b_trace)
    return new_layer

def learn_trace_layer(layer: EqPropLayer, reward: jnp.ndarray, lr: float = 0.01):
    """
    Use the recorded traces and a reward signal to update the layer’s weights.
    (Note: Here we assume that the optimizer has a pure method `delta`.)
    """
    w_update = jnp.mean(layer.w_trace * reward[:, None, None], axis=0)
    b_update = jnp.mean(layer.b_trace * reward[:, None], axis=0)
    delta_w = layer.optimizer.delta(grads=[w_update], name=layer.name+"_w", learning_rate=lr)[0]
    delta_b = layer.optimizer.delta(grads=[b_update], name=layer.name+"_b", learning_rate=lr)[0]
    new_w = layer._w + delta_w
    new_b = layer._b + delta_b
    new_layer = replace(layer, _w=new_w, _b=new_b)
    return new_layer

@jit
def clear_trace_layer(layer: EqPropLayer, mask: jnp.ndarray):
    """
    Clear part of the traces according to mask.
    """
    new_w_trace = layer.w_trace * mask.astype(jnp.float64)[:, None, None]
    new_b_trace = layer.b_trace * mask.astype(jnp.float64)[:, None]
    new_layer = replace(layer, w_trace=new_w_trace, b_trace=new_b_trace)
    return new_layer

@jit
def clear_values_layer(layer: EqPropLayer, mask: jnp.ndarray):
    """
    Clear (zero out) values for those batch entries indicated by mask.
    """
    new_values = layer.values * mask.astype(layer.values.dtype)[:, None]
    new_new_values = layer.new_values * mask.astype(layer.new_values.dtype)[:, None]
    new_layer = replace(layer, values=new_values, new_values=new_new_values)
    return new_layer

@dataclass(frozen=True)
class Network:
    layers: List[EqPropLayer]

def init_network(key, state_n, action_n, hidden: List[int],
                 var, temp, hidden_l_type, output_l_type,
                 optimizer):
    """
    Initialize a network with one input layer, several hidden layers, and one output layer.
    The layers are created sequentially.
    """
    layers = []
    in_size = state_n
    k = key
    for d, n in enumerate(hidden + [action_n]):
        l_type = output_l_type if d == len(hidden) else hidden_l_type
        layer, k = init_eq_prop_layer(k, name=f"layer_{d}",
                                      input_size=in_size, output_size=n,
                                      optimizer=optimizer, var=var, temp=temp,
                                      l_type=l_type)
        layers.append(layer)
        in_size = n
    return Network(layers), k

@jit
def forward_network(key, net: Network, state: jnp.ndarray):
    """
    Forward pass through the network.
    Each layer is sampled (using its sample_layer function).
    The key is threaded through the layers.
    """
    x = state
    new_layers = []
    k = key
    for layer in net.layers:
        layer, k = sample_layer(k, layer, x)
        x = layer.values
        new_layers.append(layer)
    return Network(new_layers), x, k

def map_grad_ascent(key, net: Network, steps: int, update_size: float):
    """
    Perform several update steps on all layers (except possibly the output).
    For simplicity, we update layers in order and use the following layer (if any)
    for feedback.
    """
    k = key
    layers = net.layers
    for i in range(steps):
        new_layers = []
        for idx, layer in enumerate(layers):
            next_layer = layers[idx+1] if idx+1 < len(layers) else None
            layer, k = update_layer(k, layer, update_size, next_layer)
            layer = refresh_layer(layer, freeze_value=(idx==len(layers)-1))
            new_layers.append(layer)
        layers = new_layers
    return Network(layers), k

def learn_network(net: Network, reward: jnp.ndarray, lr: float = 0.01):
    """
    Apply the learn_trace function to each layer.
    """
    new_layers = []
    for layer in net.layers:
        new_layer = learn_trace_layer(layer, reward, lr)
        new_layers.append(new_layer)
    return Network(new_layers)

def clear_traces_network(net: Network, mask: jnp.ndarray):
    new_layers = [clear_trace_layer(layer, mask) for layer in net.layers]
    return Network(new_layers)

def clear_values_network(net: Network, mask: jnp.ndarray):
    new_layers = [clear_values_layer(layer, mask) for layer in net.layers]
    return Network(new_layers)
