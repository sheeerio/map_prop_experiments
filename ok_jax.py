from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as jrandom
import numpy as np
import gymnax
import argparse
import configparser
import os
import sys
import json
import matplotlib.pyplot as plt
from jax import tree_util
from functools import partial
from dataclasses import dataclass, replace
from typing import Optional, List
from jax.nn import relu, softplus, sigmoid, softmax  # etc.
import optax

@jit
def jax_relu(x) -> jnp.ndarray:
    return relu(x)

@jit
def jax_relu_grad(x):
    # Note: For many purposes you can compute the derivative automatically,
    # but here we keep your threshold logic.
    return jnp.where(x > 0, 1, 0)

@jit
def jax_sigmoid(x):
    return sigmoid(x)

@jit
def jax_sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

@jit
def jax_softplus(x):
    return softplus(x)

# @jit
def jax_softmax(X, theta=1.0, axis=None):
    y = jnp.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y *= float(theta)
    y -= jnp.expand_dims(jnp.max(y, axis=axis), axis)
    y = jnp.where(y<-30, jnp.exp(-30), jnp.exp(y))
    ax_sum = jnp.expand_dims(jnp.sum(y, axis=axis), axis)
    p = y/ax_sum
    if len(X.shape) == 1 : p = jnp.ravel(p)
    return p

@jit
def jax_multinomial_rvs(key, n, p):
    key, subkey = jrandom.split(key)
    count = jnp.full(p.shape[:-1], n)
    out = jnp.zeros(p.shape, dtype=int)
    ps_prev = jnp.concatenate(
        [jnp.zeros(p.shape[:-1] + (1,)), jnp.cumsum(p, axis=-1)[..., :-1]],
        axis=-1,
    )
    condp = p / (1.0 - ps_prev)
    condp = jnp.where(jnp.isnan(condp), 0.0, condp)
    def body_fn(i, carry):
        key, count, out = carry
        key, subkey = jrandom.split(key)
        binsample = jrandom.binomial(subkey, count, condp[..., i])
        out = out.at[..., i].set(binsample)
        count = count - binsample
        return key, count, out
    key, count, out = jax.lax.fori_loop(0, p.shape[-1] - 1, body_fn, (key, count, out))
    out = out.at[..., -1].set(count)
    return out

@jit
def jax_from_one_hot(y):
    return jnp.argmax(jnp.array(y), axis=-1)

# @jit
def jax_to_one_hot(a, size):
    oh = jnp.zeros((a.shape[0], size), dtype=int)
    return oh.at[jnp.arange(a.shape[0]), a.astype(int)].set(1)

# @jit
def jax_getl(x, n):
    return x[n] if type(x) == list else x

@jit
def jax_equal_zero(x):
    return jnp.logical_and(jnp.where(x > -1e8, 1, 0),
    jnp.where(x < 1e-8, 1, 0))

@jit
def jax_mask_neg(x):
    return (x < 0).astype(jnp.float32)

@jit
def jax_apply_mask(x, mask):
    return (x.T * mask).T

@jit
def jax_sign(x):
    return (x > 1e-8).astype(jnp.float32) - (x < -1e-8).astype(jnp.float32)

@jit
def jax_neg_to_zero(x):
    return jnp.where(x > 1e-8, 1.0, 0.0)

@jit
def jax_zero_to_neg(x):
    return jnp.where(x > 1e-8, 1.0, -1.0)


@jit
def jax_linear_interpolat(start, end, end_t, cur_t):
    if type(start) == list:
        if type(end_t) == list:
            return [(e - s) * min(cur_t, d) / d + s for (s, e, d) in zip(start, end, end_t)]
        else:    
            return [(e - s) * min(cur_t, end_t)  / end_t + s for (s, e) in zip(start, end)]
    else:
        if type(end_t) == list:
            return [(end - start) * min(cur_t, d) / d + start for d in end_t]
        else:          
            return (end - start) * min(cur_t, end_t) / end_t + start

class jax_simple_grad_optimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @jit
    def delta(self, name, grads, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        return [learning_rate * i for i in grads]

# @jit
# def compute_delta(g, m, v, beta_1, beta_2, t, learning_rate, epsilon):
#     new_m = beta_1 * m + (1 - beta_1) * g
#     new_v = beta_2 * v + (1 - beta_2) * jnp.power(g, 2)
#     m_hat = new_m / (1 - jnp.power(beta_1, t))
#     v_hat = new_v / (1 - jnp.power(beta_2, t))
#     delta = jnp.array([lr * m_hat / (jnp.sqrt(v_hat) + epsilon) for lr in learning_rate])
#     return delta, new_m, new_v

# class jax_adam_optimizer():
#     def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.learning_rate = learning_rate
#         self.epsilon = epsilon
#         self._cache = {}    
    
#     def delta(self, grads, name="w", learning_rate=None, gate=None):
#         if name not in self._cache:
#             self._cache[name] = [
#                 [jnp.zeros_like(i) for i in grads],
#                 [jnp.zeros_like(i) for i in grads],
#                 0
#             ]
#         self._cache[name][2] += 1
#         t = self._cache[name][2]
#         deltas = []
#         beta_1, beta_2 = self.beta_1, self.beta_2
#         learning_rate = self.learning_rate if learning_rate is None else learning_rate
        
#         for n, g in enumerate(grads):
#             m = self._cache[name][0][n]
#             v = self._cache[name][1][n]
#             delta_n, new_m, new_v = compute_delta(
#                 g, m, v, beta_1, beta_2, t, learning_rate, self.epsilon
#             )
#             self._cache[name][0][n] = new_m
#             self._cache[name][1][n] = new_v
#             deltas.append(delta_n)
        
#         return deltas

class MDP(ABC): 
  def __init__(self):
    super().__init__()        

  @abstractmethod
  def reset(self, batch_size):
    pass

  @abstractmethod
  def act(self, actions):
    pass

def reset_env(key, batch_size, addr_size, action_size, x_size, zero):
    new_key, subkey = jrandom.split(key)
    x = jrandom.binomial(subkey, n=1, p=0.5, shape=(batch_size, x_size))
    
    factors = 2 ** (addr_size - 1 - jnp.arange(addr_size))
    reshaped = x[:, : addr_size * action_size].reshape(batch_size, action_size, addr_size)
    addr = jnp.sum(reshaped * factors, axis=2)  # shape: (batch_size, action_size)
    
    col_indices = (addr_size * action_size + addr).astype(jnp.int32) # shape: (batch_size, action_size)
    y = jnp.take_along_axis(x, col_indices, axis=1)
    
    new_x = jax_zero_to_neg(x) if not zero else x
    new_y = jax_zero_to_neg(y) if not zero else y
    return new_key, new_x, new_y

@partial(jit, static_argnames=("reward_zero",))
def act_env(y, actions, reward_zero):
    corr = (y == actions.astype(jnp.int32)).astype(jnp.float32)
    return corr if reward_zero else jax_zero_to_neg(corr)


@partial(jit, static_argnames=("zero", "reward_zero"))
def expected_reward_env(x, y, p, zero, reward_zero):
    """
    Pure jitted function to compute expected reward.
    p: an array of probabilities, shape (batch_size, 2)
    """
    batch_size = x.shape[0]
    init_val = 0 if reward_zero else -1
    reward_f = jnp.full((batch_size, 2), init_val)
    y_use = y if zero else jax_neg_to_zero(y)
    reward_f = reward_f.at[jnp.arange(batch_size), y_use[:, 0].astype(jnp.int32)].set(1)
    return jnp.sum(reward_f * p, axis=-1)

# --- Environment class with a similar API ---

class jax_complex_multiplexer_MDP:
    def __init__(self, addr_size=2, action_size=2, zero=True, reward_zero=True):
        self.addr_size = addr_size
        self.action_size = action_size      
        self.x_size = addr_size * action_size + 2 ** addr_size
        self.zero = zero
        self.reward_zero = reward_zero
        self.key = jrandom.PRNGKey(0)
        self.x = None
        self.y = None

    def reset(self, key, batch_size):
        # Make sure batch_size and self.x_size are Python ints:
        batch_size = int(batch_size)
        x_size = int(self.x_size)
        self.key, self.x, self.y = reset_env(key, batch_size,
                                            self.addr_size,
                                            self.action_size,
                                            x_size,
                                            self.zero)
        return self.x

    def act(self, actions):
        """
        Compute rewards given actions by calling the jitted act_env function.
        """
        return act_env(self.y, actions, self.reward_zero)

    def expected_reward(self, p):
        """
        Compute the expected reward for a probability distribution over actions
        by calling the jitted expected_reward_env function.
        """
        return expected_reward_env(self.x, self.y, p, self.zero, self.reward_zero)


class jax_reg_MDP(MDP):
    def __init__(self, x_size=8, layers=2, load_file="reg.npy", clean=False):
        self.x_size = x_size
        self.action_size = 1
        self.layers = layers
        if os.path.exists(load_file) and not clean:
            ws = np.load(load_file, allow_pickle=True).item()
            self._ws = {n: jnp.array(ws[n]) for n in ws}
        else:
            self._ws = {}
            for n in range(layers - 1):
                self._ws[n] = jnp.array(np.random.normal(size=(x_size, x_size)))
            self._ws[layers - 1] = jnp.array(np.random.normal(size=(x_size, 1)))
            np.save(load_file, {n: np.array(self._ws[n]) for n in self._ws})
        self.key = jrandom.PRNGKey(0)
        self.x = None
        self.y = None
        super().__init__()
    
    @jit
    def reset(self, key, batch_size):
        """
        Reset the regression MDP.
          key: a JAX PRNGKey
          batch_size: number of independent samples
        Returns:
          x: the generated input (batch_size, x_size)
        """
        self.key = key
        self.key, subkey = jrandom.split(self.key)
        self.x = jrandom.normal(subkey, shape=(batch_size, self.x_size))
        self.y = self.x
        for n in range(self.layers):
            self.y = jnp.dot(self.y, self._ws[n])
            self.y = jax_relu(self.y)
        self.y = self.y[:, 0]
        return self.x

    def act(self, actions):
        """
        Compute the (negative squared error) reward given actions.
        """
        return -(actions - self.y) ** 2

@tree_util.register_pytree_node_class
@dataclass
class EnvWrapperState:
    obs: jnp.ndarray
    env_state: any
    reward: jnp.ndarray
    is_end: jnp.ndarray
    truncated_end: jnp.ndarray
    rest: jnp.ndarray
    warm: jnp.ndarray
    state_code: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.obs,
            self.env_state,
            self.reward,
            self.is_end,
            self.truncated_end,
            self.rest,
            self.warm,
            self.state_code,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@jit
def batch_reset(key, env, env_params, batch_size):
    """Reset a batch of gymnax environments and initialize counters."""
    keys = jrandom.split(key, batch_size)
    v_reset = jax.vmap(env.reset, in_axes=(0, None))
    obs, env_state = v_reset(keys, env_params)
    reward = jnp.zeros(batch_size)
    is_end = jnp.ones(batch_size, dtype=bool)
    truncated_end = jnp.zeros(batch_size, dtype=bool)
    rest = jnp.zeros(batch_size)
    warm = jnp.zeros(batch_size)
    state_code = jnp.zeros(batch_size, dtype=jnp.int32)
    return EnvWrapperState(obs, env_state, reward, is_end, truncated_end, rest, warm, state_code)

@jit
def env_wrapper_step(key, state: EnvWrapperState, actions, env, env_params, rest_n, warm_n):
    batch_size = state.obs.shape[0]

    new_rest = state.rest + state.is_end.astype(jnp.int32)
    new_warm = state.warm + 1

    reset_mask = new_rest > rest_n

    def maybe_reset(do_reset, key, obs, env_state):
        return jax.lax.cond(
            do_reset,
            lambda _: env.reset(key, env_params),  # if True: perform reset
            lambda _: (obs, env_state),             # if False: return current values
            operand=None
        )
    v_maybe_reset = jax.vmap(maybe_reset, in_axes=(0, 0, 0, 0))
    keys_reset = jrandom.split(key, batch_size)
    new_obs, new_env_state = v_maybe_reset(reset_mask, keys_reset, state.obs, state.env_state)
    live_mask = jnp.logical_and(new_warm > warm_n, jnp.logical_not(state.is_end))

    def maybe_step(live, key, env_state, action, obs):
        return jax.lax.cond(
            live,
            lambda _: env.step(key, env_state, action, env_params),
            lambda _: (obs, env_state, 0.0, jnp.array(False), {'discount': jnp.array(1.0)}),
            operand=None
        )
    v_maybe_step = jax.vmap(maybe_step, in_axes=(0, 0, 0, 0, 0))
    keys_step = jrandom.split(key, batch_size)
    step_out = v_maybe_step(live_mask, keys_step, new_env_state, actions, new_obs)
    next_obs, next_env_state, rewards, dones, infos = step_out

    batch_size = state.obs.shape[0]
    truncated_end = jnp.full((batch_size,), False)
    new_rest = jnp.where(truncated_end, rest_n, new_rest)
    state_code = jnp.where(live_mask, 0, 0)  # start from zeros
    state_code = jnp.where(new_rest >= 1, 1, state_code)
    state_code = jnp.where(new_warm <= warm_n, 3, state_code)
    state_code = jnp.where(new_warm == 0, 2, state_code)

    new_state = EnvWrapperState(
        obs=next_obs,
        env_state=next_env_state,
        reward=rewards,
        is_end=dones,
        truncated_end=truncated_end,
        rest=new_rest,
        warm=new_warm,
        state_code=state_code
    )

    info = {'state_code': state_code, 'truncated_end': truncated_end}
    return new_state, (next_obs, rewards, dones, info)

def plot(curves, names, mv_n=100, end_n=1000, xlabel="Episodes", ylabel="Running Average Return", ylim=None, loc=4, save=True, save_dir="./result/plots/", filename="plot.png"):  
    plt.figure(figsize=(10, 7), dpi=150) 
    colors = ['red', 'blue', 'green', 'crimson','orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    
    for i, m in enumerate(names.keys()):    
        # Apply moving average
        v = np.array([mv(ep[:end_n], mv_n) for ep in curves[m][0]])
        v = np.mean(v, axis=0)
        r_std = np.std(v, axis=0) / np.sqrt(len(curves[m][0]))
        v = np.concatenate([np.full([mv_n-1,], np.nan), v])
        
        k = names[m]
        ax = plt.gca()         
        ax.plot(np.arange(len(v)), v, label=k, color=colors[i % len(colors)])
        ax.fill_between(np.arange(len(v)), v - r_std, v + r_std, label=None, alpha=0.2, color=colors[i % len(colors)])
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)      
    if ylim is not None: 
        plt.ylim(ylim)
    plt.legend(loc=loc, fontsize=12)
    # plt.title(save_dir[-4:])
    plt.title("third layer var")
    plt.tight_layout()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

# Print the statistic of episode return

def print_stat(curves, names):
  for m in names.keys():
    print("Stat. on %s:" %m)
    r = np.average(np.array(curves[m][0]), axis=1)
    print("Return: avg. %.2f median %.2f min %.2f max %.2f std %.2f" % (np.average(r),
                                                                        np.median(r),
                                                                        np.amin(r),
                                                                        np.amax(r),
                                                                        np.std(r)))#/np.sqrt(len(r)))) 

L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4
LS_REAL = [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID]

ACT_F = {
    L_SOFTPLUS: jax_softplus,
    L_RELU: jax_relu,
    L_SIGMOID: jax_sigmoid,
    L_LINEAR: lambda x: x,
}

ACT_D_F = {
    L_SOFTPLUS: jax_sigmoid,
    L_RELU: jax_relu_grad,
    L_SIGMOID: jax_sigmoid_grad,
    L_LINEAR: lambda x: 1,
}


@dataclass
class EqPropLayer:
    name: str
    input_size: int
    output_size: int
    optimizer: optax.GradientTransformation  # Optax optimizer instance
    l_type: int
    temp: float
    _w: jnp.ndarray   # Weight matrix of shape (input_size, output_size)
    _b: jnp.ndarray   # Bias vector of shape (output_size,)
    _inv_var: jnp.ndarray  # Inverse variance vector, shape (output_size,)
    inputs: Optional[jnp.ndarray] = None   # Latest inputs
    pot: Optional[jnp.ndarray] = None        # Pre-activation (potential)
    mean: Optional[jnp.ndarray] = None       # Post-activation mean (or softmax probabilities)
    values: Optional[jnp.ndarray] = None     # Current sampled value
    new_values: Optional[jnp.ndarray] = None # New values (after an update step)
    w_trace: Optional[jnp.ndarray] = None      # Trace for weights, shape (batch, input_size, output_size)
    b_trace: Optional[jnp.ndarray] = None      # Trace for biases, shape (batch, output_size)
    opt_state: Optional[optax.OptState] = None   # Optimizer state for this layer

def init_eq_prop_layer(key, name, input_size, output_size, lr, var, temp, l_type):
    # Compute the initialization limit for weights (e.g., He initialization).
    lim = jnp.sqrt(2.0 / (input_size + output_size))
    key, subkey = jrandom.split(key)
    # Initialize the weight matrix uniformly in [-lim, lim].
    _w = jrandom.uniform(subkey, shape=(input_size, output_size), minval=-lim, maxval=lim)
    # Initialize the biases to zeros.
    _b = jnp.zeros((output_size,))
    # Set the inverse variance for this layer.
    _inv_var = jnp.full((output_size,), 1/var)
    
    # Initialize additional fields.
    # Start values and new_values as zero vectors (shape: (1, output_size)).
    values = jnp.zeros((1, output_size))
    new_values = jnp.zeros((1, output_size))
    # Initialize weight and bias traces (for learning updates).
    w_trace = jnp.zeros((1, input_size, output_size))
    b_trace = jnp.zeros((1, output_size))
    
    # Create an Optax Adam optimizer instance with the specified learning rate.
    # Wrap lr in float() to ensure it is a Python scalar.
    optimizer = optax.adam(learning_rate=float(lr), b1=0.9, b2=0.999)
    # Initialize the optimizer state for the parameters we want to update.
    params = {"_w": _w, "_b": _b}
    opt_state = optimizer.init(params)
    
    # Create and return the EqPropLayer dataclass instance.
    layer = EqPropLayer(
        name=name,
        input_size=input_size,
        output_size=output_size,
        optimizer=optimizer,
        l_type=l_type,
        temp=temp,
        _w=_w,
        _b=_b,
        _inv_var=_inv_var,
        inputs=None,
        pot=None,
        mean=None,
        values=values,
        new_values=new_values,
        w_trace=w_trace,
        b_trace=b_trace,
        opt_state=opt_state
    )
    return layer, key



def learn_trace_layer(layer: EqPropLayer, reward: jnp.ndarray) -> EqPropLayer:
    # Compute gradient estimates from recorded traces:
    w_grad = jnp.mean(layer.w_trace * reward[:, None, None], axis=0)
    b_grad = jnp.mean(layer.b_trace * reward[:, None], axis=0)
    grads = {"_w": w_grad, "_b": b_grad}
    params = {"_w": layer._w, "_b": layer._b}
    updates, new_opt_state = layer.optimizer.update(grads, layer.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return replace(layer, _w=new_params["_w"], _b=new_params["_b"], opt_state=new_opt_state)

# @jit
def compute_pot_mean(layer: EqPropLayer, inputs: jnp.ndarray):
    pot = jnp.dot(inputs, layer._w) + layer._b
    if layer.l_type in LS_REAL:
        mean = ACT_F[layer.l_type](pot)
    else:
        mean = jax_softmax(pot / layer.temp, axis=-1)
    return pot, mean

# @jit
def sample_layer(key, layer: EqPropLayer, inputs: jnp.ndarray):
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

# @jit
def refresh_layer(layer: EqPropLayer, freeze_value: bool):
    if layer.inputs is None:
        raise ValueError("Layer inputs are not set. Run a forward pass first.")
    pot, mean = compute_pot_mean(layer, layer.inputs)
    values = layer.values if freeze_value else layer.new_values
    new_layer = replace(layer, pot=pot, mean=mean, values=values)
    return new_layer

# @jit
def update_layer(key, layer: EqPropLayer, update_size: float, next_layer: Optional[EqPropLayer] = None):
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
                 layer_lrs: List[float]):
    """
    Initialize a network with one input layer, several hidden layers, and one output layer.
    """
    layers = []
    in_size = state_n
    k = key
    for d, n in enumerate(hidden + [action_n]):
        l_type = output_l_type if d == len(hidden) else hidden_l_type
        layer, k = init_eq_prop_layer(key=k, name=f"layer_{d}",
                                      input_size=in_size, output_size=n,
                                      lr=layer_lrs[d], var=var[d], temp=temp,
                                      l_type=l_type)
        layers.append(layer)
        in_size = n
    return Network(layers=tuple(layers)), k

# forward_network, map_grad_ascent, learn_network, etc.
# In learn_network, call learn_trace_layer on each layer:
def learn_network(net: Network, reward: jnp.ndarray, lr: float = 0.01):
    new_layers = []
    for layer in net.layers:
        new_layer = learn_trace_layer(layer, reward)
        new_layers.append(new_layer)
    return Network(new_layers)


# @partial(jax.jit, static_argnames=("net",))
def forward_network(key, net: Network, state: jnp.ndarray):
    x = state
    new_layers = []
    k = key
    for layer in net.layers:
        layer, k = sample_layer(k, layer, x)
        x = layer.values
        new_layers.append(layer)
    return Network(new_layers), x, k

def map_grad_ascent(key, net: Network, steps: int, update_size: float):
    k = key
    layers = net.layers
    for i in range(steps):
        new_layers = []
        for idx, layer in enumerate(layers):
            next_layer = layers[idx+1] if idx+1 < len(layers) else None
            layer, k = update_layer(k, layer, update_size[idx], next_layer)
            layer = refresh_layer(layer, freeze_value=(idx==len(layers)-1))
            new_layers.append(layer)
        layers = new_layers
    return Network(layers), k


def clear_traces_network(net: Network, mask: jnp.ndarray):
    new_layers = [clear_trace_layer(layer, mask) for layer in net.layers]
    return Network(new_layers)

def clear_values_network(net: Network, mask: jnp.ndarray):
    new_layers = [clear_values_layer(layer, mask) for layer in net.layers]
    return Network(new_layers)


def main():
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config",default="config_mp.ini",
    help="Location of config file (default: config_mp.ini)")
    args, _ = initial_parser.parse_known_args()

    config_dir = "config"
    f_name = os.path.join(config_dir, args.config)
    print(f"Loading config from {f_name}")

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    if not config.read(f_name):
        print(f"Error: Config file '{f_name}' not found or is invalid.")
        sys.exit(1)

    # Step 2: Create a new parser with defaults from config
    parser = argparse.ArgumentParser(
        parents=[initial_parser],
        description="Script with configurable parameters via config file and command-line flags."
    )

    # General parameters
    parser.add_argument("--name",default=config.get("DEFAULT", "name"),
        help="Name identifier for the run.")
    parser.add_argument("--exp_num",type=int,default=1,
        help="Experiment number to help with tracking")
    parser.add_argument("--max_eps",type=int,default=config.getint("DEFAULT", "max_eps"),
        help="Number of episodes per run.")
    parser.add_argument("--n_run",type=int,default=config.getint("DEFAULT", "n_run"),
        help="Number of runs.")
    # Task parameters
    parser.add_argument("--env_name",default=config.get("DEFAULT", "env_name"),
        choices=["Multiplexer", "Regression"],
        help="Environment name (e.g., Multiplexer, Regression).")
    parser.add_argument("--batch_size",type=int,default=config.getint("DEFAULT", "batch_size"),
        help="Batch size.")
    parser.add_argument("--hidden",type=str,default=config.get("DEFAULT", "hidden"),
        help="JSON list of hidden units per layer (e.g., '[64, 32]').")
    parser.add_argument("--l_type",type=int,choices=[0, 1, 2, 3, 4],
        default=config.getint("DEFAULT", "l_type"),
        help="Activation function type: 0=Softplus, 1=ReLU, 2=Linear, 3=Sigmoid, 4=Discrete.")
    parser.add_argument("--temp",type=float,default=config.getfloat("DEFAULT", "temp"),
        help="Temperature for the network if applicable.")
    parser.add_argument("--var",type=str,default=config.get("DEFAULT", "var"),
        help="JSON list of variances in hidden layers (e.g., '[0.3, 1, 1]').")
    parser.add_argument("--update_adj",type=float,default=config.getfloat("DEFAULT", "update_adj"),
        help="Step size for energy minimization adjustment.")
    parser.add_argument("--map_grad_ascent_steps",type=int,
        default=config.getint("DEFAULT", "map_grad_ascent_steps"),
        help="Number of gradient ascent steps for energy minimization.")
    parser.add_argument("--lr",type=str,default=config.get("DEFAULT", "lr"),
        help="JSON list of learning rates (e.g., '[0.04, 0.00004, 0.000004]').")
    parser.add_argument("--key",type=int,default=0,help="PRNG Key for generating keys and subkey(s).")
    # Parse all arguments
    args = parser.parse_args()
    
    try:
        hidden = json.loads(args.hidden)
        if not isinstance(hidden, list):
            raise ValueError
    except (json.JSONDecoderError, ValueError):
        print("Error: `hidden` not a valid JSON list")
        sys.exit(1)

    try:
        var = json.loads(args.var)
        if not isinstance(var, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        print("Error: 'var' must be a valid JSON")
        sys.exit(1)

    try:
        lr = json.loads(args.lr)
        if not isinstance(lr, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        print("Error: 'lr' must be a valid JSON list")
        sys.exit(1)
    
    L_SOFTPLUS = 0
    L_RELU = 1
    L_LINEAR = 2
    L_SIGMOID = 3
    L_DISCRETE = 4

    key = jax.random.PRNGKey(args.key)

    # Initialize environment (as before)
    if args.env_name == "Multiplexer":
        env = jax_complex_multiplexer_MDP(
            addr_size=5,
            action_size=1,
            zero=False,
            reward_zero=False
        )
        gate = False
        output_l_type = L_DISCRETE
        action_n = 2 ** env.action_size
    elif args.env_name == "Regression":
        env = jax_reg_MDP()
        gate = True
        output_l_type = L_LINEAR
        action_n = 1
    else:
        print(f"Error: Unsupported environment '{args.env_name}'.")
        sys.exit(1)
    
    update_size = [i * args.update_adj for i in var]
    print_every = 128 * 500
    # Use Optax’s Adam optimizer (instead of a custom optimizer)
    print(lr)

    eps_ret_hist_full = []
    for j in range(args.n_run):
        net, key = init_network(
            key=key,
            state_n=env.x_size,
            action_n=action_n,
            layer_lrs=lr,
            hidden=hidden,
            var=var,
            temp=args.temp,
            hidden_l_type=args.l_type,
            output_l_type=output_l_type
        )

        eps_ret_hist = []
        print_count = print_every
        for i in range(args.max_eps // args.batch_size):
            state = env.reset(key, args.batch_size)
            net, action, key = forward_network(key=key, net=net, state=state)
            if args.env_name == "Multiplexer":
                action = jax_zero_to_neg(jax_from_one_hot(action))[:, jnp.newaxis]
                reward = env.act(action)[:, 0]
            elif args.env_name == "Regression":
                action = action[:, 0]
                reward = env.act(action)
            eps_ret_hist.append(jnp.average(reward))
            map_grad_ascent(
                key,
                net=net,
                steps=args.map_grad_ascent_steps,
                update_size=update_size
            )

            if args.env_name == "Regression":
                reward = env.y - net.layers[-1].mean[:, 0]
            net = learn_network(net, reward, lr=lr)
            if (i * args.batch_size) > print_count:
                running_avg = jnp.average(jnp.array(eps_ret_hist[-print_every // args.batch_size:]))
                print(f"Run {j} Step {i} Running Avg. Reward\t{running_avg:.6f}")
                print_count += print_every
        eps_ret_hist_full.append(eps_ret_hist)

    eps_ret_hist_full = jnp.asarray(eps_ret_hist_full, dtype=float)
    print("Finished Training")

    curves = {}
    curves[args.name] = (eps_ret_hist_full,)
    names = {k: k for k in curves.keys()}

    plots_dir = os.path.join(f"result/exp{args.exp_num}", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    result_dir = f"result/exp{args.exp_num}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.name}.npy")
    print(f"Results (saved to {result_file}):")
    np.save(result_file, curves)
    
    # Optionally plot results:
    # plot(curves, names, mv_n=10, end_n=args.max_eps, save=True, save_dir=plots_dir, filename=f"{args.name}_plot.png")
        
if __name__ == "__main__":
    main()
