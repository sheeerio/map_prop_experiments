import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from abc import ABC, abstractmethod
from jax import jit
import numpy as np
import jax.random as jrandom
from dataclasses import dataclass
import gymnax
import matplotlib.pyplot as plt
from jax import tree_util
import os


@jit
def jax_relu(x) -> jax.Array:
    assert isinstance(x, jax.Array) or isinstance(x, np.ndarray)
    return jnp.where(x < 0, 0, x)

@jit
def jax_relu_grad(x):
    assert isinstance(x, jax.Array) or isinstance(x, np.ndarray)
    return jnp.where(x < 0, 0, 1)

@jit
def jax_sigmoid(x):
    assert isinstance(x, jax.Array or isinstance(x, np.ndarray))
    lim = 20
    return jnp.where(x >= lim, 1, 
        jnp.where(x <= -lim, 0, 
            jnp.where(
                jnp.abs(x) < lim, 1/(1+jnp.exp(-x)), x
            )
        )
    )

@jit
def jax_sigmoid_grad(x):
    s = jax_sigmoid(x)
    return s * (1-s)

@jit
def jax_softplus(x):
    return jnp.where(
        x > 30, x, jnp.log1p(jnp.exp(x))
    )

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
def jax_zero_to_neg(x):
    return (x > 1e-8).astype(jnp.float32) - (x <= 1e-8).astype(jnp.float32)

@jit
def jax_neg_to_zero(x):
    return (x > 1e-8).astype(jnp.float32)


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

@jit
def compute_delta(g, m, v, beta_1, beta_2, t, learning_rate, epsilon):
    new_m = beta_1 * m + (1 - beta_1) * g
    new_v = beta_2 * v + (1 - beta_2) * jnp.power(g, 2)
    m_hat = new_m / (1 - jnp.power(beta_1, t))
    v_hat = new_v / (1 - jnp.power(beta_2, t))
    delta = jnp.array([lr * m_hat / (jnp.sqrt(v_hat) + epsilon) for lr in learning_rate])
    return delta, new_m, new_v

class jax_adam_optimizer():
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._cache = {}    
    
    def delta(self, grads, name="w", learning_rate=None, gate=None):
        if name not in self._cache:
            self._cache[name] = [
                [jnp.zeros_like(i) for i in grads],
                [jnp.zeros_like(i) for i in grads],
                0
            ]
        self._cache[name][2] += 1
        t = self._cache[name][2]
        deltas = []
        beta_1, beta_2 = self.beta_1, self.beta_2
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        
        for n, g in enumerate(grads):
            m = self._cache[name][0][n]
            v = self._cache[name][1][n]
            delta_n, new_m, new_v = compute_delta(
                g, m, v, beta_1, beta_2, t, learning_rate, self.epsilon
            )
            self._cache[name][0][n] = new_m
            self._cache[name][1][n] = new_v
            deltas.append(delta_n)
        
        return deltas

class MDP(ABC): 
  def __init__(self):
    super().__init__()        

  @abstractmethod
  def reset(self, batch_size):
    pass

  @abstractmethod
  def act(self, actions):
    pass

class jax_complex_multiplexer_MDP(MDP):
    def __init__(self, addr_size=2, action_size=2, zero=True, reward_zero=True):
        self.addr_size = addr_size
        self.action_size = action_size      
        self.x_size = addr_size * action_size + 2 ** addr_size
        self.zero = zero
        self.reward_zero = reward_zero
        self.key = jrandom.PRNGKey(0)
        self.x = None
        self.y = None
        super().__init__()

    def reset(self, key, batch_size):
        """
        Reset the environment.
          key: a JAX PRNGKey
          batch_size: number of independent samples
        Returns:
          x: the generated state (batch_size, x_size) as a jax.Array.
        """
        self.key = key
        addr_size = self.addr_size
        action_size = self.action_size
        x_size = self.x_size
        
        self.key, subkey = jrandom.split(self.key)
        self.x = jrandom.bernoulli(subkey, p=0.5, shape=(batch_size, x_size)).astype(jnp.int32)
        
        factors = 2 ** (addr_size - 1 - jnp.arange(addr_size))
        reshaped = self.x[:, :addr_size * action_size].reshape(batch_size, action_size, addr_size)
        addr = jnp.sum(reshaped * factors, axis=2)  # shape: (batch_size, action_size)
        
        col_indices = addr_size * action_size + addr  # shape: (batch_size, action_size)
        self.y = jnp.take_along_axis(self.x, col_indices, axis=1)
        
        if not self.zero:
            self.y = jax_neg_to_zero(self.y)
            return jax_zero_to_neg(self.x)
        return self.x

    def act(self, actions):
        """
        Compute a reward by comparing actions (an array) to the target self.y.
        Here, actions is assumed to be a jax.Array of shape matching self.y.
        """
        corr = (self.y == actions.astype(jnp.int32)).astype(jnp.float32)
        return corr if self.reward_zero else jax_zero_to_neg(corr)

    def expected_reward(self, p):
        """
        Compute the expected reward for a probability distribution over actions.
        (For action_size=2 only.)
        p: an array of probabilities with shape (batch_size, 2)
        """
        init_val = 0 if self.reward_zero else -1
        reward_f = jnp.full((self.x.shape[0], 2), init_val)
        y_zero = self.y if self.zero else jax_neg_to_zero(self.y)
        reward_f = reward_f.at[jnp.arange(self.x.shape[0]), y_zero[:, 0].astype(jnp.int32)].set(1)
        return jnp.sum(reward_f * p, axis=-1)

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
    """
    Plots the running average return for each curve and saves the plot as an image.

    Parameters:
    - curves (dict): Dictionary containing the curves to plot.
    - names (dict): Dictionary mapping curve keys to their display names.
    - mv_n (int): Window size for moving average.
    - end_n (int): Number of episodes to consider for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - ylim (tuple, optional): Y-axis limits.
    - loc (int): Location code for the legend.
    - save (bool): Whether to save the plot. If False, the plot will not be saved.
    - save_dir (str): Directory where the plot image will be saved.
    - filename (str): Name of the plot image file.
    """
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