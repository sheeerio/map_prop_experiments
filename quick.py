#!/usr/bin/env python
from abc import ABC, abstractmethod
import argparse
import configparser
import os
import sys
import json
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from functools import partial
import jax
import jax.numpy as jnp
from jax import tree_util
from jax import jit

key = jax.random.PRNGKey(0)

def simple_grad_update(params, grads, lr):
    return tree_util.tree_map(lambda p, g: p + lr * g, params, grads)

def adam_update(params, grads, opt_state, lr, beta1=0.9, beta2=0.999, epsilon=1e-09):
    t = opt_state['t'] + 1
    new_m = tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, opt_state['m'], grads)
    new_v = tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g), opt_state['v'], grads)
    m_hat = tree_util.tree_map(lambda m: m / (1 - beta1 ** t), new_m)
    v_hat = tree_util.tree_map(lambda v: v / (1 - beta2 ** t), new_v)
    new_params = tree_util.tree_map(lambda p, m, v: p + lr * m / (jnp.sqrt(v) + epsilon),
                                      params, m_hat, v_hat)
    new_state = {'m': new_m, 'v': new_v, 't': t}
    return new_params, new_state

simple_grad_update_jit = jax.jit(simple_grad_update)
adam_update_jit = jax.jit(adam_update)

@jax.jit
def relu(x):
    return jnp.maximum(x, 0)

@jax.jit
def relu_d(x):
    return jnp.where(x < 0, 0, 1)

@jax.jit
def sigmoid(x):
    lim = 20.0
    return jnp.where(x <= -lim, 0.0,
                     jnp.where(x >= lim, 1.0,
                               1.0 / (1.0 + jnp.exp(-x))))

@jax.jit
def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

@jax.jit
def softplus(x):
    return jnp.where(x > 30, x, jnp.log1p(jnp.exp(x)))

@partial(jax.jit, static_argnums=(2,))
def softmax(X, theta=1.0, axis=-1):
    y = X * theta
    max_val = jnp.max(y, axis=axis, keepdims=True)
    y = y - max_val
    # Clip values to avoid numerical underflow
    y = jnp.clip(y, a_min=-30, a_max=None)
    y = jnp.exp(y)
    ax_sum = jnp.sum(y, axis=axis, keepdims=True)
    return y / ax_sum

from jax import lax
@jax.jit
def multinomial_rvs(key, n, p):
    num_categories = p.shape[-1]
    count = jnp.full(p.shape[:-1], n)
    out = jnp.zeros_like(p, dtype=jnp.int32)
    ps = jnp.cumsum(p, axis=-1)
    condp = jnp.where(ps == 0, 0.0, p / ps)

    def body(i, state):
        key, count, out = state
        idx = num_categories - i
        key, subkey = jax.random.split(key)
        sample = jax.random.binomial(subkey, n=count, p=condp[..., idx]).astype(jnp.int32)
        out = out.at[..., idx].set(sample)
        count = count - sample
        return (key, count, out)

    state = (key, count, out)
    state = lax.fori_loop(1, num_categories, body, state)
    key, count, out = state
    out = out.at[..., 0].set(count)
    return key, out

@jax.jit
def from_one_hot(y):
    return jnp.argmax(y, axis=-1)

@jax.jit
def to_one_hot(a, size):
    return jnp.eye(size, dtype=jnp.int32)[a.astype(jnp.int32)]

def getl(x, n):
    return x[n] if isinstance(x, list) else x

@jax.jit
def equal_zero(x):
    return jnp.logical_and(x > -1e-8, x < 1e-8).astype(jnp.float32)

@jax.jit
def mask_neg(x):
    return (x < 0).astype(jnp.float32)

@jax.jit
def sign(x):
    return (x > 1e-8).astype(jnp.float32) - (x < -1e-8).astype(jnp.float32)

@jax.jit
def zero_to_neg(x):
    return (x > 1e-8).astype(jnp.float32) - (x <= 1e-8).astype(jnp.float32)

@jax.jit
def neg_to_zero(x):
    return (x > 1e-8).astype(jnp.float32)

@jax.jit
def apply_mask(x, mask):
    return jnp.where(mask, x, 0)

@jax.jit
def linear_interpolat(start, end, end_t, cur_t):
    if isinstance(start, list):
        if isinstance(end_t, list):
            return [(e - s) * jnp.minimum(cur_t, d) / d + s for (s, e, d) in zip(start, end, end_t)]
        else:
            return [(e - s) * jnp.minimum(cur_t, end_t) / end_t + s for s, e in zip(start, end)]
    else:
        if isinstance(end_t, list):
            return [(end - start) * jnp.minimum(cur_t, d) / d + start for d in end_t]
        else:
            return (end - start) * jnp.minimum(cur_t, end_t) / end_t

class simple_grad_optimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class adam_optimizer():
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}

class MDP(ABC):
    def __init__(self):
        super().__init__()
        self.rng = jax.random.PRNGKey(random.randint(0, 2**31 - 1))

    @abstractmethod
    def reset(self, batch_size):
        pass

    @abstractmethod
    def act(self, actions):
        pass

class complex_multiplexer_MDP(MDP):
    def __init__(self, addr_size=2, action_size=2, zero=True, reward_zero=True):
        self.addr_size = addr_size
        self.action_size = action_size
        self.x_size = addr_size * action_size + 2 ** addr_size
        self.zero = zero
        self.reward_zero = reward_zero
        super().__init__()

    def reset(self, batch_size):
        addr_size = self.addr_size
        action_size = self.action_size
        x_size = self.x_size
        self.rng, subkey = jax.random.split(self.rng)
        self.x = jax.random.bernoulli(subkey, p=0.5, shape=(batch_size, x_size)).astype(jnp.int32)
        x_reshaped = self.x[:, :addr_size * action_size].reshape((batch_size, action_size, addr_size))
        weights = 2 ** (addr_size - 1 - jnp.arange(addr_size))
        addr = jnp.sum(x_reshaped * weights, axis=2)
        indices = ((addr_size * action_size + addr).astype(jnp.int32))[:, None]
        rows = jnp.arange(self.x.shape[0])[:, None]  # shape (batch_size, 1)
        cols = addr_size * action_size + addr         # shape (batch_size, action_size)
        self.y = self.x[rows, cols]
        if not self.zero:
            self.y = zero_to_neg(self.y)
        return self.x if self.zero else zero_to_neg(self.x)

    def act(self, actions):
        return act_complex_multiplexer(self.y, actions, self.reward_zero)

    def expected_reward(self, p):
        reward_f = jnp.full((self.x.shape[0], 2), 0.0 if self.reward_zero else -1.0)
        y_zero = self.y if self.zero else neg_to_zero(self.y)
        reward_f = reward_f.at[jnp.arange(self.x.shape[0]), y_zero.astype(jnp.int32)].set(1)
        return jnp.sum(reward_f * p, axis=-1)

@jax.jit
def act_complex_multiplexer(y, actions, reward_zero):
    corr = (y == actions.astype(jnp.int32)).astype(jnp.float32)
    return jnp.where(reward_zero, corr, (corr > 0).astype(jnp.float32) - (corr <= 0).astype(jnp.float32))

def mv(a, n=1000):
    ret = jnp.cumsum(a, dtype=jnp.float32)
    ret = ret.at[n:].set(ret[n:] - ret[:-n])
    return ret[n - 1:] / n

def plot(curves, names, mv_n=100, end_n=1000, xlabel="Episodes", ylabel="Running Average Return",
         ylim=None, loc=4, save=True, save_dir="./result/plots/", filename="plot.png"):
    plt.figure(figsize=(10, 7), dpi=150)
    colors = ['red', 'blue', 'green', 'crimson', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']

    for i, m in enumerate(names.keys()):
        v = jnp.array([mv(ep[:end_n], mv_n) for ep in curves[m][0]])
        v_mean = jnp.mean(v, axis=0)
        r_std = jnp.std(v, axis=0) / jnp.sqrt(len(curves[m][0]))
        v_full = jnp.concatenate([jnp.full((mv_n - 1,), jnp.nan), v_mean])
        ax = plt.gca()
        ax.plot(np.arange(len(v_full)), np.array(v_full), label=names[m], color=colors[i % len(colors)])
        ax.fill_between(np.arange(len(v_full)),
                        np.array(v_full - r_std),
                        np.array(v_full + r_std),
                        alpha=0.2, color=colors[i % len(colors)])
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(loc=loc, fontsize=12)
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

def print_stat(curves, names):
    for m in names.keys():
        print("Stat. on %s:" % m)
        r = np.average(np.array(curves[m][0]), axis=1)
        print("Return: avg. %.2f median %.2f min %.2f max %.2f std %.2f" %
              (np.average(r), np.median(r), np.amin(r), np.amax(r), np.std(r)))

L_SOFTPLUS = 0
L_RELU = 1
L_LINEAR = 2
L_SIGMOID = 3
L_DISCRETE = 4
LS_REAL = [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID]

ACT_F = {L_SOFTPLUS: softplus,
         L_RELU: relu,
         L_SIGMOID: sigmoid,
         L_LINEAR: lambda x: x}

ACT_D_F = {L_SOFTPLUS: sigmoid,
           L_RELU: relu_d,
           L_SIGMOID: sigmoid_d,
           L_LINEAR: lambda x: 1}

@partial(jax.jit, static_argnums=(3,4))
def eq_prop_compute_pot_mean(inputs, w, b, l_type, temp, inv_var):
    pot = jnp.dot(inputs, w) + b
    if l_type in LS_REAL:
        mean = ACT_F[l_type](pot)
    else:
        mean = softmax(pot / temp, theta=1.0, axis=-1)
    return pot, mean

@partial(jax.jit, static_argnums=(4,5))
def eq_prop_sample(key, inputs, w, b, l_type, temp, inv_var):
    pot, mean = eq_prop_compute_pot_mean(inputs, w, b, l_type, temp, inv_var)
    if l_type in LS_REAL:
        sigma = jnp.sqrt(1.0 / inv_var)
        key, subkey = jax.random.split(key)
        values = mean + sigma * jax.random.normal(subkey, shape=pot.shape)
    else:
        key, values = multinomial_rvs(key, 1, mean)
    return inputs, key, pot, mean, values

@partial(jax.jit, static_argnums=(5,6))
def eq_prop_record_trace(inputs, values, mean, pot, inv_var, temp, l_type, gate, lambda_, w_trace, b_trace):
    if l_type in LS_REAL:
        v_ch = (values - mean) * ACT_D_F[l_type](pot) * inv_var
    else:
        v_ch = (values - mean) / temp
    if gate is not None:
        v_ch = v_ch * gate[:, None]
    new_w_trace = w_trace * lambda_ + inputs[:, :, None] * v_ch[:, None, :]
    new_b_trace = b_trace * lambda_ + v_ch
    return new_w_trace, new_b_trace

@jax.jit
def eq_prop_learn_trace(w_trace, b_trace, reward, w, b, opt_state_w, opt_state_b, lr):
    w_update = w_trace * reward[:, None, None]
    b_update = b_trace * reward[:, None]
    w_update = jnp.mean(w_update, axis=0)
    b_update = jnp.mean(b_update, axis=0)
    new_w, new_opt_state_w = adam_update_jit(w, w_update, opt_state_w, lr)
    new_b, new_opt_state_b = adam_update_jit(b, b_update, opt_state_b, lr)
    return new_w, new_b, new_opt_state_w, new_opt_state_b

class eq_prop_layer():
    def __init__(self, name, input_size, output_size, optimizer, var, temp, l_type):
        if l_type not in [L_SOFTPLUS, L_RELU, L_LINEAR, L_SIGMOID, L_DISCRETE]:
            raise Exception('l_type (%d) not implemented' % l_type)
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.l_type = l_type
        self.temp = temp if l_type == L_DISCRETE else 1

        lim = jnp.sqrt(2 / (input_size + output_size))
        self.rng = jax.random.PRNGKey(random.randint(0, 2**31 - 1))
        self.rng, subkey = jax.random.split(self.rng)
        self._w = jax.random.uniform(subkey, shape=(input_size, output_size), minval=-lim, maxval=lim)
        self._b = jnp.zeros((output_size,))
        self._inv_var = jnp.full((output_size,), 1/var)
        
        # Initialize optimizer states for weights and biases:
        self.opt_state_w = {'m': jnp.zeros_like(self._w), 'v': jnp.zeros_like(self._w), 't': 0}
        self.opt_state_b = {'m': jnp.zeros_like(self._b), 'v': jnp.zeros_like(self._b), 't': 0}
        


        self.prev_layer = None
        self.next_layer = None
        self.values = jnp.zeros((1, output_size))
        self.new_values = jnp.zeros((1, output_size))
        self.w_trace = jnp.zeros((1, input_size, output_size))
        self.b_trace = jnp.zeros((1, output_size))

    def compute_pot_mean(self, inputs):
        self.inputs = inputs
        self.pot, self.mean = eq_prop_compute_pot_mean(self.inputs, self._w, self._b, self.l_type, self.temp, self._inv_var)

    def sample(self, inputs):
        self.inputs, self.rng, self.pot, self.mean, self.values = eq_prop_sample(self.rng, inputs, self._w, self._b, self.l_type, self.temp, self._inv_var)
        return self.values
  
    def refresh(self, freeze_value):
        if self.prev_layer is not None:
            self.inputs = self.prev_layer.new_values
        self.compute_pot_mean(self.inputs)
        if not freeze_value:
            self.values = self.new_values

    def update(self, update_size):
        if self.next_layer is None:
            if self.l_type in LS_REAL:
                sigma = jnp.sqrt(1 / self._inv_var)
                self.rng, subkey = jax.random.split(self.rng)
                self.new_values = self.mean + sigma * jax.random.normal(subkey, shape=self.pot.shape)
            elif self.l_type == L_DISCRETE:
                self.rng, subkey = jax.random.split(self.rng)
                self.rng, sampled = multinomial_rvs(subkey, 1, self.mean)
                self.new_values = sampled
        elif self.l_type in LS_REAL:
            lower_pot = (self.mean - self.values) * self._inv_var
            if self.next_layer is None:
                upper_pot = 0.
            else:
                fb_w = self.next_layer._w.T
                if self.next_layer.l_type in LS_REAL:
                    upper_pot = jnp.dot((self.next_layer.values - self.next_layer.mean) *
                                          ACT_D_F[self.next_layer.l_type](self.next_layer.pot) *
                                          self.next_layer._inv_var,
                                          fb_w)
                else:
                    upper_pot = jnp.dot((self.next_layer.values - self.next_layer.mean), fb_w) / self.next_layer.temp
            update_pot = lower_pot + upper_pot
            update_step = update_size * update_pot
            self.new_values = self.values + update_step

    def record_trace(self, gate=None, lambda_=0):
        self.w_trace, self.b_trace = eq_prop_record_trace(
            self.inputs, self.values, self.mean, self.pot,
            self._inv_var, self.temp, self.l_type,
            gate, lambda_, self.w_trace, self.b_trace
        )

    def learn_trace(self, reward, lr=0.01):
        self._w, self._b, self.opt_state_w, self.opt_state_b = eq_prop_learn_trace(
            self.w_trace, self.b_trace, reward,
            self._w, self._b, self.opt_state_w, self.opt_state_b, lr
        )

    def clear_trace(self, mask):
        mask = mask.astype(jnp.float32)
        self.w_trace = self.w_trace * mask[:, None, None]
        self.b_trace = self.b_trace * mask[:, None]

    def clear_values(self, mask):
        self.values = self.values * mask[:, None]
        self.new_values = self.new_values * mask[:, None]

class Network():
    def __init__(self, state_n, action_n, hidden, var, temp, hidden_l_type, output_l_type):
        self.layers = []
        in_size = state_n
        optimizer = adam_optimizer(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        self.rng = jax.random.PRNGKey(random.randint(0, 2**31 - 1))
        for d, n in enumerate(hidden + [action_n]):
            l_type = output_l_type if d == len(hidden) else hidden_l_type
            a = eq_prop_layer(name="layer_%d" % d, input_size=in_size, output_size=n,
                              optimizer=optimizer, var=getl(var, d), temp=temp, l_type=l_type)
            self.rng, layer_key = jax.random.split(self.rng)
            a.rng = layer_key
            if d > 0:
                a.prev_layer = self.layers[-1]
                self.layers[-1].next_layer = a
            self.layers.append(a)
            in_size = n

    def forward(self, state):
        self.state = state
        h = state
        for a in self.layers:
            h = a.sample(h)
        self.action = h
        return self.action

    def map_grad_ascent(self, steps, state=None, gate=False, lambda_=0, update_size=0.01):
        for i in range(steps):
            for n, a in enumerate(self.layers if state is not None else self.layers[:-1]):
                a.update(getl(update_size, n))
            for n, a in enumerate(self.layers):
                a.refresh(freeze_value=(n == (len(self.layers) - 1) and state is None))
            if i == steps - 1:
                gate_c = 1 / (self.layers[-1].values - self.layers[-1].mean)[:, 0] if gate else None
                for a in self.layers:
                    a.record_trace(gate=gate_c, lambda_=lambda_)

    def learn(self, reward, lr=0.01):
        for n, a in enumerate(self.layers):
            a.learn_trace(reward=reward, lr=getl(lr, n))

    def clear_trace(self, mask):
        for a in self.layers:
            a.clear_trace(mask)

    def clear_values(self, mask):
        for a in self.layers:
            a.clear_values(mask)

def main():
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument(
        "-c", "--config",
        default="config_mp.ini",
        help="Location of config file (default: config_mp.ini)"
    )
    args, remaining_argv = initial_parser.parse_known_args()

    config_dir = "config"
    f_name = os.path.join(config_dir, args.config)
    print(f"Loading config from {f_name}")

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    if not config.read(f_name):
        print(f"Error: Config file '{f_name}' not found or is invalid.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        parents=[initial_parser],
        description="Script with configurable parameters via config file and command-line flags."
    )

    parser.add_argument(
        "--name",
        default=config.get("DEFAULT", "name"),
        help="Name identifier for the run."
    )
    parser.add_argument(
        "--exp_num",
        type=int,
        default=1,
        help="Experiment number to help with tracking"
    )
    parser.add_argument(
        "--max_eps",
        type=int,
        default=config.getint("DEFAULT", "max_eps"),
        help="Number of episodes per run."
    )
    parser.add_argument(
        "--n_run",
        type=int,
        default=config.getint("DEFAULT", "n_run"),
        help="Number of runs."
    )

    parser.add_argument(
        "--env_name",
        default=config.get("DEFAULT", "env_name"),
        choices=["Multiplexer", "Regression"],
        help="Environment name (e.g., Multiplexer, Regression)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.getint("DEFAULT", "batch_size"),
        help="Batch size."
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default=config.get("DEFAULT", "hidden"),
        help="JSON list of hidden units per layer (e.g., '[64, 32]')."
    )
    parser.add_argument(
        "--l_type",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=config.getint("DEFAULT", "l_type"),
        help="Activation function type: 0=Softplus, 1=ReLU, 2=Linear, 3=Sigmoid, 4=Discrete."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=config.getfloat("DEFAULT", "temp"),
        help="Temperature for the network if applicable."
    )
    parser.add_argument(
        "--var",
        type=str,
        default=config.get("DEFAULT", "var"),
        help="JSON list of variances in hidden layers (e.g., '[0.3, 1, 1]')."
    )
    parser.add_argument(
        "--update_adj",
        type=float,
        default=config.getfloat("DEFAULT", "update_adj"),
        help="Step size for energy minimization adjustment."
    )
    parser.add_argument(
        "--map_grad_ascent_steps",
        type=int,
        default=config.getint("DEFAULT", "map_grad_ascent_steps"),
        help="Number of gradient ascent steps for energy minimization."
    )
    parser.add_argument(
        "--lr",
        type=str,
        default=config.get("DEFAULT", "lr"),
        help="JSON list of learning rates (e.g., '[0.04, 0.00004, 0.000004]')."
    )

    args = parser.parse_args()

    try:
        hidden = json.loads(args.hidden)
        if not isinstance(hidden, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        print("Error: 'hidden' not a valid JSON")
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

    if args.env_name == "Multiplexer":
        env = complex_multiplexer_MDP(
            addr_size=5,
            action_size=1,
            zero=False,
            reward_zero=False
        )
        gate = False
        output_l_type = L_DISCRETE
        action_n = 2 ** env.action_size
    else:
        print(f"Error: Unsupported environment '{args.env_name}'.")
        sys.exit(1)

    update_size = [i * args.update_adj for i in var]
    print_every = 128 * 500

    eps_ret_hist_full = []
    for j in range(args.n_run):
        net = Network(
            state_n=env.x_size,
            action_n=action_n,
            hidden=hidden,
            var=var,
            temp=args.temp,
            hidden_l_type=args.l_type,
            output_l_type=output_l_type
        )

        eps_ret_hist = []
        print_count = print_every
        for i in range(args.max_eps // args.batch_size):
            state = env.reset(args.batch_size)
            action = net.forward(state)
            if args.env_name == "Multiplexer":
                action = zero_to_neg(from_one_hot(action))[:, jnp.newaxis]
                reward = env.act(action)[:, 0]
            elif args.env_name == "Regression":
                action = action[:, 0]
                reward = env.act(action)
            eps_ret_hist.append(np.average(np.array(reward)))

            net.map_grad_ascent(
                steps=args.map_grad_ascent_steps,
                state=None,
                gate=gate,
                lambda_=0,
                update_size=update_size
            )

            if args.env_name == "Regression":
                reward = env.y - net.layers[-1].mean[:, 0]

            net.learn(reward, lr=lr)

            if (i * args.batch_size) > print_count:
                running_avg = np.average(eps_ret_hist[-print_every // args.batch_size:])
                print(f"Run {j} Step {i} Running Avg. Reward\t{running_avg:.6f}")
                print_count += print_every
        eps_ret_hist_full.append(eps_ret_hist)

    eps_ret_hist_full = np.asarray(eps_ret_hist_full, dtype=np.float32)
    print("Finished Training")

    curves = {args.name: (eps_ret_hist_full,)}
    names = {k: k for k in curves.keys()}

    plots_dir = os.path.join(f"result/exp{args.exp_num}", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    result_dir = f"result/exp{args.exp_num}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.name}.npy")
    print(f"Results (saved to {result_file}):")
    np.save(result_file, curves)
    print_stat(curves, names)

    # plot_filename = f"{args.name}_plot.png"
    # plot(curves, names, mv_n=10, end_n=args.max_eps, save=True, save_dir=plots_dir, filename=plot_filename)

if __name__ == "__main__":
    main()
