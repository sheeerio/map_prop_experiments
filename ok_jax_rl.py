#!/usr/bin/env python
from abc import ABC, abstractmethod
import argparse
import configparser
import os
import sys
import json
import random
import numpy as np
import gymnax
import matplotlib.pyplot as plt
from functools import partial
import jax
import jax.numpy as jnp
from jax import tree_util
from jax import jit
import gym
from gymnax.experimental import RolloutWrapper


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
    return jax.nn.sigmoid(x)

@jax.jit
def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

@jax.jit
def softplus(x):
    return jnp.where(x > 30, x, jnp.log1p(jnp.exp(x)))

def softmax(X, theta=1.0, axis=None):
    y = jnp.atleast_2d(X)
    if axis is None:
        for i, dim in enumerate(y.shape):
            if dim > 1:
                axis = i
                break
    y = y * theta
    max_val = jnp.max(y, axis=axis, keepdims=True)
    y = y - max_val
    y = jnp.where(y < -30, -30, y)
    y = jnp.exp(y)
    ax_sum = jnp.sum(y, axis=axis, keepdims=True)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p

def multinomial_rvs(key, n, p):
    count = jnp.full(p.shape[:-1], n)
    out = jnp.zeros(p.shape, dtype=jnp.int32)
    ps = jnp.cumsum(p, axis=-1)
    condp = jnp.where(ps == 0, 0.0, p / ps)
    for i in range(p.shape[-1] - 1, 0, -1):
        key, subkey = jax.random.split(key)
        binsample = jax.random.binomial(subkey, n=count, p=condp[..., i]).astype(jnp.int32)
        out = out.at[..., i].set(binsample)
        count = count - binsample
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

def apply_mask(x, mask):
    return (x.T * mask).T

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

    def delta(self, name, grads, learning_rate=None):
        lr = self.learning_rate if learning_rate is None else learning_rate
        return [lr * g for g in grads]

class adam_optimizer():
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}

    def delta(self, grads, name="w", learning_rate=None, gate=None):
        if name not in self._cache:
            self._cache[name] = [[jnp.zeros_like(g) for g in grads],
                                 [jnp.zeros_like(g) for g in grads],
                                 0]
        self._cache[name][2] += 1
        t = self._cache[name][2]
        deltas = []
        lr = self.learning_rate if learning_rate is None else learning_rate
        for n, g in enumerate(grads):
            m = self._cache[name][0][n]
            v = self._cache[name][1][n]
            m = self.beta_1 * m + (1 - self.beta_1) * g
            v = self.beta_2 * v + (1 - self.beta_2) * jnp.power(g, 2)
            self._cache[name][0][n] = m
            self._cache[name][1][n] = v
            m_hat = m / (1 - jnp.power(self.beta_1, t))
            v_hat = v / (1 - jnp.power(self.beta_2, t))
            deltas.append(lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon))
        return deltas

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
        self.pot = jnp.dot(inputs, self._w) + self._b
        if self.l_type in LS_REAL:
            self.mean = ACT_F[self.l_type](self.pot)
        else:
            self.mean = softmax(self.pot / self.temp, axis=-1)

    def sample(self, inputs):
        self.compute_pot_mean(inputs)
        if self.l_type in LS_REAL:
            sigma = jnp.sqrt(1 / self._inv_var)
            self.rng, subkey = jax.random.split(self.rng)
            self.values = self.mean + sigma * jax.random.normal(subkey, shape=self.pot.shape)
            return self.values
        elif self.l_type == L_DISCRETE:
            self.rng, subkey = jax.random.split(self.rng)
            self.rng, sampled = multinomial_rvs(subkey, 1, self.mean)
            self.values = sampled
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
        if self.l_type in LS_REAL:
            v_ch = (self.values - self.mean) * ACT_D_F[self.l_type](self.pot) * self._inv_var
        else:
            v_ch = (self.values - self.mean) / self.temp
        if gate is not None:
            v_ch = v_ch * gate[:, None]
        self.w_trace = self.w_trace * lambda_ + self.inputs[:, :, None] * v_ch[:, None, :]
        self.b_trace = self.b_trace * lambda_ + v_ch

    def learn_trace(self, reward, lr=0.01):
        w_update = self.w_trace * reward[:, None, None]
        b_update = self.b_trace * reward[:, None]
        w_update = jnp.mean(w_update, axis=0)
        b_update = jnp.mean(b_update, axis=0)

        self._w, self.opt_state_w = adam_update_jit(self._w, w_update, self.opt_state_w, lr)
        self._b, self.opt_state_b = adam_update_jit(self._b, b_update, self.opt_state_b, lr)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=False, default="config_cp.ini",
    help="location of config file")
    args = ap.parse_args()
    f_name = os.path.join("config", "%s" % args.config)
    print("Loading config from %s" % f_name)

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(f_name)

    name = config.get("USER", "name") # Name of the run
    max_eps = config.getint("USER", "max_eps") # Number of episode per run
    n_run = config.getint("USER", "n_run") # Number of runs

    batch_size = config.getint("USER", "batch_size") # Batch size
    env_name = config.get("USER", "env_name") # Environment name
    gamma = config.getfloat("USER", "gamma") # Discount rate

    hidden = json.loads(config.get("USER","hidden")) # Number of hidden units on each layer
    critic_l_type = config.getint("USER", "critic_l_type")  # Activation function for hidden units in critic network; 0 for softplus and 1 for ReLu
    actor_l_type = config.getint("USER", "actor_l_type")  # Activation function for hidden units in actor network; 0 for softplus and 1 for ReLu
    temp = config.getfloat("USER", "temp") # Temperature for actor network if applicable

    critic_var = json.loads(config.get("USER","critic_var")) # Variance in the normal distribution of critic network's layer
    critic_update_adj = config.getfloat("USER", "critic_update_adj") # Step size for minimizing the energy of critic network equals to the layer's variance multiplied by this constant
    critic_lambda_ = config.getfloat("USER", "critic_lambda_") # Trace decay rate for critic network

    actor_var = json.loads(config.get("USER","actor_var")) # Variance in the normal distribution of actor network's layer
    actor_update_adj = config.getfloat("USER", "actor_update_adj") # Step size for minimizing the energy of actor network equals to the layer's variance multiplied by this constant
    actor_lambda_ = config.getfloat("USER", "actor_lambda_") # Trace decay rate for actor network

    map_grad_ascent_steps = config.getint("USER", "map_grad_ascent_steps") # number of step for minimizing the energy
    reward_lim = config.getfloat("USER", "reward_lim") # whether limit the size of reward

    critic_lr_st = json.loads(config.get("USER","critic_lr_st")) # Learning rate for each critic network's layer at the beginning
    critic_lr_end = json.loads(config.get("USER","critic_lr_end")) # Learning rate for each critic network's layer at the beginning
    actor_lr_st = json.loads(config.get("USER","actor_lr_st")) # Learning rate for each actor network's layer at the end
    actor_lr_end = json.loads(config.get("USER","actor_lr_end")) # Learning rate for each actor network's layer at the end
    end_t = config.getint("USER", "end_t") # Number of step to reach the final learning rate (linear interpolation for in-between steps)

    L_SOFTPLUS = 0
    L_RELU = 1
    L_LINEAR = 2
    L_SIGMOID = 3
    L_DISCRETE = 4


    env = BatchEnvs(name=env_name, batch_size=batch_size, rest_n=0, warm_n=0)   
    dis_act = type(env.action_space) != gymnax.environments.spaces.Box
    
    critic_update_size = [i * critic_update_adj for i in critic_var]
    actor_update_size = [i * actor_update_adj for i in actor_var]
    critic_lambda_ *= gamma
    actor_lambda_ *= gamma

    print_every = 10000
    eps_ret_hist_full = []
    print("Starting experiments on environment %s" % env_name)

    for j in range(n_run):
        # this critic network takes in the state dim, action dim, number of hidden units per layer, variance per layer, 
        # hidden_l_type : activation function for hidden units in critic network; 0 for softplus and 1 for ReLu
        # output_l_type : activation function for the output
        critic_net = Network(state_n=env.state.shape[1], action_n=1, hidden=hidden, var=critic_var, 
                            temp=None, hidden_l_type=critic_l_type, output_l_type=L_LINEAR,)   
        output_l_type = L_DISCRETE if dis_act else L_LINEAR   
        action_n = env.action_space.n if dis_act else env.action_space.shape[0]    
        actor_net = Network(state_n=env.state.shape[1], action_n=action_n, hidden=hidden, var=actor_var, 
                            temp=temp, hidden_l_type=actor_l_type, output_l_type=output_l_type)
        
        eps_ret_hist = []  
        c_eps_ret = jnp.zeros(batch_size)
            
        print_count = print_every         
        value_old = None
        isEnd = env.isEnd
        prev_isEnd = env.isEnd
        truncated, solved, f_perfect = False, False, False
        
        state = env.reset()
        for i in range(int(1e9)):     
            action = actor_net.forward(state)

            if not dis_act:
                action = action * (env.action_space.high[0] - env.action_space.low[0]) - env.action_space.low[0]
                action = jnp.clip(action, env.action_space.low, env.action_space.high)
            
            value_new = critic_net.forward(state)[:,0]
            mean_value_new = critic_net.layers[-1].mean[:,0]   
            
            if value_old is not None:      
                if reward_lim > 0: reward = jnp.clip(reward, -reward_lim, +reward_lim)
                targ_value = reward + gamma * mean_value_new * (~isEnd).astype(float)      
                critic_reward = targ_value - mean_value_old
                critic_reward = critic_reward.at[prev_isEnd].set(0)
                actor_reward = targ_value - mean_value_old
                actor_reward = actor_reward.at[prev_isEnd].set(0)            
                
                cur_critic_lr = linear_interpolat(start=critic_lr_st, end=critic_lr_end, end_t=end_t, cur_t=i)
                cur_actor_lr = linear_interpolat(start=actor_lr_st, end=actor_lr_end, end_t=end_t, cur_t=i)  
                
                critic_net.learn(critic_reward, lr=cur_critic_lr)      
                actor_net.learn(actor_reward, lr=cur_actor_lr)          
        
            critic_net.clear_trace(~prev_isEnd)
            critic_net.map_grad_ascent(steps=map_grad_ascent_steps, state=None, gate=True, lambda_=critic_lambda_, 
                            update_size=critic_update_size)           

            actor_net.clear_trace(~prev_isEnd)  
            actor_net.map_grad_ascent(steps=map_grad_ascent_steps, state=None, gate=None, lambda_=actor_lambda_, 
                                    update_size=actor_update_size)              

            value_old = jnp.copy(value_new)
            mean_value_old = jnp.copy(mean_value_new)
            prev_isEnd = jnp.copy(isEnd)    
            state, reward, isEnd, info = env.step(from_one_hot(action) if dis_act else action)
            
            c_eps_ret += reward
            
            if jnp.any(isEnd):
                eps_ret_hist.extend(c_eps_ret[isEnd].tolist())
                c_eps_ret = c_eps_ret.at[isEnd].set(0.)
            
            if len(eps_ret_hist) >= max_eps: break       
            
            if i*batch_size > print_count and len(eps_ret_hist) > 0:      
                f_str = "Run %d: Step %d Eps %d\t Running Avg. Return %f\t Max Return %f \t" 
                eps_ret_hist_jp = jnp.array(eps_ret_hist)
                f_arg = [j+1, i, len(eps_ret_hist), jnp.average(eps_ret_hist_jp[-100:]), jnp.amax(eps_ret_hist_jp),]
                print(f_str % tuple(f_arg))
                print_count += print_every          
        eps_ret_hist_full.append(eps_ret_hist)

    print("Finished Training")
    curves = {}  
    curves[name] = (eps_ret_hist_full,)
    names = {k:k for k in curves.keys()}
    f_name = os.path.join("result", "%s.npy" % name) 
    print("Results (saved to %s):" % f_name)
    np.save(f_name, curves)
    print_stat(curves, names)
    plot(curves, names, mv_n=100, end_n=max_eps)
  # to get the gym action_space

class BatchEnvs:
    def __init__(self, name, batch_size=1, rest_n=100, warm_n=100):
        self.batch_size = int(batch_size)
        self.rest_n = rest_n
        self.warm_n = warm_n

        # Use Gymnax's RolloutWrapper to create the environment and its parameters.
        self.rollout_wrapper = RolloutWrapper(env_name=name)
        self.env = self.rollout_wrapper.env
        self.env_params = self.rollout_wrapper.env_params

        # For access to the Gym action space.
        self._gym_action_space = gym.make(name).action_space

        # Initialize a PRNG key.
        self.key = jax.random.PRNGKey(0)

        # Initialize the batched environment state via vmap over the reset function.
        keys = jax.random.split(self.key, self.batch_size)
        # env.reset returns (obs, full_state)
        obs, full_state = jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)
        self._obs = obs
        self._full_state = full_state

        # Initialize our extra counters.
        self._rest = jnp.zeros(self.batch_size)
        self._warm = jnp.zeros(self.batch_size)
        self._stateCode = jnp.zeros(self.batch_size, dtype=jnp.int32)

        # Placeholders for rewards, done flags, and info.
        self.reward = jnp.zeros(self.batch_size)
        self.done = jnp.zeros(self.batch_size, dtype=bool)
        self.info = {"stateCode": self._stateCode,
                     "truncatedEnd": jnp.zeros(self.batch_size, dtype=bool)}

    def reset(self):
        """Reset all environments and counters; return the observation."""
        self.key, subkey = jax.random.split(self.key)
        keys = jax.random.split(subkey, self.batch_size)
        obs, full_state = jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)
        self._obs = obs
        self._full_state = full_state

        self._rest = jnp.zeros(self.batch_size)
        self._warm = jnp.zeros(self.batch_size)
        self._stateCode = jnp.zeros(self.batch_size, dtype=jnp.int32)
        self.reward = jnp.zeros(self.batch_size)
        self.done = jnp.zeros(self.batch_size, dtype=bool)
        self.info = {"stateCode": self._stateCode,
                     "truncatedEnd": jnp.zeros(self.batch_size, dtype=bool)}
        return self.observation

    def step(self, actions):
        # Generate a separate rng key for each environment in the batch.
        self.key, subkey = jax.random.split(self.key)
        step_keys = jax.random.split(subkey, self.batch_size)

        # step_fn returns ((next_obs, next_state), reward, done, info).
        step_fn = lambda key, st, action: self.env.step(key, st, action, self.env_params)

        next_obs, next_state, reward, done, info = jax.vmap(step_fn, in_axes=(0, 0, 0))(
            step_keys, self._full_state, actions
        )

        # Update custom counters, etc...
        self._rest = self._rest + jnp.where(done, 1, 0)
        self._warm = self._warm + 1
        ...
        # Now store them properly:
        self._full_state = next_state   # The internal state used by Gymnax
        self._obs = next_obs           # The observation used by your agent

        self.reward = reward
        self.done = done
        self.info = info  # or add your own fields if needed

        return self._obs, self.reward, self.done, self.info

    @property
    def observation(self):
        """Return the observation for training (not the full state)."""
        return self._obs

    @property
    def full_state(self):
        """Return the full environment state (for internal use)."""
        return self._full_state
      
    @property
    def state(self):
        return self._obs

    @property
    def action_space(self):
        return self._gym_action_space

    @property
    def isEnd(self):
        return self.done




if __name__ == "__main__":
    main()
