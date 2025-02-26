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
from jax import tree_util, jit
import gym
from gymnax.experimental import RolloutWrapper

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

simple_grad_update_jit = jit(simple_grad_update)
adam_update_jit = jit(adam_update)

@jit
def relu(x):
    return jnp.maximum(x, 0)

@jit
def relu_d(x):
    return jnp.where(x < 0, 0, 1)

@jit
def sigmoid(x):
    return jax.nn.sigmoid(x)

@jit
def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

@jit
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

@jit
def multinomial_rvs(key, n, p):
    count = jnp.full(p.shape[:-1], n)
    out = jnp.zeros(p.shape, dtype=jnp.int32)
    ps = jnp.cumsum(p, axis=-1)
    condp = jnp.where(ps == 0, 0.0, p / ps)
    def body_fn(carry, i):
        key, count, out = carry
        key, subkey = jax.random.split(key)
        binsample = jax.random.binomial(subkey, n=count, p=condp[..., i]).astype(jnp.int32)
        out = out.at[..., i].set(binsample)
        count = count - binsample
        return (key, count, out), None
    indices = jnp.arange(p.shape[-1] - 1, 0, -1)
    (key, count, out), _ = jax.lax.scan(body_fn, (key, count, out), indices)
    out = out.at[..., 0].set(count)
    return key, out

@jit
def from_one_hot(y):
    return jnp.argmax(y, axis=-1)

@jit
def to_one_hot(a, size):
    return jnp.eye(size, dtype=jnp.int32)[a.astype(jnp.int32)]

def getl(x, n):
    return x[n] if isinstance(x, list) else x

def zero_to_neg(x):
    return (x > 1e-8).astype(jnp.float32) - (x <= 1e-8).astype(jnp.float32)

def neg_to_zero(x):
    return (x > 1e-8).astype(jnp.float32)

@jit
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
        self._w = jax.random.uniform(jax.random.PRNGKey(random.randint(0, 2**31 - 1)),
                                       shape=(input_size, output_size), minval=-lim, maxval=lim)
        self._b = jnp.zeros((output_size,))
        self._inv_var = jnp.full((output_size,), 1/var)
        
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

    def sample(self, inputs, rng):
        """Accept an RNG key explicitly and return (new_rng, sample)"""
        self.compute_pot_mean(inputs)
        if self.l_type in LS_REAL:
            sigma = jnp.sqrt(1 / self._inv_var)
            rng, subkey = jax.random.split(rng)
            sample = self.mean + sigma * jax.random.normal(subkey, shape=self.pot.shape)
        elif self.l_type == L_DISCRETE:
            rng, subkey = jax.random.split(rng)
            rng, sample = multinomial_rvs(subkey, 1, self.mean)
        self.values = sample
        return rng, sample

    def refresh(self, freeze_value):
        self.compute_pot_mean(self.inputs)
        if not freeze_value:
            self.values = self.new_values

    def update(self, update_size):
        if self.next_layer is None:
            if self.l_type in LS_REAL:
                sigma = jnp.sqrt(1 / self._inv_var)
                self.new_values = self.mean + sigma * jax.random.normal(
                    jax.random.PRNGKey(random.randint(0, 2**31 - 1)), shape=self.pot.shape)
            elif self.l_type == L_DISCRETE:
                self.new_values = multinomial_rvs(
                    jax.random.PRNGKey(random.randint(0, 2**31 - 1)), 1, self.mean)[1]
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
        for d, n in enumerate(hidden + [action_n]):
            l_type = output_l_type if d == len(hidden) else hidden_l_type
            a = eq_prop_layer(name=f"layer_{d}", input_size=in_size, output_size=n,
                              optimizer=optimizer, var=getl(var, d), temp=temp, l_type=l_type)
            if d > 0:
                a.prev_layer = self.layers[-1]
                self.layers[-1].next_layer = a
            self.layers.append(a)
            in_size = n

    def forward(self, state, rng):
        h = state
        for a in self.layers:
            rng, h = a.sample(h, rng)
        self.action = h
        return rng, self.action

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
    f_name = os.path.join("config", f"{args.config}")
    print("Loading config from", f_name)

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(f_name)

    name = config.get("USER", "name")
    max_eps = config.getint("USER", "max_eps")
    n_run = config.getint("USER", "n_run")
    batch_size = config.getint("USER", "batch_size")
    env_name = config.get("USER", "env_name")
    gamma = config.getfloat("USER", "gamma")
    hidden = json.loads(config.get("USER", "hidden"))
    critic_l_type = config.getint("USER", "critic_l_type")
    actor_l_type = config.getint("USER", "actor_l_type")
    temp = config.getfloat("USER", "temp")
    critic_var = json.loads(config.get("USER", "critic_var"))
    critic_update_adj = config.getfloat("USER", "critic_update_adj")
    critic_lambda_ = config.getfloat("USER", "critic_lambda_")
    actor_var = json.loads(config.get("USER", "actor_var"))
    actor_update_adj = config.getfloat("USER", "actor_update_adj")
    actor_lambda_ = config.getfloat("USER", "actor_lambda_")
    map_grad_ascent_steps = config.getint("USER", "map_grad_ascent_steps")
    reward_lim = config.getfloat("USER", "reward_lim")
    critic_lr_st = json.loads(config.get("USER", "critic_lr_st"))
    critic_lr_end = json.loads(config.get("USER", "critic_lr_end"))
    actor_lr_st = json.loads(config.get("USER", "actor_lr_st"))
    actor_lr_end = json.loads(config.get("USER", "actor_lr_end"))
    end_t = config.getint("USER", "end_t")

    env = BatchEnvs(name=env_name, batch_size=batch_size, rest_n=0, warm_n=0)
    dis_act = type(env.action_space) != gymnax.environments.spaces.Box

    critic_update_size = [i * critic_update_adj for i in critic_var]
    actor_update_size = [i * actor_update_adj for i in actor_var]
    critic_lambda_ *= gamma
    actor_lambda_ *= gamma

    print_every = 1000
    eps_ret_hist_full = []

    print("Starting experiments on environment", env_name)

    rng = jax.random.PRNGKey(0)

    for j in range(n_run):
        critic_net = Network(state_n=env.state.shape[1], action_n=1, hidden=hidden, var=critic_var, 
                               temp=None, hidden_l_type=critic_l_type, output_l_type=L_LINEAR)
        output_l_type = L_DISCRETE if dis_act else L_LINEAR
        action_n = env.action_space.n if dis_act else env.action_space.shape[0]
        actor_net = Network(state_n=env.state.shape[1], action_n=action_n, hidden=hidden, var=actor_var, 
                            temp=temp, hidden_l_type=actor_l_type, output_l_type=output_l_type)
        
        eps_ret_hist = []  
        c_eps_ret = jnp.zeros(batch_size)
            
        print_count = print_every         
        isEnd = env.isEnd
        prev_isEnd = env.isEnd
        state = env.reset()
        value_old = None

        for i in range(int(1e9)):     
            rng, action = actor_net.forward(state, rng)
            if not dis_act:
                action = action * (env.action_space.high[0] - env.action_space.low[0]) - env.action_space.low[0]
                action = jnp.clip(action, env.action_space.low, env.action_space.high)
            
            rng, critic_out = critic_net.forward(state, rng)
            value_new = critic_out[:,0]
            mean_value_new = critic_net.layers[-1].mean[:,0]   
            
            if value_old is not None:      
                if reward_lim > 0:
                    reward = jnp.clip(env.reward, -reward_lim, +reward_lim)
                else:
                    reward = env.reward
                targ_value = reward + gamma * mean_value_new * (~env.isEnd).astype(float)      
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

            value_old = value_new
            mean_value_old = mean_value_new
            prev_isEnd = env.isEnd   
            state, reward, isEnd, info = env.step(from_one_hot(action) if dis_act else action)
            
            c_eps_ret += reward
            
            if jnp.any(isEnd):
                eps_ret_hist.extend(c_eps_ret[isEnd].tolist())
                c_eps_ret = c_eps_ret.at[isEnd].set(0.)
            
            if len(eps_ret_hist) >= max_eps:
                break       
            
            if i * batch_size > print_count and len(eps_ret_hist) > 0:      
                eps_ret_hist_jp = jnp.array(eps_ret_hist)
                f_str = "Run %d: Step %d Eps %d\t Running Avg. Return %f\t Max Return %f \t"
                f_arg = [j+1, i, len(eps_ret_hist), jnp.average(eps_ret_hist_jp[-100:]), jnp.amax(eps_ret_hist_jp)]
                print(f_str % tuple(f_arg))
                print_count += print_every          
        eps_ret_hist_full.append(eps_ret_hist)

    print("Finished Training")
    curves = {}  
    curves[name] = (eps_ret_hist_full,)
    names = {k: k for k in curves.keys()}
    f_name = os.path.join("result", "%s.npy" % name) 
    print("Results (saved to %s):" % f_name)
    np.save(f_name, curves)
    print_stat(curves, names)
    plot(curves, names, mv_n=100, end_n=max_eps)

class BatchEnvs:
    def __init__(self, name, batch_size=1, rest_n=100, warm_n=100):
        self.batch_size = int(batch_size)
        self.rest_n = rest_n
        self.warm_n = warm_n

        self.rollout_wrapper = RolloutWrapper(env_name=name)
        self.env = self.rollout_wrapper.env
        self.env_params = self.rollout_wrapper.env_params

        self._gym_action_space = gym.make(name).action_space

        self.key = jax.random.PRNGKey(0)

        keys = jax.random.split(self.key, self.batch_size)
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

        step_fn = lambda key, st, action: self.env.step(key, st, action, self.env_params)

        next_obs, next_state, reward, done, info = jax.vmap(step_fn, in_axes=(0, 0, 0))(
            step_keys, self._full_state, actions
        )

        self._rest = self._rest + jnp.where(done, 1, 0)
        self._warm = self._warm + 1
        self._full_state = next_state 
        self._obs = next_obs      

        self.reward = reward
        self.done = done
        self.info = info 

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
