#!/usr/bin/env python
import argparse
import configparser
import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import tree_util
from typing import NamedTuple, List, Tuple

# ---------------------- Global RNG ----------------------
GLOBAL_KEY = jax.random.PRNGKey(0)


def split_key(key):
    return jax.random.split(key)


# ---------------------- Optimizer -----------------------
class AdamState(NamedTuple):
    m: any
    v: any
    t: int


class AdamParams(NamedTuple):
    lr: float
    beta1: float
    beta2: float
    eps: float


def adam_init(params):
    # params is pytree of arrays
    m = tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    v = tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    return AdamState(m=m, v=v, t=0)


@jax.jit
def adam_update(params, grads, state: AdamState, meta: AdamParams):
    t = state.t + 1
    m = tree_util.tree_map(
        lambda m, g: meta.beta1 * m + (1 - meta.beta1) * g, state.m, grads
    )
    v = tree_util.tree_map(
        lambda v, g: meta.beta2 * v + (1 - meta.beta2) * (g**2), state.v, grads
    )
    m_hat = tree_util.tree_map(lambda m_: m_ / (1 - meta.beta1**t), m)
    v_hat = tree_util.tree_map(lambda v_: v_ / (1 - meta.beta2**t), v)
    new_params = tree_util.tree_map(
        lambda p, m_h, v_h: p + meta.lr * m_h / (jnp.sqrt(v_h) + meta.eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, AdamState(m=m, v=v, t=t)


# -------------------- Activations ----------------------
def relu(x):
    return jnp.maximum(x, 0)


def relu_d(x):
    return jnp.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-jnp.clip(x, -20, 20)))


def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)


def softplus(x):
    return jnp.log1p(jnp.exp(jnp.clip(x, -30, 30)))


def softmax(x, temp=1.0):
    z = x / temp - jnp.max(x / temp, axis=-1, keepdims=True)
    e = jnp.exp(jnp.clip(z, -30, 30))
    return e / jnp.sum(e, axis=-1, keepdims=True)


# ----------------------- Layer --------------------------
class LayerParams(NamedTuple):
    w: jnp.ndarray
    b: jnp.ndarray
    inv_var: jnp.ndarray


class LayerOptState(NamedTuple):
    w_adam: AdamState
    b_adam: AdamState


class LayerState(NamedTuple):
    values: jnp.ndarray
    new_values: jnp.ndarray
    w_trace: jnp.ndarray
    b_trace: jnp.ndarray


class LayerMeta(NamedTuple):
    l_type: int
    temp: float
    params: LayerParams
    opt: LayerOptState
    state: LayerState


LS_REAL = {0, 1, 2, 3}  # softplus, relu, linear, sigmoid


def init_layer(rng, in_dim, out_dim, var, temp, l_type, lr):
    rng, key = split_key(rng)
    lim = jnp.sqrt(2 / (in_dim + out_dim))
    w = jax.random.uniform(key, (in_dim, out_dim), -lim, lim)
    b = jnp.zeros((out_dim,))
    inv_var = jnp.full((out_dim,), 1 / var)
    params = LayerParams(w=w, b=b, inv_var=inv_var)
    w_adam = adam_init(params.w)
    b_adam = adam_init(params.b)
    opt = LayerOptState(w_adam=w_adam, b_adam=b_adam)
    zero = jnp.zeros((1, out_dim))
    trace_w = jnp.zeros((1, in_dim, out_dim))
    trace_b = jnp.zeros((1, out_dim))
    state = LayerState(values=zero, new_values=zero, w_trace=trace_w, b_trace=trace_b)
    meta = LayerMeta(l_type=l_type, temp=temp, params=params, opt=opt, state=state)
    return rng, meta


def compute_pot_mean(
    params: LayerParams, inputs: jnp.ndarray, l_type: int, temp: float
):
    pot = inputs @ params.w + params.b
    if l_type in LS_REAL:
        mean = [softplus, relu, lambda x: x, sigmoid][l_type](pot)
    else:
        mean = softmax(pot, temp)
    return pot, mean


@jax.jit
def layer_sample(
    rng, meta: LayerMeta, inputs: jnp.ndarray
) -> Tuple[jnp.ndarray, LayerMeta, jax.random.KeyArray]:
    pot, mean = compute_pot_mean(meta.params, inputs, meta.l_type, meta.temp)
    if meta.l_type in LS_REAL:
        sigma = jnp.sqrt(1 / meta.params.inv_var)
        rng, key = split_key(rng)
        vals = mean + sigma * jax.random.normal(key, pot.shape)
    else:
        rng, key = split_key(rng)
        vals = jax.random.categorical(key, jnp.log(mean), axis=-1)
    meta = meta._replace(state=meta.state._replace(values=vals, new_values=vals))
    return vals, meta, rng


@jax.jit
def layer_map_ascent(
    meta: LayerMeta,
    inputs: jnp.ndarray,
    next_pot_mean_inv: Tuple[jnp.ndarray, jnp.ndarray],
    update_size: float,
) -> LayerMeta:
    # unpack
    l_type, temp, params, opt_state, state = meta
    pot, mean = compute_pot_mean(params, inputs, l_type, temp)
    # compute new_values
    if next_pot_mean_inv is None:
        # output layer
        if l_type in LS_REAL:
            sigma = jnp.sqrt(1 / params.inv_var)
            vals = mean + sigma * jax.random.normal(GLOBAL_KEY, pot.shape)
        else:
            vals = jax.random.categorical(GLOBAL_KEY, jnp.log(mean), axis=-1)
    else:
        next_pot, next_mean, next_inv_var, next_ltype = next_pot_mean_inv
        lower = (mean - state.values) * params.inv_var
        if next_ltype in LS_REAL:
            upper = (
                (state.new_values - next_mean)
                * [relu_d, softplus, lambda x: 1, sigmoid_d][next_ltype](next_pot)
                * next_inv_var
            ) @ params.w.T
        else:
            upper = ((state.new_values - next_mean) / temp) @ params.w.T
        vals = state.values + update_size * (lower + upper)
    new_state = state._replace(new_values=vals)
    return meta._replace(state=new_state)


# ---------------------- Network ------------------------
class NetParams(NamedTuple):
    layers: List[LayerMeta]


@jax.jit
def net_forward(params: NetParams, state, inputs):
    rng = state["rng"]
    metas = state["metas"]
    h = inputs
    new_metas = []
    for meta in metas:
        h, meta, rng = layer_sample(rng, meta, h)
        new_metas.append(meta)
    new_state = {"rng": rng, "metas": new_metas}
    return new_state, h


@jax.jit
def net_map_ascent(params: NetParams, state, inputs, update_sizes: List[float]):
    # one step across all layers
    metas = state["metas"]
    pots_means = []
    # first compute pot/mean for all
    h = inputs
    for meta in metas:
        pot, mean = compute_pot_mean(meta.params, h, meta.l_type, meta.temp)
        pots_means.append((pot, mean, meta.params.inv_var, meta.l_type))
        h = meta.state.new_values
    # update each layer
    new_metas = []
    for i, meta in enumerate(metas):
        next_info = pots_means[i + 1] if i + 1 < len(metas) else None
        meta = layer_map_ascent(
            meta,
            pots_means[i][1].shape and inputs
            if i == 0
            else metas[i - 1].state.new_values,
            next_info,
            update_sizes[i],
        )
        new_metas.append(meta)
    new_state = {"rng": state["rng"], "metas": new_metas}
    return new_state


import argparse, configparser, os, sys, json, random
import numpy as np
import jax, jax.numpy as jnp
from jax import value_and_grad

# Assume all functions and classes (AdamParams, init_layer, NetParams, net_forward, net_map_ascent, etc.)
# are imported or defined above.


def pure_complex_multiplexer(batch_size, addr_size=5, action_size=1):
    # generate random inputs and compute true y, returns x, y
    key = jax.random.PRNGKey(random.randint(0, 2**31 - 1))
    x = jax.random.bernoulli(
        key, 0.5, (batch_size, addr_size * action_size + 2**addr_size)
    ).astype(jnp.int32)
    # compute y as before
    x_reshaped = x[:, : addr_size * action_size].reshape(
        batch_size, action_size, addr_size
    )
    weights = 2 ** (addr_size - 1 - jnp.arange(addr_size))
    addr = jnp.sum(x_reshaped * weights, axis=2)
    indices = addr_size * action_size + addr
    y = x[jnp.arange(batch_size), indices[:, 0]]
    return x, y


def pure_mnist_batch(batch_size, dataset, ptr, epoch_idx):
    # dataset: (X, y) arrays
    X, y = dataset
    if ptr + batch_size > X.shape[0]:
        epoch_idx = (epoch_idx + 1) % X.shape[0]
        ptr = 0
    batch_x = X[ptr : ptr + batch_size]
    batch_y = y[ptr : ptr + batch_size]
    return batch_x, batch_y, ptr + batch_size, epoch_idx


@jax.jit
def from_one_hot(p):
    return jnp.argmax(p, axis=-1)


def zero_to_neg(x):
    return (x > 1e-8).astype(jnp.float32) - (x <= 1e-8).astype(jnp.float32)


# main integrating pure net_forward and net_map_ascent
def main():
    # parse config (omitted for brevity, same as user provided)
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument(
        "-c",
        "--config",
        default="config_mn.ini",
        help="Location of config file (default: config_mp.ini)",
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
        description="Script with configurable parameters via config file and command-line flags.",
    )

    parser.add_argument(
        "--name",
        default=config.get("DEFAULT", "name"),
        help="Name identifier for the run.",
    )
    parser.add_argument(
        "--exp_num", type=int, default=1, help="Experiment number to help with tracking"
    )
    parser.add_argument(
        "--max_eps",
        type=int,
        default=config.getint("DEFAULT", "max_eps"),
        help="Number of episodes per run.",
    )
    parser.add_argument(
        "--n_run",
        type=int,
        default=config.getint("DEFAULT", "n_run"),
        help="Number of runs.",
    )

    parser.add_argument(
        "--env_name",
        default=config.get("DEFAULT", "env_name"),
        choices=["Multiplexer", "Regression", "MNIST"],
        help="Environment name (e.g., Multiplexer, Regression).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.getint("DEFAULT", "batch_size"),
        help="Batch size.",
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default=config.get("DEFAULT", "hidden"),
        help="JSON list of hidden units per layer (e.g., '[64, 32]').",
    )
    parser.add_argument(
        "--print_every", type=int, default=config.get("DEFAULT", "print_every")
    )
    parser.add_argument(
        "--l_type",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=config.getint("DEFAULT", "l_type"),
        help="Activation function type: 0=Softplus, 1=ReLU, 2=Linear, 3=Sigmoid, 4=Discrete.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=config.getfloat("DEFAULT", "temp"),
        help="Temperature for the network if applicable.",
    )
    parser.add_argument(
        "--var",
        type=str,
        default=config.get("DEFAULT", "var"),
        help="JSON list of variances in hidden layers (e.g., '[0.3, 1, 1]').",
    )
    parser.add_argument(
        "--update_adj",
        type=float,
        default=config.getfloat("DEFAULT", "update_adj"),
        help="Step size for minimizing the energy of the network equals to the layer's variance multiplied by this constant",
    )
    parser.add_argument(
        "--map_grad_ascent_steps",
        type=int,
        default=config.getint("DEFAULT", "map_grad_ascent_steps"),
        help="Number of gradient ascent steps for energy minimization.",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default=config.get("DEFAULT", "lr"),
        help="JSON list of learning rates (e.g., '[0.04, 0.00004, 0.000004]').",
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

    batch_size = args.batch_size
    hidden = json.loads(args.hidden)  # e.g. [64, 32]
    var = json.loads(args.var)  # one variance per hidden layer
    lr_list = json.loads(args.lr)  # one lr per layer (if you want per‐layer)
    temp = args.temp
    l_types = [args.l_type] * len(hidden) + [
        4
    ]  # e.g. same hidden type, discrete output
    update_adj = args.update_adj

    # compute per-layer MAP‐Prop step sizes
    update_sizes = [v * update_adj for v in var] + [update_adj]

    # now initialize the network
    rng = jax.random.PRNGKey(random.randint(0, 2**31 - 1))
    layers_meta = []
    env = pure_complex_multiplexer(batch_size=batch_size)
    in_dim = env.x_size  # input dimension (state size)
    for idx, out_dim in enumerate(hidden + [env.action_n]):
        this_var = var[idx]  # variance for this layer
        this_ltype = l_types[idx]
        this_lr = lr_list[idx]  # if you want per-layer learning‐rate in Adam
        rng, meta = init_layer(
            rng,
            in_dim=in_dim,
            out_dim=out_dim,
            var=this_var,
            temp=temp,
            l_type=this_ltype,
            lr=this_lr,
        )
        layers_meta.append(meta)
        in_dim = out_dim  # next layer’s input is this layer’s output

    # bundle into your NetParams and state
    net_params = NetParams(layers=layers_meta)
    net_state = {"rng": rng, "metas": layers_meta}

    # create a matching Adam state for the whole network
    net_opt_state = adam_init(net_params)

    # training loop skeleton
    for epoch in range(1):
        x, y = pure_complex_multiplexer(batch_size)
        net_state, action_probs = net_forward(net_params, net_state, x)
        actions = zero_to_neg(from_one_hot(action_probs))[:, None]
        rewards = (actions.flatten() == y).astype(jnp.float32)

        # MAP-Prop
        net_state = net_map_ascent(net_params, net_state, x, update_sizes)
        # REINFORCE (sketch)
        # compute loss, grads, update params
        loss = -jnp.mean(rewards)
        grads = jax.grad(lambda p: -jnp.mean(rewards))(net_params)
        net_params, _ = adam_update(
            net_params, grads, net_opt_state, AdamParams(lr_meta, 0.9, 0.999, 1e-9)
        )

        # periodic print
        print(f"Loss: {loss}")


if __name__ == "__main__":
    main()
