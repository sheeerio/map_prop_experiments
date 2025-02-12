import argparse
import configparser
import os
import sys
import json
import jax.numpy as jnp
from util_jax import *
from jax_mapprop import *

@partial(jax.jit, static_argnames=('env', 'env_name', 'batch_size', 'map_grad_ascent_steps'))
def train_step(carry, _, env, env_name, batch_size, map_grad_ascent_steps, update_size, lr):
    """
    A single training step.

    Args:
      carry: Tuple (net, key)
      _: dummy scan index (unused)
      env: the environment (marked static)
      env_name: string, either "Multiplexer" or "Regression" (static)
      batch_size: int (static)
      map_grad_ascent_steps: int (static)
      update_size: list of floats
      lr: learning rate (or list) for learning update
      
    Returns:
      A tuple: ((updated_net, updated_key), avg_reward)
    """
    net, key = carry

    # Split key and reset the environment:
    key, subkey = jax.random.split(key)
    state = env.reset(subkey, batch_size)
    
    # Forward pass through the network:
    net, action, key = forward_network(key, net, state)
    
    # Process the network’s output into an action and get the reward:
    if env_name == "Multiplexer":
        # Convert one-hot output to integer, then map 0 to 0 and 1 to 1.
        action_int = jax_from_one_hot(action)
        # Remove extra conversion—just use the integer directly:
        action_processed = action_int[:, None]
        reward = env.act(action_processed)[:, 0]
    else:  # Regression
        action = action[:, 0]
        reward = env.act(action)
    
    # Compute average reward for logging:
    avg_reward = jnp.mean(reward)
    
    # Perform the MAP gradient ascent update:
    net, key = map_grad_ascent(key, net, steps=map_grad_ascent_steps, update_size=update_size)
    
    # For Regression, recompute the reward based on the network’s final output:
    if env_name == "Regression":
        reward = env.y - net.layers[-1].mean[:, 0]
    
    # Apply learning (update network weights):
    net = learn_network(net, reward, lr=lr)
    
    return (net, key), avg_reward

# Now, define a function that runs many training steps using jax.lax.scan:
def train_loop(net, key, env, env_name, batch_size, map_grad_ascent_steps, update_size, lr, steps):
    """
    Run the training loop for a given number of steps.
    
    Args:
      net: initial network
      key: initial PRNG key
      env: environment (static)
      env_name: string identifier (static)
      batch_size: batch size (static)
      map_grad_ascent_steps: number of MAP ascent steps (static)
      update_size: update size list for each layer
      lr: learning rate (or list) for weight updates
      steps: number of training steps
      
    Returns:
      Updated network, key, and an array of average rewards per step.
    """
    # The lambda wraps train_step, passing the static arguments.
    scan_fn = lambda carry, _: train_step(carry, _, env, env_name, batch_size, map_grad_ascent_steps, update_size, lr)
    (net, key), rewards = jax.lax.scan(scan_fn, (net, key), None, length=steps)
    return net, key, rewards


def main():
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config",default="config_mp.ini",
    help="Location of config file (default: config_mp.ini)")
    args, _ = initial_parser.parse_known_args()

    config_dir = "../config"
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

    # key
    key = jax.random.PRNGKey(args.key)

    # Initialize environment based on env_name
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
    print_every = 128 * 5
    optimizer = jax_adam_optimizer(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    eps_ret_hist_full = []
    for j in range(args.n_run):
        net, key = init_network(
            key=key,
            state_n=env.x_size,
            action_n=action_n,
            optimizer=optimizer,
            hidden=hidden,
            var=var,
            temp=args.temp,
            hidden_l_type=args.l_type,
            output_l_type=output_l_type
        )

        eps_ret_hist = []
        print_count = print_every
        steps = args.max_eps // args.batch_size  # number of training steps
        net, key, rewards = train_loop(
            net, key, env, args.env_name, args.batch_size,
            args.map_grad_ascent_steps, update_size, lr, steps
        )

        rewards_np = jnp.array(rewards)
        print("Finished Training, average rewards per step:", rewards_np)
        
        eps_ret_hist.append(eps_ret_hist)

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
    print_stat(curves, names)
    
    # Define the filename for the plot based on the run name
    plot_filename = f"{args.name}_plot.png"
    
    # Call the updated plot function with the save parameters
    plot(
        curves, 
        names, 
        mv_n=10, 
        end_n=args.max_eps, 
        save=True, 
        save_dir=plots_dir, 
        filename=plot_filename
    )
        
if __name__ == "__main__":
    main()