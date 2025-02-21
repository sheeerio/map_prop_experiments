import argparse
import configparser
import os
import sys
import json
import numpy as np
import gym
from util import *
from mapprop import Network

def main():
    # Step 1: Initial parser to get config file path
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument(
        "-c", "--config",
        default="config_mp.ini",
        help="Location of config file (default: config_mp.ini)"
    )
    args, remaining_argv = initial_parser.parse_known_args()

    # Construct full config file path
    config_dir = "config"
    f_name = os.path.join(config_dir, args.config)
    print(f"Loading config from {f_name}")

    # Read the config file
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

    # Task parameters
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

    # Parse all arguments
    args = parser.parse_args()

    # Convert JSON strings to Python lists
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

    # Define constants for activation functions
    L_SOFTPLUS = 0
    L_RELU = 1
    L_LINEAR = 2
    L_SIGMOID = 3
    L_DISCRETE = 4

    # Initialize environment based on env_name
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

    elif args.env_name == "Regression":
        env = reg_MDP()
        gate = True
        output_l_type = L_LINEAR
        action_n = 1

    else:
        print(f"Error: Unsupported environment '{args.env_name}'.")
        sys.exit(1)

    update_size = [i * args.update_adj for i in var]
    print_every = 128 * 5

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
                action = zero_to_neg(from_one_hot(action))[:, np.newaxis]
                reward = env.act(action)[:, 0]
            elif args.env_name == "Regression":
                action = action[:, 0]
                reward = env.act(action)
            eps_ret_hist.append(np.average(reward))

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
        break
        eps_ret_hist_full.append(eps_ret_hist)

    eps_ret_hist_full = np.asarray(eps_ret_hist_full, dtype=np.float32)
    print("Finished Training")

    curves = {}
    curves[args.name] = (eps_ret_hist_full,)
    names = {k: k for k in curves.keys()}
    
    # Ensure the plots directory 
    plots_dir = os.path.join(f"result/exp{args.exp_num}", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    result_dir = f"result/exp{args.exp_num}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.name}.npy")
    print(f"Results (saved to {result_file}):")
    np.save(result_file, curves)
    # print_stat(curves, names)
    
    # Define the filename for the plot based on the run name
    plot_filename = f"{args.name}_plot.png"
    
    # Call the updated plot function with the save parameters
    # plot(
    #     curves, 
    #     names, 
    #     mv_n=10, 
    #     end_n=args.max_eps, 
    #     save=True, 
    #     save_dir=plots_dir, 
    #     filename=plot_filename
    # )


if __name__ == "__main__":
    main()