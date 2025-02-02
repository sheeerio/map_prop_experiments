import numpy as np
import os
from util import plot
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_num",
        type=int,
        default=1,
        help="Experiment number to help with tracking"
    )

    args = parser.parse_args()
    npy_dir = f"./result/exp{args.exp_num}/"
    save_dir = f"./result/exp{args.exp_num}/plots/"
    filename = f"combined_plot_exp{args.exp_num}.png"
    
    # Load all .npy files
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
    # npy_files = [f for f in os.listdir([x for x in os.listdir('./result') if x.startswith("exp")]) if f.endswith(".npy")]
    
    curves = {}
    names = {}
    
    # Populate the curves and names dictionaries
    for i, npy_file in enumerate(npy_files):
        file_path = os.path.join(npy_dir, npy_file)
        data = np.load(file_path, allow_pickle=True)  # Load .npy file
        name = npy_file.replace(".npy", "")
        print(f"{npy_file}: {data.item()[name][0].shape}, type={type(data)}")
        curve_key = f"curve_{i}"  # Unique key for each curve
        curves[curve_key] = data.item()[name]
        names[curve_key] = name
    
    # Plot the curves
    plot(
        curves=curves,
        names=names,
        mv_n=100,  # Moving average window
        end_n=10000,  # Number of episodes to consider
        xlabel="Episodes",
        ylabel="Running Average Return",
        ylim=None,
        loc=4,
        save=True,
        save_dir=save_dir,
        filename=filename
    )

if __name__ == "__main__":
    main()