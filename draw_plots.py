import numpy as np
import os
from get_algorithms_results import algorithms, test_functions, dims, output
import matplotlib.pyplot as plt
import glob

plots_folder = "plots"

def draw_ecdf_lines_plots():
    """
    Generates ECDF plots for different algorithms on each test function and dimension.
    Plots multiple ECDF curves on one graph to compare algorithm performance.
    """
    os.makedirs(os.path.join(plots_folder, "ecdf"), exist_ok=True)
    for dim in dims:
        for name, func in test_functions.items():
            plt.figure(figsize=(8, 6))

            for algorithm in algorithms:
                file_path = os.path.join(output, f"dim_{dim}", f"{name}", f"{algorithm.__name__}_overview.txt")

                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist, skipping...")
                    continue

                with open(file_path, "r") as f:
                    lines = f.readlines()

                # Extract objective function values from the file
                for line in lines:
                    if line.startswith("obj func values:"):
                        values_str = line.split(":")[1].strip().strip("[]")
                        values = np.array([float(x) for x in values_str.split()])
                        break
                else:
                    print(f"Objective function values not found for {algorithm.__name__}")
                    continue

                # Compute ECDF (Empirical Cumulative Distribution Function)
                sorted_values = np.sort(values)
                ecdf_y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

                # Plot ECDF curve for the algorithm
                plt.step(sorted_values, ecdf_y, where="post", label=algorithm.__name__, linewidth=2)

            # Set plot labels and title
            plt.xlabel("Objective Function Value")
            plt.ylabel("Empirical Cumulative Distribution")
            plt.title(f"ECDF for {dim}D {name}")
            plt.legend()
            plt.grid()

            # Save the plot to a file
            plot_path = os.path.join(plots_folder, "ecdf", f"dim_{dim}_{name}.png")      
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved: {plot_path}")

def draw_iters_plots(max_iters=800, iter_threshold=20):
    """
    Reads iteration logs for each algorithm, computes the average best fitness per iteration,
    and plots all algorithms on a single graph for comparison.
    """
    os.makedirs(os.path.join(plots_folder, "iters"), exist_ok=True)

    for dim in dims:
        for name, func in test_functions.items():
            dir_path = os.path.join(output, f"dim_{dim}", f"{name}", "iters")
            plt.figure(figsize=(8, 6))

            for algorithm in algorithms:
                # Get all files that start with the algorithm's name
                files = glob.glob(os.path.join(dir_path, f"{algorithm.__name__}_iters_seed_*.txt"))

                if not files:
                    print(f"No files found for {algorithm.__name__} in {dir_path}, skipping...")
                    continue

                all_data = []

                for file in files:
                    data = []
                    with open(file, "r") as f:
                        lines = f.readlines()[1:]  # Skip header

                    # Read iteration numbers and best fitness values
                    for line in lines:
                        iter_num, best_fitness = line.strip().split(",")
                        data.append((int(iter_num), float(best_fitness)))

                    # Ensure at least 10 rows, fill missing with last available value
                    if data:
                        iter_entries = max_iters / iter_threshold
                        while len(data) < iter_entries:
                            data.append((data[-1][0] + iter_threshold, data[-1][1]))
                        all_data.append(data)
                    else:
                        print(f"Skipping empty file: {file}")

                # Ensure at least one valid dataset
                if not all_data:
                    print(f"No valid data for {algorithm.__name__}, skipping plot...")
                    continue

                # Convert list to NumPy array for averaging
                all_data = np.array(all_data)  # Shape: (num_files, num_iters, 2)
                avg_values = np.mean(all_data[:, :, 1], axis=0)
                iter_nums = all_data[0, :, 0]  # Iteration numbers (assumed same for all runs)

                # Plot average best fitness for this algorithm
                plt.plot(iter_nums, avg_values, label=algorithm.__name__, linewidth=2)

            # Configure plot labels and title
            plt.xlabel("Iteration")
            plt.ylabel("Average Best Fitness")
            plt.title(f"Optimization Progress - {dim}D {name}")
            plt.legend()
            plt.grid()

            # Save plot
            plot_path = os.path.join(plots_folder, "iters", f"dim_{dim}_{name}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    draw_ecdf_lines_plots()
    draw_iters_plots()

