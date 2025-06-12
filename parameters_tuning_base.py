import numpy as np
from utils import custom_cma_es_ext, sphere, rastrigin, rosenbrock, ackley
import os

## output folder
output = "parameters_tuning"
os.makedirs(output, exist_ok=True)

## static parameters
dims = [5, 10, 20, 30, 50, 100]
max_iter = 1000

## parametrs for grid search
sigma_values = [0.1, 0.3, 0.5, 0.7, 1.0]
population_sizes = [10, 20, 50, 100]

test_functions = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Ackley": ackley
}

seeds = [42, 1234, 987654, 20240406, 777, 314159, 8675309, 99999, 1337, 55555]

def main():
    results = []

    for dim in dims:
        for name, func in test_functions.items():
            for sigma in sigma_values:
                for population_size in population_sizes:
                    avg_values = []

                    for seed in seeds:
                        np.random.seed(seed)
                        x0 = np.random.uniform(-5, 5, size=dim)

                        result = custom_cma_es_ext(func, x0, sigma, population_size, max_iter, seed)
                        final_value = func(result)

                        avg_values.append(final_value)

                    avg_value = np.mean(avg_values)
                    results.append((dim, name, sigma, population_size, avg_value))

                    print(f"Done: dim={dim}, func={name}, sigma={sigma}, pop_size={population_size}, avg_value={avg_value:.6f}")

    # save result to file
    with open(os.path.join(output, "grid_search_results_base.txt"), "w") as f:
        f.write("dim,function,sigma,population_size,avg_obj_func_value\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f}\n")

if __name__ == "__main__":
    main()