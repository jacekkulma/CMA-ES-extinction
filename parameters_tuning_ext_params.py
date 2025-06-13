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
extinction_thresholds = [5, 10, 20]
extinction_rate_worst_values = [0.2, 0.4, 0.6]
extinction_rate_best_values = [0.05, 0.1, 0.2]


test_functions = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Ackley": ackley
}

## from parameters_tuning_base
optimal_params = {
    5: {"sigma": 0.3, "population_size": 20},
    10: {"sigma": 0.3, "population_size": 20},
    20: {"sigma": 0.5, "population_size": 50},
    30: {"sigma": 0.5, "population_size": 50},
    50: {"sigma": 0.7, "population_size": 100},
    100: {"sigma": 0.7, "population_size": 100}
}

seeds = [42, 1234, 987654, 20240406, 777, 314159, 8675309, 99999, 1337, 55555]

def main():
    results = []

    for dim in dims:
        for name, func in test_functions.items():
            sigma = optimal_params[dim]["sigma"]
            population_size = optimal_params[dim]["population_size"]

            for extinction_threshold in extinction_thresholds:
                for extinction_rate_worst in extinction_rate_worst_values:
                    for extinction_rate_best in extinction_rate_best_values:
                        avg_values = []

                        for seed in seeds:
                            np.random.seed(seed)
                            x0 = np.random.uniform(-5, 5, size=dim)

                            result = custom_cma_es_ext(
                                func, x0, sigma, population_size, max_iter, seed,
                                extinction_threshold=extinction_threshold,
                                extinction_rate_worst=extinction_rate_worst,
                                extinction_rate_best=extinction_rate_best
                            )
                            final_value = func(result)

                            avg_values.append(final_value)

                        avg_value = np.mean(avg_values)
                        results.append((dim, name, sigma, population_size, extinction_threshold, extinction_rate_worst, extinction_rate_best, avg_value))

                        print(f"Done: dim={dim}, func={name}, sigma={sigma}, pop_size={population_size}, "
                              f"ext_threshold={extinction_threshold}, worst={extinction_rate_worst}, best={extinction_rate_best}, "
                              f"avg_value={avg_value:.6f}")

    # save result to file
    with open(os.path.join(output, "grid_search_results_ext_params.txt"), "w") as f:
        f.write("dim,function,sigma,population_size,ext_threshold,ext_rate_worst,ext_rate_best,avg_obj_func_value\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]:.6f}\n")

if __name__ == "__main__":
    main()