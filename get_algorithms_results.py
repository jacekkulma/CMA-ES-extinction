import numpy as np
from utils import custom_cma_es, custom_cma_es_ext, sphere, rastrigin, rosenbrock, ackley, cma_es_cma
import os

algorithms = [cma_es_cma, custom_cma_es, custom_cma_es_ext]

## output folder
output = "tests_results"

## parameters
dims = [5, 10, 20, 30, 50, 100]
sigma = 0.3
population_size = 20
max_iter = 1000
extinction_threshold = 10
extinction_rate_worst = 0.4
extinction_rate_best = 0.1

test_functions = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Ackley": ackley
}

seeds = [42, 1234, 987654, 20240406, 777, 314159, 8675309, 99999, 1337, 55555]

## tests
def main() -> None:
    for dim in dims:
        for name, func in test_functions.items():   
            # create output dir
            dir = os.path.join(output, f"dim_{dim}", f"{name}")
            os.makedirs(dir, exist_ok=True)
            for algorithm in algorithms:
                function_values = []
                best_solutions = []
                for seed in seeds:
                    np.random.seed(seed)
                    x0 = np.random.uniform(-5, 5, size=dim)
                    os.makedirs(os.path.join(dir, "iters"), exist_ok=True)
                    file_path = os.path.join(dir, "iters", f"{algorithm.__name__}_iters_seed_{seed}.txt")
                    result = algorithm(function=func, x0=x0, sigma=sigma, population_size=population_size, max_iter=max_iter,
                                       seed=seed, output_file=file_path, iter_threshold=100, extinction_threshold=extinction_threshold,
                                       extinction_rate_worst=extinction_rate_worst, extinction_rate_best=extinction_rate_best)
                    final_value = func(result)
                    
                    function_values.append(final_value)
                    best_solutions.append(result)

                avg_value = np.mean(function_values)
                avg_best_solution = np.mean(best_solutions, axis=0)

                # save results for algorithm, dim and function
                file_path = os.path.join(dir, f"{algorithm.__name__}_overview.txt")
                with open(file_path, "w") as f:
                    f.write(f"algorithm: {algorithm.__name__}, dimension: {dim}, function: {name}, iters: {max_iter}\n")
                    f.write(f"avg obj func value: {avg_value:.6f}\n")
                    f.write(f"obj func values: {np.round(function_values, 6)}\n")
                    f.write(f"avg best solution: {np.round(avg_best_solution, 6)}\n")
                    f.write(f"best solutions: {np.round(best_solutions, 6)}")
                print(f"Results - algorithm: {algorithm.__name__}, dimension: {dim}, function: {name} - saved.")

    print("\nALL DONE.")

if __name__ == "__main__":
    main()