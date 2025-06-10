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
max_iter = 100

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
                    result = algorithm(func, x0, sigma, population_size, max_iter, seed)
                    final_value = func(result)
                    
                    function_values.append(final_value)
                    best_solutions.append(result)

                avg_value = np.mean(function_values)
                avg_best_solution = np.mean(best_solutions, axis=0)

                # save results for algorithm, dim and function (avg from all seeds)
                file_path = os.path.join(dir, f"{algorithm.__name__}.txt")
                with open(file_path, "w") as f:
                    f.write(f"algorithm: {algorithm.__name__}, dimension: {dim}, function: {name}\n")
                    f.write(f"{avg_value:.6f}\n")
                    f.write(f"{np.round(avg_best_solution, 6)}\n")
                print(f"Results - algorithm: {algorithm.__name__}, dimension: {dim}, function: {name} - saved.")

    print("\nALL DONE.")

if __name__ == "__main__":
    main()