import numpy as np
from utils import custom_cma_es, custom_cma_es_ext, sphere, rastrigin, rosenbrock, ackley, run_cma_es_cma

algorithms = [run_cma_es_cma, custom_cma_es, custom_cma_es_ext]

## parameters
dim = 10
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
for algorithm in algorithms:
    print(f"Running algorithm: {algorithm.__name__}\n")
    for name, func in test_functions.items():
        print(f" Testing function: {name}")
        for seed in seeds:
            print(f"random seed: {seed}")
            x0 = np.random.uniform(-5, 5, size=dim)
            result = algorithm(func, x0, sigma, population_size, max_iter)
            final_value = func(result)
            print(f"   Best solution: {result}")
            print(f"   Objective function value: {final_value:.6f}\n")