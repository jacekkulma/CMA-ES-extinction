import numpy as np
import cocoex
from utils import custom_cma_es, custom_cma_es_ext, run_cma_es_cma

## folder -> algorithm dict
algorithms = [
    ("cma_es", run_cma_es_cma),
    ("custom_cma", custom_cma_es),
    ("custom_cma_ext", custom_cma_es_ext),
]

## parameters
sigma = 0.3
population_size = 20
max_iter = 100

## random seed and x0
np.random.seed(42)
x0 = np.random.uniform(-5, 5, size=10)

## testing
for folder_name, algorithm in algorithms:
    print(f"Running algorithm: {algorithm.__name__}\'name")

    ## coco settings
    observer = cocoex.Observer("bbob", f"result_folder: {folder_name}")
    suite = cocoex.Suite("bbob", "", "dimensions:10 function_indices:1-5")
    for problem in suite:
        print(f"Testing function: {problem.id}")

        problem.observe_with(observer)
        solution = algorithm(problem, x0, sigma, population_size, max_iter)
        fitness = problem(solution)

        print(f"  Best solution: {np.round(solution, 4)}")
        print(f"  Objective function value: {fitness:.6f}\n")


