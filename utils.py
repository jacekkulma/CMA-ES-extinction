import numpy as np
import math
import cma

def save_file_header(output_file):
    with open(output_file, "w") as f:
            f.write("iteration,best_fitness\n")

def save_stats(output_file, iteration, best_fitness):
    with open(output_file, "a") as f:
                f.write(f"{iteration + 1},{best_fitness:.6f}\n")


## built-in cma-es
def cma_es_cma(problem, x0, sigma, population_size, max_iter, seed=None, output_file=None, iter_threshold=1, **kwargs):
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': population_size, 'maxiter': max_iter,
                                              'verb_log': 0, 'verb_disp': 0, 'seed': seed})
    # file header
    if output_file:
        save_file_header(output_file)

    iteration = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [problem(x) for x in solutions]
        es.tell(solutions, fitnesses)

        iteration += 1

        # save to file
        if output_file and (iteration + 1) % iter_threshold == 0:
            best_fitness = min(fitnesses)
            save_stats(output_file, iteration, best_fitness)

    result = es.result.xbest
    return result


## basic cma-es
def custom_cma_es(function, x0, sigma, population_size, max_iter=100, seed=None, output_file=None, iter_threshold=1, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    dim = len(x0)
    mu = math.floor(population_size / 2)

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1 / np.sum(weights**2)

    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c_sig = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sig

    pc = np.zeros(dim)
    ps = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    C = B @ np.diag(D**2) @ B.T
    inv_sqrt_C = B @ np.diag(D**-1) @ B.T

    eigeneval = 0
    xmean = np.array(x0)

    # file header
    if output_file:
        save_file_header(output_file)

    for generation in range(max_iter):
        std_matrix = np.random.randn(population_size, dim)
        x_matrix = xmean + sigma * (std_matrix @ B @ np.diag(D))

        fitness = np.array([function(x) for x in x_matrix])
        idx = np.argsort(fitness)
        x_matrix = x_matrix[idx]
        std_matrix = std_matrix[idx]

        xold = xmean.copy()
        xmean = np.dot(weights, x_matrix[:mu])

        y = (xmean - xold) / sigma
        z = inv_sqrt_C @ y

        ps = (1 - c_sig) * ps + np.sqrt(c_sig * (2 - c_sig) * mu_eff) * z
        hsig = int(
            np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sig) ** (2 * (generation + 1))) / np.sqrt(dim)
            < (1.4 + 2 / (dim + 1))
        )

        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y
        artmp = std_matrix[:mu].T * weights
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * (artmp @ artmp.T)
        )

        sigma *= np.exp((c_sig / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

        if generation - eigeneval > population_size / (c1 + cmu) / dim / 10:
            eigeneval = generation
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            inv_sqrt_C = B @ np.diag(D**-1) @ B.T

        # save to file
        if output_file and (generation + 1) % iter_threshold == 0:
            best_fitness = fitness[0]
            save_stats(output_file, generation, best_fitness)

    return xmean


## cma-es with extinction mechanism
def custom_cma_es_ext(function, x0, sigma, population_size, max_iter=100, seed=None, extinction_threshold=10,
                      extinction_rate_worst=0.4, extinction_rate_best=0.1, output_file=None, iter_threshold=1, **kwargs):
    if seed is not None:
            np.random.seed(seed)

    dim = len(x0)
    mu = math.floor(population_size / 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1 / np.sum(weights**2)

    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

    pc = np.zeros(dim)
    ps = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    C = B @ np.diag(D**2) @ B.T
    inv_sqrt_C = B @ np.diag(D**-1) @ B.T
    eigeneval = 0
    xmean = np.array(x0)
    best_fitness = float('inf')
    no_improvement_count = 0

    # file header
    if output_file:
        save_file_header(output_file)

    for generation in range(max_iter):
        arz = np.random.randn(population_size, dim)
        arx = xmean + sigma * (arz @ B @ np.diag(D))

        # --- Extinction strategy ---
        if no_improvement_count >= extinction_threshold:
            num_extinct_worst = int(extinction_rate_worst * population_size)
            num_extinct_best = int(extinction_rate_best * population_size)

            fitness = np.array([function(ind) for ind in arx])
            idx = np.argsort(fitness)
            arx = arx[idx]
            arz = arz[idx]

            extinct_indices = (
                list(range(0, num_extinct_best)) +
                list(range(population_size - num_extinct_worst, population_size))
            )

            survivors = [i for i in range(population_size) if i not in extinct_indices]

            mutation_strength = sigma
            for ei in extinct_indices:
                parent_idx = np.random.choice(survivors)
                parent = arx[parent_idx]
                mutant = parent + mutation_strength * np.random.randn(dim)
                arx[ei] = mutant
                arz[ei] = (mutant - xmean) / sigma @ np.linalg.inv(B @ np.diag(D))

            no_improvement_count = 0
        else:
            fitness = np.array([function(ind) for ind in arx])
            idx = np.argsort(fitness)
            arx = arx[idx]
            arz = arz[idx]

        if fitness[idx[0]] < best_fitness:
            best_fitness = fitness[idx[0]]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        xold = xmean.copy()
        xmean = np.dot(weights, arx[:mu])
        y = (xmean - xold) / sigma
        z = np.dot(inv_sqrt_C, y)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z
        hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / np.sqrt(dim) < (1.4 + 2 / (dim + 1)))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y
        artmp = (arz[:mu]).T * weights
        C = (1 - c1 - cmu) * C + \
            c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu * (artmp @ artmp.T)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

        if generation - eigeneval > population_size / (c1 + cmu) / dim / 10:
            eigeneval = generation
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            inv_sqrt_C = B @ np.diag(D ** -1) @ B.T

        # save to file
        if output_file and (generation + 1) % iter_threshold == 0:
            best_fitness = fitness[0]
            save_stats(output_file, generation, best_fitness)

    return xmean

## test functions
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1)