import numpy as np
import math

def custom_cma_es(function, x0, sigma, population_size, max_iter=100):
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
    extinction_threshold = 10  # liczba generacji bez poprawy

    for generation in range(max_iter):
        arz = np.random.randn(population_size, dim)
        arx = xmean + sigma * (arz @ B @ np.diag(D))

        # --- Extinction strategy ---
        if no_improvement_count >= extinction_threshold:
            extinction_rate_worst = 0.4
            extinction_rate_best = 0.1
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

    return xmean

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1)

# --- Parametry wspólne ---
dim = 10
sigma = 0.3
population_size = 20
max_iter = 100

# --- Lista funkcji testowych ---
test_functions = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Ackley": ackley
}

# --- Testy ---
for name, func in test_functions.items():
    print(f"\n Test funkcji: {name}")
    x0 = np.random.uniform(-5, 5, size=dim)
    result = custom_cma_es(func, x0, sigma, population_size, max_iter)
    final_value = func(result)
    print(f"Najlepsze rozwiązanie: {result}")
    print(f"Wartość funkcji celu: {final_value:.6f}")