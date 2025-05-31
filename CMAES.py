import numpy as np
import cocoex
import math


def custom_cma_es(function, x0, sigma, population_size, max_iter=100):
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

    return xmean


# --- Funkcje testowe ---
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


# --- Parametry testowe ---
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

# --- Wykonanie testów ---
for name, func in test_functions.items():
    print(f" Test funkcji: {name}")
    x0 = np.random.uniform(-5, 5, size=dim)
    result = custom_cma_es(func, x0, sigma, population_size, max_iter)
    final_value = func(result)
    print(f"   Najlepsze rozwiązanie: {result}")
    print(f"   Wartość funkcji celu: {final_value:.6f}")

