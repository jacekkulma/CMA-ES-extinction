import numpy as np
import math

#BASIC CMA-ES
def custom_cma_es(function, x0, sigma, population_size, max_iter=1000):
    dim = len(x0)
    mu = math.floor(population_size / 2)

    weights = np.log(mu + 0.5) - np.log(np.arange(1,mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1 / np.sum(weights**2)

    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff+2) / (dim+ mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

    pc = np.zeros(dim)
    ps = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    C = B @ np.diag(D**2) @ B.T
    inv_sqrt_C = B @ np.diag(D**-1) @ B.T
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    decomp_time = 0
    xmean = np.array(x0)

    for generation in range(max_iter):
        arz = np.random.randn(population_size, dim)
        arx = xmean + sigma * (arz @ B @ np.diag(D))

        fitness = np.array([function(x) for x in arx])
        idx = np.argsort(fitness)
        arx = arx[idx]
        arz = arz[idx]

        xold = xmean.copy()
        xmean = np.dot(weights, arx[:mu])
        y = (xmean - xold) / sigma
        z = np.dot(inv_sqrt_C, y)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z
        expected_norm = np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) * chiN
        threshold = 1.4 + 2 / (dim + 1)
        hsig = int(np.linalg.norm(ps) < expected_norm * threshold)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y
        artmp = (arz[:mu]).T * weights
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu * (artmp @ artmp.T)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        if generation - decomp_time > population_size / (c1 + cmu) / dim / 10:
            decomp_time = generation
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            inv_sqrt_C = B @ np.diag(D**-1) @ B.T

        if sigma < 1e-12 or np.allclose(xmean, xold, atol=1e-12):
            break

    return xmean

#CMA-ES WITH EXTINCTION MECHANISM
def custom_cma_es(function, x0, sigma, population_size, max_iter=1000):
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
    inv_sqrt_C = B @ np.diag(D**(-1)) @ B.T
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    decomp_time = 0
    xmean = np.array(x0)
    best_fitness = float('inf')
    no_improvement_time = 0
    extinction_threshold = 6

    for generation in range(max_iter):
        arz = np.random.randn(population_size, dim)
        arx = xmean + sigma * (arz @ B @ np.diag(D))

        fitness = np.array([function(x) for x in arx])
        idx = np.argsort(fitness)
        arx = arx[idx]
        arz = arz[idx]

        if no_improvement_time >= extinction_threshold:
            extinct_rate_worst = 0.5
            extinct_rate_best = 0.02
            num_extinct_worst = int(extinct_rate_worst * population_size)
            num_extinct_best = int(extinct_rate_best * population_size)

            extinct_indices = (
                list(range(0, num_extinct_best)) +
                list(range(population_size - num_extinct_worst, population_size))
            )

            for ei in extinct_indices:
                parent_idx = np.random.choice(idx[:mu])
                parent = arx[parent_idx]
                mutant = parent + sigma * np.random.randn(dim)
                arx[ei] = mutant
                arz[ei] = inv_sqrt_C @ ((mutant - xmean) / sigma)

            no_improvement_time = 0

        if fitness[idx[0]] < best_fitness:
            best_fitness = fitness[idx[0]]
            no_improvement_time = 0
        else:
            no_improvement_time += 1

        xold = xmean.copy()
        xmean = np.dot(weights, arx[:mu])
        y = (xmean - xold) / sigma
        z = np.dot(inv_sqrt_C, y)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z
        expected_norm = np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) * chiN
        threshold = 1.4 + 2 / (dim + 1)
        hsig = int(np.linalg.norm(ps) < expected_norm * threshold)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y
        artmp = (arz[:mu]).T * weights
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu * (artmp @ artmp.T)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        if generation - decomp_time > population_size / (c1 + cmu) / dim / 10:
            decomp_time = generation
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            inv_sqrt_C = B @ np.diag(D ** -1) @ B.T

        if sigma < 1e-12 or np.allclose(xmean, xold, atol=1e-12):
            break

    return xmean