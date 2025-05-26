import cocoex
import cma

def run_optimizer_on_problem(problem):
    dim = problem.dimension
    x0 = [0.5] * dim
    sigma0 = 0.3

    # COCO convension: budget is 100 * n (n = dimension)
    max_evals = 100 * dim
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxfevals': max_evals,
        'verb_log': 0,
        'verb_disp': 0
    })

    evals_done = 0
    while not es.stop() and evals_done < max_evals:
        solutions = es.ask()
        fitnesses = [problem(x) for x in solutions]
        es.tell(solutions, fitnesses)
        evals_done += len(solutions)

    print(f"Best solution: {es.result.xbest}, fitness: {es.result.fbest}")

def main():
    suite = cocoex.Suite("bbob", "", "dimensions:5 function_indices:1-5 instance_indices:1-3")
    observer = cocoex.Observer("bbob", "result_folder: CMAES_Classic")

    for problem in suite:
        problem.observe_with(observer)
        run_optimizer_on_problem(problem)

if __name__ == "__main__":
    main()
