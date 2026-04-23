import numpy as np
import pandas as pd
from opfunu.cec_based.cec2017 import (
    F12017, F22017, F32017, F42017, F52017,
    F62017, F72017, F82017, F92017, F102017
)

from algorithm import adaptive_bee_optimization_live


# CONFIG
DIMENSION = 30
BOUNDS = [-100, 100]
NUM_BEES = 300
MAX_ITER = 500
RUNS = 4 # increase to 30 later

FUNCTIONS = [
    F12017, F22017, F32017, F42017, F52017,
    F62017, F72017, F82017, F92017, F102017
]


# OBJECTIVE (BIAS-REMOVED)
def make_objective(func):
    f_opt = func.f_global  # true optimum (100, 200, ..., 1000)

    def obj(x):
        raw = func.evaluate(x)
        normalized = raw - f_opt  # 🔥 shift so optimum = 0
        return [normalized]

    return obj


# MAIN BENCHMARK LOOP
results_summary = []

for F in FUNCTIONS:
    print(f"\n==============================")
    print(f"Running {F.__name__}")
    print(f"==============================")

    func = F(ndim=DIMENSION)
    f_opt = func.f_global

    print(f"Original optimum (bias): {f_opt}")
    print("After normalization → optimum = 0")

    run_results = []

    for run in range(RUNS):

        cec_objective = make_objective(func)

        best_pos, best_fit, history = adaptive_bee_optimization_live(
            objective_functions=[cec_objective],
            min_bounds=[BOUNDS[0]] * DIMENSION,
            max_bounds=[BOUNDS[1]] * DIMENSION,
            num_bees=NUM_BEES,
            max_iterations=MAX_ITER
        )

        # now best_fit SHOULD be near 0 if good
        error = abs(best_fit)

        run_results.append(error)

        print(
            f"Run {run+1:02d} | "
            f"Best fitness (normalized) = {best_fit:.6e} | "
            f"Error = {error:.6e}"
        )

    # STATS
    mean = np.mean(run_results)
    std = np.std(run_results)
    best = np.min(run_results)
    worst = np.max(run_results)

    results_summary.append({
        "Function": F.__name__,
        "Mean": mean,
        "Std": std,
        "Best": best,
        "Worst": worst
    })

    print(f"\n{F.__name__} Summary:")
    print(f"Mean  = {mean:.6e}")
    print(f"Std   = {std:.6e}")
    print(f"Best  = {best:.6e}")
    print(f"Worst = {worst:.6e}")


# SAVE RESULTS
df = pd.DataFrame(results_summary)
df.to_csv("cec_results_normalized.csv", index=False)

print("\n==============================")
print("FINAL RESULTS TABLE")
print("==============================")
print(df)