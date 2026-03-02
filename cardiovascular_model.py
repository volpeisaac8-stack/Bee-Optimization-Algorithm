import numpy as np
from opfunu.cec_based.cec2014 import F12014, F82014, F32014, F102014
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import warnings
from algorithm import *
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initial synthetic cardiovascular state
# [rhythm_signal, variability_proxy, vascular_resistance]
initial_cardiovascular_state = np.array([
    0.7,
    0.2,
    0.3
])

def cardiac_rhythm_layer(state, control):

    rhythm = state[0]
    variability = state[1]

    alpha = 0.97
    beta = 0.02

    noise = np.random.normal(0, 0.01)

    next_rhythm = (
        alpha * rhythm
        - beta * variability
        + 0.03 * control[0]
        + noise
    )

    return np.clip(next_rhythm, 0, 1)

def vascular_layer(state, control):

    resistance = state[2]

    resistance = (
        0.95 * resistance
        + 0.02 * control[1]
        + np.random.normal(0, 0.01)
    )

    return np.clip(resistance, 0, 1)

def cardiovascular_test_fitness(control_parameters):

    state = initial_cardiovascular_state.copy()

    projection_steps = 40
    total_risk_score = 0

    for _ in range(projection_steps):

        state[0] = cardiac_rhythm_layer(state, control_parameters)
        state[2] = vascular_layer(state, control_parameters)

        # Physiological penalties

        rhythm_variance_penalty = np.var(state)
        energy_cost = np.sum(control_parameters ** 2)

        total_risk_score += rhythm_variance_penalty + 0.05 * energy_cost

    return total_risk_score



if __name__ == "__main__":

    # Cardiovascular control model uses 2 parameters
    dims = 2

    min_bounds = [0]*dims
    max_bounds = [1]*dims

    num_bees = 200
    max_iterations = 300
    
    # # of
    runs = 10

    objective_functions = [cardiovascular_test_fitness]

    def run_trials(archive_size):

        results = []
        histories = []

        for r in range(runs):

            print(f"Run {r+1}/{runs} | Archive size = {archive_size}")

            best_pos, best_fit, history = adaptive_bee_optimization_live(
                objective_functions,
                min_bounds,
                max_bounds,
                num_bees=num_bees,
                max_iterations=max_iterations,
                switch_iterations=10**9,
                archive_size=archive_size,
                seed=r
            )

            results.append(best_fit)
            histories.append(history)

        return np.array(results), np.array(histories)

    print("\n--- Running WITHOUT memory ---")
    results_no_mem, histories_no_mem = run_trials(archive_size=0)

    print("\n--- Running WITH memory ---")
    results_mem, histories_mem = run_trials(archive_size=10)

    print("\n==============================")
    print("WITHOUT Memory:")
    print("Mean:", np.mean(results_no_mem))
    print("Std :", np.std(results_no_mem))
    print("Best:", np.min(results_no_mem))
    print("Worst:", np.max(results_no_mem))

    print("\nWITH Memory:")
    print("Mean:", np.mean(results_mem))
    print("Std :", np.std(results_mem))
    print("Best:", np.min(results_mem))
    print("Worst:", np.max(results_mem))

    avg_no_mem = np.mean(histories_no_mem, axis=0)
    avg_mem = np.mean(histories_mem, axis=0)

    plt.figure()
    plt.plot(avg_no_mem, label="No Memory")
    plt.plot(avg_mem, label="With Memory")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (avg)")
    plt.title("BOA Convergence Comparison")
    plt.legend()

    plt.show(block=False)
    plt.close()