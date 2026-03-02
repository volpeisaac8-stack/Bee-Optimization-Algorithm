import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from algorithm import *


# Synthetic Multi-Organ Homeostasis Sandbox Model

STATE_DIM = 5

initial_system_state = np.array([
    0.7,
    0.3,
    0.5,
    0.6,
    0.8
])


# Diversity stability metric (not true biological entropy)
def system_diversity_metric(state):
    return np.mean(np.std(state))




def simulate_organ_signals(control, steps=30):

    state = initial_system_state.copy()

    cardiac_hist = []
    resp_hist = []
    neural_hist = []
    metabolic_hist = []

    for _ in range(steps):

        state[0] = cardiovascular_layer(state, control)
        state[2] = metabolic_layer(state, control)
        state[3] = neural_layer(state, control)
        state[4] = respiratory_layer(state, control)

        cardiac_hist.append(state[0])
        metabolic_hist.append(state[2])
        neural_hist.append(state[3])
        resp_hist.append(state[4])

    return (
        np.mean(cardiac_hist),
        np.mean(resp_hist),
        np.mean(neural_hist),
        np.mean(metabolic_hist)
    )

# Organ Dynamical Layers

def cardiovascular_layer(state, control):

    rhythm = state[0]

    noise = np.random.normal(0, 0.008)

    rhythm = (
        0.97 * rhythm
        - 0.02 * state[1]
        + 0.03 * control[0]
        + noise
    )

    return np.clip(rhythm, 0, 1)


def metabolic_layer(state, control):

    glucose = state[2]

    noise = np.random.normal(0, 0.008)

    glucose = (
        0.96 * glucose
        - 0.04 * control[1]
        + noise
    )

    return np.clip(glucose, 0, 1)


def neural_layer(state, control):

    neural = state[3]

    noise = np.random.normal(0, 0.008)

    neural = (
        0.98 * neural
        - 0.01 * np.var(state)
        + 0.02 * control[0]
        + noise
    )

    return np.clip(neural, 0, 1)


def respiratory_layer(state, control):

    resp = state[4]

    noise = np.random.normal(0, 0.008)

    resp = (
        0.97 * resp
        + 0.03 * control[1]
        - 0.02 * state[0]
        + noise
    )

    return np.clip(resp, 0, 1)



# Pareto Archive

pareto_archive = []


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def update_archive(solution, fitness_vector):

    global pareto_archive

    new_archive = []

    for s, f in pareto_archive:
        if dominates(fitness_vector, f):
            continue
        new_archive.append((s, f))

    new_archive.append((solution, fitness_vector))

    pareto_archive = new_archive[:50]


# Multi-Objective Projection Fitness

import numpy as np

def multi_organ_coupled_system(params):

    # Unpack parameters (what swarm searches for)
    k_hr, k_oxygen, k_neural, k_metabolic = params

    # Initial states
    H = 1.0      # heart strength
    R = 1.0      # respiratory oxygen
    N = 1.0      # neural stability
    M = 1.0      # metabolic load

    dt = 0.05
    T = 200

    heart_trace = []
    resp_trace = []
    neural_trace = []
    metabolic_trace = []

    for _ in range(T):

        # Coupled dynamics

        dH = k_hr*(1 - H) - 0.3*M + 0.2*R
        dR = k_oxygen*(H - R)
        dN = k_neural*(R - 0.8) - 0.4*(N-1)**3
        dM = k_metabolic*(1.2 - R) + 0.1*abs(N-1)

        H += dt*dH
        R += dt*dR
        N += dt*dN
        M += dt*dM

        heart_trace.append(H)
        resp_trace.append(R)
        neural_trace.append(N)
        metabolic_trace.append(M)

    heart_trace = np.array(heart_trace)
    resp_trace = np.array(resp_trace)
    neural_trace = np.array(neural_trace)
    metabolic_trace = np.array(metabolic_trace)

    # Stability metrics

    heart_var = np.var(heart_trace)
    neural_var = np.var(neural_trace)
    metabolic_mean = np.mean(metabolic_trace)
    oxygen_deficit = np.mean(np.maximum(0, 1 - resp_trace))

    # Return multi-objective vector
    return np.array([
        heart_var,
        neural_var,
        metabolic_mean,
        oxygen_deficit
        ])



#benchmarking code

if __name__ == "__main__":

    dims = 4

    min_bounds = [0, 0, 0, 0]
    max_bounds = [2, 2, 2, 2]

    num_bees = 120
    max_iterations = 200
    runs = 30

    def run_experiment():

        results = []
        histories = []

        for seed in range(runs):

            print(f"Running trial {seed+1}/{runs}")

            best_pos, best_fit, history = adaptive_bee_optimization_live(
                objective_functions=[multi_organ_coupled_system],
                min_bounds=min_bounds,
                max_bounds=max_bounds,
                num_bees=num_bees,
                max_iterations=max_iterations,
                archive_size=10,
                seed=seed
            )

            results.append(best_fit)
            histories.append(history)

        return np.array(results), np.array(histories)

    print("\n===== Portfolio Benchmark Run =====")

    results, histories = run_experiment()

    print("\nBenchmark Summary")
    print("--------------------")
    print("Mean Fitness :", np.mean(results))
    print("Std Fitness  :", np.std(results))
    print("Best Fitness :", np.min(results))
    print("Worst Fitness:", np.max(results))

    # =====================================================
    # Convergence Visualization with Confidence Interval
    # =====================================================

    avg_curve = np.mean(histories, axis=0)

    ci_upper = []
    ci_lower = []

    for i in range(len(avg_curve)):

        sample = histories[:, i]

        mean = np.mean(sample)
        std = np.std(sample)

        margin = 1.96 * std / np.sqrt(len(sample))

        ci_upper.append(mean + margin)
        ci_lower.append(mean - margin)

    plt.figure(figsize=(7, 4))

    plt.plot(avg_curve, label="BOA Multi-Organ Model")

    plt.fill_between(
        range(len(avg_curve)),
        ci_lower,
        ci_upper,
        alpha=0.2
    )

    plt.yscale("log")

    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("Convergence Performance")

    plt.legend()
    plt.show()