import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from algorithm import *
from animation import demo_animation
import json
import pandas as pd 
import os

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

    noise = np.random.normal(0, 0.01 + 0.02*np.abs(state[0] - 1))

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

        chaos_drive = (
            0.2*np.sin(3*H)*np.cos(2*N) +
            0.15*np.sin(7*H + 0.5*N) +
            0.1*np.cos(11*N) +
            0.05*np.sin(20*H*N)
        )

        dH = (
            k_hr*(1 - H)
            - 0.7*M
            + 0.4*np.tanh(R)
            - 0.3*H**3
            + 0.15*chaos_drive
        )

        dR = (
            k_oxygen*(H - R)
            + 0.25*np.sin(5*R)
            - 0.1*N
        )

        dN = (
            k_neural*(R - 0.8)
            - 1.0*(N-0.5)*(N-1.5)*(N-1.0)
            + 0.35*np.sin(4*N)
        )

        dM = (
            k_metabolic*(1.2 - R)
            + 0.15*np.abs(N-1)
            + 0.1*np.cos(3*M)
        )

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
    metabolic_balance = abs(np.mean(metabolic_trace) - 0.9)
    metabolic_instability = np.var(metabolic_trace)
    oxygen_deficit = np.mean(np.maximum(0, 1 - resp_trace))

    # Stability diagnostics printing (research logging only)



    # Return multi-objective vector
    return np.array([
        heart_var,
        neural_var,
        metabolic_instability,
        oxygen_deficit,
        metabolic_balance
        ])

def robustness_under_stress(params, shock_magnitude=0.3):

    k_hr, k_oxygen, k_neural, k_metabolic = params

    H = 1.0
    R = 1.0
    N = 1.0
    M = 1.0

    dt = 0.05
    T = 200

    heart_trace = []
    resp_trace = []
    neural_trace = []
    metabolic_trace = []

    for t in range(T):

        # Apply shock at midpoint
        if t == int(T/2):
            H += np.random.uniform(-shock_magnitude, shock_magnitude)
            R += np.random.uniform(-shock_magnitude, shock_magnitude)

        dH = k_hr*(1 - H) - 0.5*M + 0.3*R - 0.2*H**3
        dR = k_oxygen*(H - R)
        dN = k_neural*(R - 0.8) - 0.8*(N - 0.6)*(N - 1.4)*(N - 1.0)
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

    # Robustness metrics

    recovery_time_penalty = np.mean(np.abs(heart_trace[-50:] - 1))

    shock_response_variance = np.var(
        heart_trace[int(T/2):]
    )

    oxygen_violation = np.mean(np.maximum(0, 1 - resp_trace))

    return np.array([
        np.var(heart_trace),
        np.var(neural_trace),
        np.var(metabolic_trace),
        oxygen_violation,
        recovery_time_penalty + shock_response_variance
    ])


#test graph 
def plot_search_landscape(model_func, resolution=120, avg_runs=3):

    import numpy as np
    import matplotlib.pyplot as plt

    # Parameter ranges
    x = np.linspace(0, 2, resolution)
    y = np.linspace(0, 2, resolution)

    X, Y = np.meshgrid(x, y)

    # Allocate surfaces
    Z_total = np.zeros_like(X)
    Z_heart = np.zeros_like(X)
    Z_oxygen = np.zeros_like(X)

    # Averaged evaluation (handles stochasticity)
    def evaluate_avg(params):
        vals = []
        objs = []

        for _ in range(avg_runs):
            obj = model_func(params)
            objs.append(obj)
            vals.append(np.mean(obj))  # scalar projection

        return np.mean(vals), np.mean(objs, axis=0)

    # Evaluate landscape
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            params = np.array([
                X[i, j],   # k_hr
                Y[i, j],   # k_oxygen
                1.0,       # fixed
                1.0        # fixed
            ])

            val, obj = evaluate_avg(params)

            Z_total[i, j] = val
            Z_heart[i, j] = obj[0]        # heart variance
            Z_oxygen[i, j] = obj[3]       # oxygen deficit

    # Log scaling (reveals subtle variation)
    Z_total = np.log1p(Z_total)
    Z_heart = np.log1p(Z_heart)
    Z_oxygen = np.log1p(Z_oxygen)

    # Plot multiple surfaces
    fig = plt.figure(figsize=(18, 5))

    # --- Total Fitness ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z_total, cmap='viridis')
    ax1.set_title("Scalar Fitness Landscape (Log Scaled)")
    ax1.set_xlabel("k_hr")
    ax1.set_ylabel("k_oxygen")

    # --- Heart Variance ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z_heart, cmap='plasma')
    ax2.set_title("Heart Variance Landscape")
    ax2.set_xlabel("k_hr")
    ax2.set_ylabel("k_oxygen")

    # --- Oxygen Deficit ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z_oxygen, cmap='inferno')
    ax3.set_title("Oxygen Deficit Landscape")
    ax3.set_xlabel("k_hr")
    ax3.set_ylabel("k_oxygen")

    plt.tight_layout()
    plt.show()
plot_search_landscape(robustness_under_stress)

#saving code

def save_experiment_results(run_id, best_pos, best_fit, history, folder="results"):
    
    os.makedirs(folder, exist_ok=True)

    # Evaluate full objective vector (IMPORTANT)
    true_obj = robustness_under_stress(best_pos)

    summary_file = os.path.join(folder, "summary.csv")

    summary_data = pd.DataFrame([{
        "run": run_id,
        "seed": run_id,
        "best_fitness": best_fit,

        # Parameters
        "k_hr": best_pos[0],
        "k_oxygen": best_pos[1],
        "k_neural": best_pos[2],
        "k_metabolic": best_pos[3],

        # Objective breakdown (CRITICAL FOR RESEARCH)
        "heart_var": true_obj[0],
        "neural_var": true_obj[1],
        "metabolic_var": true_obj[2],
        "oxygen_deficit": true_obj[3],
        "recovery_penalty": true_obj[4]
    }])

    if os.path.exists(summary_file):
        summary_data.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        summary_data.to_csv(summary_file, index=False)

    # Save convergence curve
    history_file = os.path.join(folder, f"history_run_{run_id}.csv")

    history_df = pd.DataFrame({
        "iteration": np.arange(len(history)),
        "fitness": history
    })

    history_df.to_csv(history_file, index=False)


def save_metadata():

    os.makedirs("results", exist_ok=True)

    metadata = {
        "num_bees": num_bees,
        "max_iterations": max_iterations,
        "runs": runs,
        "bounds": {
            "min": min_bounds,
            "max": max_bounds
        },
        "algorithm": "Adaptive Bee Optimization",
        "objective": "robustness_under_stress"
    }

    with open("results/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


#plot generation

def generate_research_plots(histories):
    import numpy as np
    import matplotlib.pyplot as plt

    histories = np.array(histories)

    # --- Aggregate stats ---
    mean_curve = np.mean(histories, axis=0)
    std_curve = np.std(histories, axis=0)
    iterations = np.arange(len(mean_curve))

    # 1. LOG-SCALE CONVERGENCE
    plt.figure(figsize=(7,4))
    plt.plot(iterations, mean_curve, label="Mean Fitness")
    plt.fill_between(iterations,
                     mean_curve - std_curve,
                     mean_curve + std_curve,
                     alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("Convergence with Confidence Interval")
    plt.legend()
    plt.show()

    # 2. IMPROVEMENT PER ITERATION
    improvement = -np.diff(mean_curve)

    plt.figure(figsize=(7,4))
    plt.plot(improvement)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Improvement")
    plt.title("Improvement per Iteration")
    plt.show()

    # 3. STAGNATION ANALYSIS
    stagnation = [0]

    for i in range(1, len(mean_curve)):
        if mean_curve[i] < mean_curve[i-1]:
            stagnation.append(0)
        else:
            stagnation.append(stagnation[-1] + 1)

    plt.figure(figsize=(7,4))
    plt.plot(stagnation)
    plt.xlabel("Iteration")
    plt.ylabel("Stagnation Length")
    plt.title("Stagnation Over Time")
    plt.show()

    # 4. CUMULATIVE IMPROVEMENT
    cumulative = mean_curve[0] - mean_curve

    plt.figure(figsize=(7,4))
    plt.plot(iterations, cumulative)
    plt.xlabel("Iteration")
    plt.ylabel("Total Improvement")
    plt.title("Cumulative Improvement")
    plt.show()

    # 5. ROLLING VARIANCE 
    window = 10
    rolling_var = [
        np.var(mean_curve[max(0, i-window):i+1])
        for i in range(len(mean_curve))
    ]

    plt.figure(figsize=(7,4))
    plt.plot(rolling_var)
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.title("Rolling Variance of Fitness")
    plt.show()


#benchmarking code
if __name__ == "__main__":

    dims = 5

    min_bounds = [0, 0, 0, 0]
    max_bounds = [2, 2, 2, 2]

    num_bees = 200
    max_iterations = 300
    runs = 20
    
    


    def run_experiment():

        results = []
        histories = []

        for seed in range(runs):

            print(f"Running trial {seed+1}/{runs}")

            best_pos, best_fit, history = adaptive_bee_optimization_live(
                objective_functions=[robustness_under_stress],
                min_bounds=min_bounds,
                max_bounds=max_bounds,
                num_bees=num_bees,
                max_iterations=max_iterations,
                archive_size=10, #10 -12
                seed=seed
            )

            results.append(best_fit)
            histories.append(history)

            save_experiment_results(
                run_id=seed,
                best_pos=best_pos,
                best_fit=best_fit,
                history=history
            )

        return np.array(results), np.array(histories)

    print("\n===== Portfolio Benchmark Run =====")

    results, histories = run_experiment()

    print("\nBenchmark Summary")
    print("--------------------")
    print("Mean Fitness :", np.mean(results))
    print("Std Fitness  :", np.std(results))
    print("Best Fitness :", np.min(results))
    print("Worst Fitness:", np.max(results))



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

    save_metadata()
    print("Results saved to /results folder")

    generate_research_plots(histories)

