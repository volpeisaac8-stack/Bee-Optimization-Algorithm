import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from algorithm import adaptive_bee_optimization_live, robustness_batch
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


def system_diversity_metric(state):
    return np.mean(np.std(state))


def simulate_organ_signals(control, steps=30):

    state = initial_system_state.copy()

    cardiac_hist    = []
    resp_hist       = []
    neural_hist     = []
    metabolic_hist  = []

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
    noise  = np.random.normal(0, 0.01 + 0.02 * np.abs(state[0] - 1))
    rhythm = (
        0.97 * rhythm
        - 0.02 * state[1]
        + 0.03 * control[0]
        + noise
    )
    return np.clip(rhythm, 0, 1)


def metabolic_layer(state, control):
    glucose = state[2]
    noise   = np.random.normal(0, 0.008)
    glucose = (
        0.96 * glucose
        - 0.04 * control[1]
        + noise
    )
    return np.clip(glucose, 0, 1)


def neural_layer(state, control):
    neural = state[3]
    noise  = np.random.normal(0, 0.008)
    neural = (
        0.98 * neural
        - 0.01 * np.var(state)
        + 0.02 * control[0]
        + noise
    )
    return np.clip(neural, 0, 1)


def respiratory_layer(state, control):
    resp  = state[4]
    noise = np.random.normal(0, 0.008)
    resp  = (
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
        # if existing solution dominates new one → discard new
        if dominates(f, fitness_vector):
            new_archive.append((s, f))

    # check if new solution is dominated by anything kept
    if not any(dominates(f, fitness_vector) for _, f in new_archive):
        new_archive.append((solution, fitness_vector))

    pareto_archive = new_archive

# Multi-Objective Projection Fitness (kept for reference / landscape plotting)

def multi_organ_coupled_system(params):

    k_hr, k_oxygen, k_neural, k_metabolic = params

    H = 1.0
    R = 1.0
    N = 1.0
    M = 1.0

    dt = 0.02
    T  = 100

    heart_trace     = []
    resp_trace      = []
    neural_trace    = []
    metabolic_trace = []

    for _ in range(T):

        chaos_drive = (
            0.2  * np.sin(3*H) * np.cos(2*N) +
            0.15 * np.sin(7*H + 0.5*N) +
            0.1  * np.cos(11*N) +
            0.05 * np.sin(20*H*N)
        )

        dH = (
            k_hr * (1 - H)
            - 0.7 * M
            + 0.4 * np.tanh(R)
            - 0.3 * np.tanh(H)
            + 0.15 * chaos_drive
        )

        OXYGEN_TARGET = 1.0

        dR = (
            k_oxygen * (OXYGEN_TARGET - R)   # restores oxygen toward baseline
            + 0.25 * (H - R)                 # coupling to heart
            - 0.05 * N                       # metabolic/neural consumption effect
        )

        dN = k_neural * (R - 0.8) - 0.5 * np.tanh(3*(N-1)) + 0.35 * np.sin(4*N)
        dM = k_metabolic * (1.2 - R) + 0.15 * np.abs(N-1) + 0.1 * np.cos(3*M)

        H += dt * dH
        R += dt * dR
        N += dt * dN
        M += dt * dM

        heart_trace.append(H)
        resp_trace.append(R)
        neural_trace.append(N)
        metabolic_trace.append(M)

    heart_trace     = np.array(heart_trace)
    resp_trace      = np.array(resp_trace)
    neural_trace    = np.array(neural_trace)
    metabolic_trace = np.array(metabolic_trace)

    heart_var            = np.var(heart_trace)
    neural_var           = np.var(neural_trace)
    metabolic_balance    = abs(np.mean(metabolic_trace) - 0.9)
    metabolic_instability = np.var(metabolic_trace)
    oxygen_deficit       = np.mean(np.maximum(0, 1 - resp_trace))

    return np.array([
        heart_var,
        neural_var,
        metabolic_instability,
        oxygen_deficit,
        metabolic_balance
    ])


#function
def robustness_under_stress(params, shock_magnitude=0.3):
    params = np.clip(params, 0, 2)

    
    N_bees = 1
    T = 200
    dt = 0.02
    shock_t = T // 2

    k_hr, k_oxygen, k_neural, k_metabolic = params

    H = np.ones(N_bees)
    R = np.ones(N_bees)
    N = np.ones(N_bees)
    M = np.ones(N_bees)

    heart_trace     = np.empty((T, N_bees))
    resp_trace      = np.empty((T, N_bees))
    neural_trace    = np.empty((T, N_bees))
    metabolic_trace = np.empty((T, N_bees))

    shock_H = np.random.uniform(-shock_magnitude, shock_magnitude, N_bees)
    shock_R = np.random.uniform(-shock_magnitude, shock_magnitude, N_bees)

    for t in range(T):
        if t == shock_t:
            H += shock_H
            R += shock_R

        dH = k_hr*(1 - H) - 0.5*M + 0.3*R - 0.2*H**3
        dR = k_oxygen*(H - R)
        dN = k_neural*(R - 0.8) - 0.8*(N - 0.6)*(N - 1.4)*(N - 1.0)
        dM = k_metabolic*(1.2 - R) + 0.1*np.abs(N - 1)

        H = np.clip(H + dt*dH, 0, 2)
        R = np.clip(R + dt*dR, 0, 2)
        N = np.clip(N + dt*dN, 0, 2)
        M = np.clip(M + dt*dM, 0, 2)

        heart_trace[t]     = H
        resp_trace[t]      = R
        neural_trace[t]    = N
        metabolic_trace[t] = M

    recovery_penalty   = np.mean(np.abs(heart_trace[-50:] - 1))
    shock_response_var = np.var(heart_trace[shock_t:])
    LOW = 0.85
    HIGH = 1.05

    oxygen_violation = np.mean(
        np.maximum(0, LOW - resp_trace) +
        np.maximum(0, resp_trace - HIGH)
    )

    # Returns full 5-element objective vector (needed for save_experiment_results)
    return np.array([
        np.var(heart_trace) * 5,        # heart stability
        np.var(neural_trace) * 5,       # neural stability
        np.var(metabolic_trace) * 5,    # metabolic stability
        oxygen_violation * 3,           # constraint violation
        recovery_penalty + shock_response_var  # combined robustness
    ])


# Search landscape (uses batched eval per row for speed)

def plot_search_landscape(resolution=60, avg_runs=2, fixed_k_neural=1.0, fixed_k_metabolic=1.0):


    print(f"Computing landscape ({resolution}x{resolution} grid, {avg_runs} avg runs)...")
    print(f"Fixed: k_neural={fixed_k_neural}, k_metabolic={fixed_k_metabolic}")

    x = np.linspace(0, 2, resolution)
    y = np.linspace(0, 2, resolution)
    X, Y = np.meshgrid(x, y)

    Z_total  = np.zeros_like(X)
    Z_heart  = np.zeros_like(X)
    Z_neural = np.zeros_like(X)
    Z_metabolic = np.zeros_like(X)
    Z_oxygen = np.zeros_like(X)
    Z_recovery = np.zeros_like(X)

    T       = 200
    dt      = 0.02
    shock_t = T // 2
    N_cols  = resolution

    for i in range(resolution):

        if i % 10 == 0:
            print(f"  Row {i}/{resolution}...")

        # Build full row of param combos — shape (resolution, 4)
        row_params = np.column_stack([
            X[i, :],
            Y[i, :],
            np.full(N_cols, fixed_k_neural),
            np.full(N_cols, fixed_k_metabolic)
        ])

        acc = np.zeros((5, N_cols))   # accumulator: [heart, neural, metabolic, oxygen, recovery]

        for _ in range(avg_runs):

            H = np.ones(N_cols)
            R = np.ones(N_cols)
            N = np.ones(N_cols)
            M = np.ones(N_cols)

            k_hr        = row_params[:, 0]
            k_oxygen    = row_params[:, 1]
            k_neural    = row_params[:, 2]
            k_metabolic = row_params[:, 3]

            ht = np.empty((T, N_cols))
            rt = np.empty((T, N_cols))
            nt = np.empty((T, N_cols))
            mt = np.empty((T, N_cols))

            shock_H = np.random.uniform(-0.3, 0.3, N_cols)
            shock_R = np.random.uniform(-0.3, 0.3, N_cols)

            for t_step in range(T):
                if t_step == shock_t:
                    H += shock_H
                    R += shock_R

                dH = k_hr*(1 - H) - 0.5*M + 0.3*R - 0.2*H**3
                dR = k_oxygen*(H - R)
                dN = k_neural*(R - 0.8) - 0.8*(N-0.6)*(N-1.4)*(N-1.0)
                dM = k_metabolic*(1.2 - R) + 0.1*np.abs(N - 1)

                H = np.clip(H + dt*dH, 0, 2)
                R = np.clip(R + dt*dR, 0, 2)
                N = np.clip(N + dt*dN, 0, 2)
                M = np.clip(M + dt*dM, 0, 2)

                ht[t_step] = H
                rt[t_step] = R
                nt[t_step] = N
                mt[t_step] = M

            acc[0] += np.var(ht, axis=0)
            acc[1] += np.var(nt, axis=0)
            acc[2] += np.var(mt, axis=0)
            acc[3] += np.mean(np.maximum(0, 1.05 - rt), axis=0)
            acc[4] += np.mean(np.abs(ht[-50:] - 1), axis=0) + np.var(ht[shock_t:], axis=0)

        acc /= avg_runs

        Z_heart[i, :]     = acc[0]
        Z_neural[i, :]    = acc[1]
        Z_metabolic[i, :] = acc[2]
        Z_oxygen[i, :]    = acc[3]
        Z_recovery[i, :]  = acc[4]
        Z_total[i, :]     = acc.sum(axis=0)

    # Log-scale all surfaces
    Zl_total    = np.log1p(Z_total)
    Zl_heart    = np.log1p(Z_heart)
    Zl_oxygen   = np.log1p(Z_oxygen)

    # Best point (lowest total fitness)
    best_flat  = np.argmin(Z_total)
    best_i, best_j = np.unravel_index(best_flat, Z_total.shape)
    best_x     = X[best_i, best_j]   # best k_hr
    best_y     = Y[best_i, best_j]   # best k_oxygen

    print(f"\nBest grid point → k_hr={best_x:.3f}, k_oxygen={best_y:.3f}, fitness={Z_total[best_i, best_j]:.5f}")

    # ── Plotting ──────────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"Search Landscape  |  k_neural={fixed_k_neural}, k_metabolic={fixed_k_metabolic}",
        fontsize=14, fontweight='bold', y=0.98
    )

    # Row 1: 3D surfaces
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(X, Y, Zl_total, cmap='viridis', linewidth=0, antialiased=True)
    ax1.set_title("Total Fitness (log)", fontsize=11)
    ax1.set_xlabel("k_hr"); ax1.set_ylabel("k_oxygen")

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(X, Y, Zl_heart, cmap='plasma', linewidth=0, antialiased=True)
    ax2.set_title("Heart Variance (log)", fontsize=11)
    ax2.set_xlabel("k_hr"); ax2.set_ylabel("k_oxygen")

    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot_surface(X, Y, Zl_oxygen, cmap='inferno', linewidth=0, antialiased=True)
    ax3.set_title("Oxygen Deficit (log)", fontsize=11)
    ax3.set_xlabel("k_hr"); ax3.set_ylabel("k_oxygen")

    # Row 2: 2D heatmaps with best-point marker
    cmap_opts = dict(origin='lower', extent=[0, 2, 0, 2], aspect='auto')

    ax4 = fig.add_subplot(234)
    im4 = ax4.imshow(Zl_total, cmap='viridis', **cmap_opts)
    ax4.scatter(best_x, best_y, c='red', s=80, zorder=5, label=f'Best ({best_x:.2f}, {best_y:.2f})')
    ax4.set_title("Total Fitness Heatmap", fontsize=11)
    ax4.set_xlabel("k_hr"); ax4.set_ylabel("k_oxygen")
    ax4.legend(fontsize=8)
    plt.colorbar(im4, ax=ax4)

    ax5 = fig.add_subplot(235)
    im5 = ax5.imshow(np.log1p(Z_heart), cmap='plasma', **cmap_opts)
    ax5.scatter(best_x, best_y, c='red', s=80, zorder=5)
    ax5.set_title("Heart Variance Heatmap", fontsize=11)
    ax5.set_xlabel("k_hr"); ax5.set_ylabel("k_oxygen")
    plt.colorbar(im5, ax=ax5)

    ax6 = fig.add_subplot(236)
    im6 = ax6.imshow(np.log1p(Z_oxygen), cmap='inferno', **cmap_opts)
    ax6.scatter(best_x, best_y, c='red', s=80, zorder=5)
    ax6.set_title("Oxygen Deficit Heatmap", fontsize=11)
    ax6.set_xlabel("k_hr"); ax6.set_ylabel("k_oxygen")
    plt.colorbar(im6, ax=ax6)

    plt.tight_layout()
    plt.show()

    return best_x, best_y, Z_total[best_i, best_j]


choice1 = input("Plot search landscape? (y/n): ")
if choice1 == "y":
    best_khr, best_koxy, best_fit_grid = plot_search_landscape(
        resolution=60,          # increase to 100 for publication quality
        avg_runs=2,             # increase to 5 for smoother averaging
        fixed_k_neural=1.0,
        fixed_k_metabolic=1.0
    )


# Saving code

def save_experiment_results(run_id, best_pos, best_fit, history, folder="results"):

    os.makedirs(folder, exist_ok=True)

    true_obj = robustness_under_stress(best_pos)

    summary_file = os.path.join(folder, "summary.csv")

    summary_data = pd.DataFrame([{
        "run":          run_id,
        "seed":         run_id,
        "best_fitness": best_fit,
        "k_hr":         best_pos[0],
        "k_oxygen":     best_pos[1],
        "k_neural":     best_pos[2],
        "k_metabolic":  best_pos[3],
        "heart_var":    true_obj[0],
        "neural_var":   true_obj[1],
        "metabolic_var":true_obj[2],
        "oxygen_deficit":true_obj[3],
        "recovery_penalty": true_obj[4]
    }])

    if os.path.exists(summary_file):
        summary_data.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        summary_data.to_csv(summary_file, index=False)

    history_file = os.path.join(folder, f"history_run_{run_id}.csv")
    history_df   = pd.DataFrame({
        "iteration": np.arange(len(history)),
        "fitness":   history
    })
    history_df.to_csv(history_file, index=False)


def save_metadata():

    os.makedirs("results", exist_ok=True)

    metadata = {
        "num_bees":      num_bees,
        "max_iterations": max_iterations,
        "runs":          runs,
        "bounds": {
            "min": min_bounds,
            "max": max_bounds
        },
        "algorithm":  "Adaptive Bee Optimization",
        "objective":  "robustness_under_stress"
    }

    with open("results/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


# Plot generation

def generate_research_plots(histories):

    histories  = np.array(histories)
    mean_curve = np.mean(histories, axis=0)
    std_curve  = np.std(histories,  axis=0)
    iterations = np.arange(len(mean_curve))

    plt.figure(figsize=(7, 4))
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

    improvement = -np.diff(mean_curve)
    plt.figure(figsize=(7, 4))
    plt.plot(improvement)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Improvement")
    plt.title("Improvement per Iteration")
    plt.show()

    stagnation = [0]
    for i in range(1, len(mean_curve)):
        if mean_curve[i] < mean_curve[i-1]:
            stagnation.append(0)
        else:
            stagnation.append(stagnation[-1] + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(stagnation)
    plt.xlabel("Iteration")
    plt.ylabel("Stagnation Length")
    plt.title("Stagnation Over Time")
    plt.show()

    cumulative = mean_curve[0] - mean_curve
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, cumulative)
    plt.xlabel("Iteration")
    plt.ylabel("Total Improvement")
    plt.title("Cumulative Improvement")
    plt.show()

    window     = 10
    rolling_var = [
        np.var(mean_curve[max(0, i-window):i+1])
        for i in range(len(mean_curve))
    ]
    plt.figure(figsize=(7, 4))
    plt.plot(rolling_var)
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.title("Rolling Variance of Fitness")
    plt.show()


# Interpretive data analysis

def interpret_experiment_results(csv_file="results/summary.csv", alpha=0.05):

    df = pd.read_csv(csv_file)

    print("\n" + "="*60)
    print("        EXPERIMENT INTERPRETATION REPORT")
    print("="*60 + "\n")

    # 1. BASIC PERFORMANCE SUMMARY
    print("1. PERFORMANCE SUMMARY")
    print("-" * 40)

    mean_fit = df["best_fitness"].mean()
    std_fit  = df["best_fitness"].std()
    best_fit = df["best_fitness"].min()
    worst_fit = df["best_fitness"].max()

    print(f"Mean best fitness : {mean_fit:.6f}")
    print(f"Std deviation     : {std_fit:.6f}")
    print(f"Best run fitness  : {best_fit:.6f}")
    print(f"Worst run fitness : {worst_fit:.6f}")

    ci = stats.t.interval(
        1 - alpha,
        len(df) - 1,
        loc=mean_fit,
        scale=std_fit / np.sqrt(len(df))
    )

    print(f"{int((1-alpha)*100)}% confidence interval: [{ci[0]:.6f}, {ci[1]:.6f}]\n")


    # 2. PARAMETER STABILITY
    print("2. PARAMETER STABILITY")
    print("-" * 40)

    params = ["k_hr", "k_oxygen", "k_neural", "k_metabolic"]

    for p in params:
        mean = df[p].mean()
        std  = df[p].std()

        print(f"{p:15s} | mean = {mean:.4f}, std = {std:.4f}")

    print("")


    # 3. OBJECTIVE BEHAVIOR
    print("3. OBJECTIVE BEHAVIOR")
    print("-" * 40)

    objectives = [
        "heart_var",
        "neural_var",
        "metabolic_var",
        "oxygen_deficit",
        "recovery_penalty"
    ]

    corr_matrix = df[objectives + ["best_fitness"]].corr()

    corr_with_fitness = corr_matrix["best_fitness"].drop("best_fitness")

    most_influential = corr_with_fitness.abs().idxmax()

    print("Correlation with fitness:")
    print(corr_with_fitness.round(3))
    print(f"\nMost influential objective (by |correlation|): {most_influential}\n")


    # 4. TRADE-OFF ANALYSIS
    print("4. TRADE-OFF STRUCTURE")
    print("-" * 40)

    tradeoffs_found = False

    for i in objectives:
        for j in objectives:
            if i != j:
                corr = corr_matrix.loc[i, j]
                if corr < -0.5:
                    print(f"Strong trade-off: {i} vs {j} (corr = {corr:.2f})")
                    tradeoffs_found = True

    if not tradeoffs_found:
        print("No strong negative trade-offs detected (corr < -0.5)")

    print("")


    # 5. CONSTRAINT BEHAVIOR
    print("5. CONSTRAINT SATISFACTION")
    print("-" * 40)

    oxygen_max = df["oxygen_deficit"].max()

    if oxygen_max == 0:
        print("Oxygen constraint: NEVER violated (inactive constraint)")
    else:
        viol_rate = np.mean(df["oxygen_deficit"] > 0)
        print(f"Oxygen constraint active")
        print(f"Violation rate: {viol_rate:.2%}")

    print("")


    # 6. BEST OBSERVED SOLUTION
    print("6. BEST OBSERVED SOLUTION")
    print("-" * 40)

    best_idx = df["best_fitness"].idxmin()
    best_row = df.loc[best_idx]

    for col in df.columns:
        if col != "run":
            print(f"{col:18s}: {best_row[col]}")

    print("\nInterpretation note:")
    print("This represents the best observed solution across independent runs,")
    print("not a guaranteed global optimum.\n")


    # 7. BEHAVIORAL SEGMENTATION
    print("7. BEHAVIORAL REGIMES")
    print("-" * 40)

    fast_recovery = df[df["recovery_penalty"] < df["recovery_penalty"].median()]
    stable        = df[df["heart_var"] < df["heart_var"].median()]
    robust        = df[df["best_fitness"] < df["best_fitness"].median()]

    print(f"Fast recovery systems: {len(fast_recovery)} / {len(df)}")
    print(f"Stable systems       : {len(stable)} / {len(df)}")
    print(f"High-performing runs : {len(robust)} / {len(df)}")

    print("\nOverlap suggests coupling between stability and recovery behavior.\n")


    # 8. FINAL SYNTHESIS
    print("8. SYNTHESIS")
    print("-" * 40)

    print(f"- Dominant driver of fitness: {most_influential}")
    print("- System exhibits stochastic variability across runs")

    if tradeoffs_found:
        print("- Evidence of multi-objective trade-offs present")
    else:
        print("- Weak evidence of strong trade-offs at chosen thresholds")

    print("- Parameter distributions suggest convergent control regime")
    print("- Results should be interpreted as empirical, not theoretical optimum")
    print("\n" + "="*60)

#scalar fitness
def scalar_fitness(params):
        obj = robustness_under_stress(params)

        oxygen_deficit = obj[3]

        mean_ref = np.array([0.02, 0.02, 0.02, 0.05, 0.05])
        std_ref  = np.array([0.01, 0.01, 0.01, 0.02, 0.02])

        normalized = np.abs((obj - mean_ref) / std_ref)

        weights = np.array([1.0, 1.0, 1.0, 0.5, 1.0])

        base_fitness = np.sum(weights * normalized)

        # smooth constraint penalty
        oxygen_penalty = 1000 * max(0.0, oxygen_deficit - 0.08)

        return base_fitness + oxygen_penalty

# Benchmarking

if __name__ == "__main__":

    min_bounds = [0, 0, 0, 0]
    max_bounds = [2, 2, 2, 2]

    num_bees       = 200
    max_iterations = int(input("How many iterations: "))
    runs           = int(input("How many runs: "))

    #scalarization
    def scalar_fitness(params):
        obj = robustness_under_stress(params)

        oxygen_deficit = obj[3]

        mean_ref = np.array([0.02, 0.02, 0.02, 0.05, 0.05])
        std_ref  = np.array([0.01, 0.01, 0.01, 0.02, 0.02])

        normalized = np.abs((obj - mean_ref) / std_ref)

        weights = np.array([1.0, 1.0, 1.0, 0.5, 1.0])

        base_fitness = np.sum(weights * normalized)

        # smooth constraint penalty
        oxygen_penalty = 1000 * max(0.0, oxygen_deficit - 0.08)

        return base_fitness + oxygen_penalty




    def run_experiment():

        results   = []
        histories = []

        for seed in range(runs):

            print(f"Running trial {seed+1}/{runs}")

            best_pos, best_fit, history = adaptive_bee_optimization_live(
                objective_functions=[scalar_fitness],
                min_bounds=min_bounds,
                max_bounds=max_bounds,
                num_bees=num_bees,
                max_iterations=max_iterations,
                archive_size=10,
                seed=seed
            )

            results.append(best_fit)
            histories.append(history)

            update_archive(best_pos, robustness_under_stress(best_pos))

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
        mean   = np.mean(sample)
        std    = np.std(sample)
        margin = 1.96 * std / np.sqrt(len(sample))
        ci_upper.append(mean + margin)
        ci_lower.append(mean - margin)

    plt.figure(figsize=(7, 4))
    plt.plot(avg_curve, label="BOA Multi-Organ Model")
    plt.fill_between(range(len(avg_curve)), ci_lower, ci_upper, alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("Convergence Performance")
    plt.legend()
    plt.show()

    save_metadata()
    print("Results saved to /results folder")

    generate_research_plots(histories)
    interpret_experiment_results("results/summary.csv")