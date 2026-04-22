import numpy as np
import matplotlib.pyplot as plt


# Helper Functions

def reflect_bounds(x, lb, ub):
    x = np.where(x < lb, lb + (lb - x), x)
    x = np.where(x > ub, ub - (x - ub), x)
    return np.clip(x, lb, ub)


def levy_flight(dim, beta=1.5):

    from math import gamma, pi, sin

    sigma_u = (gamma(1+beta) * sin(pi*beta/2) /
               (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)

    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)

    step = u / (np.abs(v) ** (1.0 / beta))
    return step


def mirror_population(positions, lb, ub):
    mirrored = lb + ub - positions
    return np.clip(mirrored, lb, ub)


# Batched ODE Evaluator — evaluates entire swarm in one vectorized pass

def robustness_batch(population, shock_magnitude=0.3, K=3):
    """
    Evaluate entire swarm at once using vectorized numpy.
    population shape: (N_bees, 4)
    returns fitness shape: (N_bees,)
    """
    N_bees = population.shape[0]
    T = 200
    dt = 0.02
    shock_t = T // 2

    H = np.ones(N_bees)
    R = np.ones(N_bees)
    N = np.ones(N_bees)
    M = np.ones(N_bees)

    k_hr        = population[:, 0]
    k_oxygen    = population[:, 1]
    k_neural    = population[:, 2]
    k_metabolic = population[:, 3]

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

    recovery_penalty   = np.mean(np.abs(heart_trace[-50:] - 1), axis=0)
    shock_response_var = np.var(heart_trace[shock_t:], axis=0)
    oxygen_violation   = np.mean(np.maximum(0, 1.05 - resp_trace), axis=0)

    fitness = (
        np.var(heart_trace, axis=0)
        + np.var(neural_trace, axis=0)
        + np.var(metabolic_trace, axis=0)
        + oxygen_violation
        + recovery_penalty + shock_response_var
    )

    return fitness  # shape: (N_bees,)


# Adaptive Bee Optimization Live Engine

def adaptive_bee_optimization_live(objective_functions,
                                   min_bounds,
                                   max_bounds,
                                   num_bees=300,
                                   max_iterations=500,
                                   switch_iterations=20,
                                   patience=20,
                                   initial_radius=0.3,
                                   final_radius=0.01,
                                   archive_size=10,
                                   seed=None):

    if seed is not None:
        np.random.seed(seed)

    dims = len(min_bounds)
    lb = np.array(min_bounds)
    ub = np.array(max_bounds)

    positions = np.random.uniform(lb, ub, (num_bees, dims))

    # Initial swarm fitness — one batched call
    fitness = robustness_batch(positions)

    diversity = np.mean(np.std(positions, axis=0))

    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]

    # ================= MEMORY ARCHIVE =================

    archive_positions = []
    archive_fitness = []

    if archive_size > 0:
        archive_positions.append(best_pos.copy())
        archive_fitness.append(best_fit)

    fitness_history = []

    no_improve_counter = 0
    switch_counter = 0

    elite_count = max(1, int(0.05 * num_bees))
    scout_ratio = 0.25

    # Convergence plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [])

    ax.set_xlim(0, max_iterations)
    ax.set_yscale("log")
    ax.set_title("Adaptive Bee Optimization - Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")

    # Main Optimization Loop

    for it in range(max_iterations):

        t = it / max(1, max_iterations - 1)

        radius = (
            final_radius
            + (initial_radius - final_radius) * (1 - t)
            + 0.05 * np.sin(10 * np.pi * t)
        )

        levy_prob = 0.1 + 0.8 * (1 - np.exp(-no_improve_counter / 5))
        levy_prob = min(0.8, levy_prob)

        if fitness[np.argmin(fitness)] < best_fit:
            current_best_idx = np.argmin(fitness)
            current_best_fit = fitness[current_best_idx]

            if current_best_fit < best_fit:
                best_fit = current_best_fit
                best_pos = positions[current_best_idx].copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            if archive_size > 0:
                archive_positions.append(best_pos.copy())
                archive_fitness.append(best_fit)

                if len(archive_positions) > archive_size:
                    combined = sorted(zip(archive_positions, archive_fitness), key=lambda x: x[1])
                    combined = combined[:archive_size]
                    if combined:
                        archive_positions, archive_fitness = zip(*combined)
                        archive_positions = list(archive_positions)
                        archive_fitness = list(archive_fitness)
        else:
            no_improve_counter += 1

        # Anti-stagnation

        diversity = np.mean(np.std(positions, axis=0))
        scale = np.mean(ub - lb) + 1e-12
        div_norm = diversity / scale
        if no_improve_counter > patience:
            print(f"Stagnation detected at iter {it}, injecting diversity")
            worst_idx = np.argsort(fitness)[int(0.7 * num_bees):]
            positions[worst_idx] = best_pos + np.random.normal(
                0, 0.5 * (ub - lb), size=(len(worst_idx), dims)
            )
            levy_prob = min(0.8, levy_prob + 0.2)
            no_improve_counter = 0

        fitness_history.append(best_fit)

        # Probability normalization
        inv = 1.0 / (1.0 + fitness - np.min(fitness))
        denom = np.sum(inv)

        if denom == 0 or np.isnan(denom):
            probs = np.ones_like(inv) / len(inv)
        else:
            probs = inv / denom

        order = np.argsort(fitness)
        elites_idx = order[:elite_count]

        # Diversity measurement
        diversity = np.mean(np.std(positions, axis=0))
        div_norm  = diversity / np.mean(ub - lb)

        if div_norm < 0.12:
            worst_idx = np.argsort(fitness)[int(0.5 * num_bees):]
            positions[worst_idx] = np.random.uniform(lb, ub, (len(worst_idx), dims))
            positions = reflect_bounds(positions, lb, ub)

        top_k   = min(5, num_bees)
        leaders = positions[np.argsort(fitness)[:top_k]]

        # ── Build all trials first, then evaluate in ONE batch ──────────────

        trials = positions.copy()

        # Exploration Phase
        for i in range(num_bees):

            if i in elites_idx:
                continue

            if np.random.rand() < levy_prob:
                step  = levy_flight(dims) * (ub - lb) * (0.03 + 0.07 * (1 - t))
                trial = positions[i] + step

            else:
                j        = np.random.choice(num_bees, p=probs)
                neighbor = positions[j]
                leader   = leaders[np.random.randint(top_k)]
                best_pull = 0.03 * (leader - positions[i])

                phi          = np.random.normal(0, 1, size=dims) * radius
                differential = phi * (neighbor - positions[i])

                if archive_size > 0 and len(archive_positions) > 1:
                    archive_choice = archive_positions[np.random.randint(len(archive_positions))]
                    archive_pull = 0.05 * np.random.rand(dims) * (archive_choice - positions[i])
                    archive_pull = 0         
                else:
                    archive_pull = 0

                noise = np.random.normal(0, 0.005 * (1 - t) * (ub - lb), size=dims)

                trial = (
                    positions[i]
                    + differential
                    + 0.4 * best_pull
                    + archive_pull
                    + noise
                )

            trials[i] = reflect_bounds(trial, lb, ub)

        # ONE batch eval for exploration trials
        trial_fitness = robustness_batch(trials)
        improved      = trial_fitness < fitness
        positions[improved] = trials[improved]
        fitness[improved]   = trial_fitness[improved]

        # Crossover — build trials then batch eval
        cross_trials = positions.copy()

        for i in range(num_bees):
            j          = np.random.randint(num_bees)
            cross_mask = np.random.rand(dims) < 0.5
            alpha = np.random.rand(dims)
            cross_trials[i] = reflect_bounds(
                alpha * positions[i] + (1 - alpha) * positions[j],
                lb, ub
            )

        cross_fitness = robustness_batch(cross_trials)
        improved      = cross_fitness < fitness
        positions[improved] = cross_trials[improved]
        fitness[improved]   = cross_fitness[improved]

        # Onlooker Phase — build trials then batch eval
        num_onlookers   = int((1 - scout_ratio) * num_bees)
        onlooker_trials = positions.copy()
        onlooker_mask   = np.zeros(num_bees, dtype=bool)

        for _ in range(num_onlookers):
            k = np.random.choice(num_bees, p=probs)
            if k in elites_idx:
                continue
            phi   = np.random.normal(0, 1, size=dims) * radius
            trial = positions[k] + phi * (best_pos - positions[k])
            onlooker_trials[k] = reflect_bounds(trial, lb, ub)
            onlooker_mask[k]   = True

        if onlooker_mask.any():
            onlooker_fitness = robustness_batch(onlooker_trials)
            improved = onlooker_mask & (onlooker_fitness < fitness)
            positions[improved] = onlooker_trials[improved]
            fitness[improved]   = onlooker_fitness[improved]

        # Mirror Learning — batch eval
        mirrored       = mirror_population(positions, lb, ub)
        mirror_fitness = robustness_batch(mirrored)
        improved       = mirror_fitness < fitness
        positions[improved] = mirrored[improved]
        fitness[improved]   = mirror_fitness[improved]

        # Elite Reinforcement (no eval needed, positions perturbed in place)
        for ei in elites_idx:
            if np.random.rand() < 0.7:
                if np.random.rand() < 0.3:
                    positions[ei] += np.random.normal(0, 0.01 * (ub - lb), size=dims)

        if np.random.rand() < 0.3:
            positions[0] = best_pos.copy()

        switch_counter += 1

        # Plot convergence
        if it % 1 == 0:
            line.set_data(range(len(fitness_history)), fitness_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"Iter {it:04d}: "
                f"Best fitness = {best_fit:.5e} "
                f"| Diversity = {diversity:.4f} "
                f"| Archive size = {len(archive_positions)}"
            )

    plt.ioff()
    plt.show(block=False)
    plt.close(fig)

    return best_pos, best_fit, fitness_history