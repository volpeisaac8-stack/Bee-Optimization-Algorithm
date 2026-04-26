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
        dR = k_oxygen * (H - R) - 0.3 * np.maximum(0, 1.05 - R)
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
        + recovery_penalty + shock_response_var
        + 5.0 * oxygen_violation
        + 0.2 * np.mean(np.abs(np.diff(heart_trace, axis=0)), axis=0)
    )

    return fitness, oxygen_violation  # shape: (N_bees,)


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
    fitness, violation = robustness_batch(positions)

    diversity = np.mean(np.std(positions, axis=0))

    feasible = violation < 1e-6

    if np.any(feasible):
        masked_fitness = fitness.copy()
        masked_fitness[~feasible] = 1e12
        best_idx = np.argmin(masked_fitness)
    else:
        best_idx = np.argmin(violation)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]
    best_violation = violation[best_idx]

    # Memory archive

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
        FEAS_TOL = 1e-6
# ================= EXPLOITATION PHASE =================
        if t > 0.6:   

            # soft collapse instead of hard collapse
            positions += 0.3 * (best_pos - positions)

            # Local Search
            if it % 2 == 0:
                for _ in range(10):
                    epsilon = (0.02 * (1 - t) + 0.005) * (ub - lb)

                    local_trial = best_pos + np.random.normal(0, epsilon, size=dims)
                    local_trial = reflect_bounds(local_trial, lb, ub)

                    local_fit, local_violation = robustness_batch(local_trial.reshape(1, -1))
                    local_fit = local_fit[0]
                    local_violation = local_violation[0]

                    if (local_violation < 1e-6 and best_violation >= 1e-6) or \
                    (local_violation < 1e-6 and best_violation < 1e-6 and local_fit < best_fit) or \
                    (local_violation >= 1e-6 and best_violation >= 1e-6 and local_violation < best_violation):

                        best_fit = local_fit
                        best_pos = local_trial.copy()
                        best_violation = local_violation

                fitness, violation = robustness_batch(positions)

                positions[0] = best_pos.copy()
                fitness[0] = best_fit
                violation[0] = best_violation

                continue
 

        radius = final_radius + (initial_radius - final_radius) * (0.5 + 0.5*np.cos(np.pi * t))

        levy_prob = 0.1 + 0.8 * (1 - np.exp(-no_improve_counter / 5))
        levy_prob = min(0.8, levy_prob)

        if fitness[np.argmin(fitness)] < best_fit:
            current_best_idx = np.argmin(fitness)
            current_best_fit = fitness[current_best_idx]

            if current_best_fit < best_fit:
                best_fit = current_best_fit
                best_pos = positions[current_best_idx].copy()
                no_improve_counter = 0

                for _ in range(5):
                    epsilon = 0.005 * (1 - t)**2 * (ub - lb)

                    local_trial = best_pos + np.random.normal(0, epsilon, size=dims)
                    local_trial = reflect_bounds(local_trial, lb, ub)

                    local_fit, local_violation = robustness_batch(local_trial.reshape(1, -1))
                    local_fit = local_fit[0]
                    local_violation = local_violation[0]

                    if local_fit < best_fit:
                        best_fit = local_fit
                        best_pos = local_trial.copy()

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

        if no_improve_counter > patience:
            print(f"Stagnation detected at iter {it}, injecting diversity")

            worst_idx = np.argsort(fitness)[int(0.7 * num_bees):]

            # Split worst into two groups
            split = len(worst_idx) // 2
            global_idx = worst_idx[:split]
            local_idx = worst_idx[split:]

            # GLOBAL random reset 
            positions[global_idx] = np.random.uniform(lb, ub, size=(len(global_idx), dims))

            # LOCAL noisy reset around best
            # stronger escape mechanism
            n_local = len(local_idx)
            half = n_local // 2

            # half full random restart
            positions[local_idx[:half]] = np.random.uniform(lb, ub, (half, dims))

            # half directional escape (not centered on best)
            escape = np.random.normal(0, 1, size=(n_local - half, dims))
            escape /= (np.linalg.norm(escape, axis=1, keepdims=True) + 1e-9)

            positions[local_idx[half:]] = best_pos + escape * (0.4 * (ub - lb))

            levy_prob = min(0.9, levy_prob + 0.3)
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

        #  Build all trials first then evaluate in ONE batch

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
                best_pull = (0.1 + 0.2 * (1 - t)) * (leader - positions[i])

                phi          = np.random.normal(0, 1, size=dims) * radius
                differential = phi * (neighbor - positions[i])

                if archive_size > 0 and len(archive_positions) > 1:
                    archive_choice = archive_positions[np.random.randint(len(archive_positions))]
                    archive_pull = -0.05 * (archive_choice - positions[i])                
                else:
                    archive_pull = 0

                noise = np.random.normal(0, 0.001 * (1 - t)**2 * (ub - lb), size=dims)

                trial = (
                    positions[i]
                    + differential
                    + 0.4 * best_pull
                    + archive_pull
                    + noise
                )

            trials[i] = reflect_bounds(trial, lb, ub)

        # ONE batch eval for exploration trials
        trial_fitness, trial_violation = robustness_batch(trials)
        feasible_current = violation < FEAS_TOL
        feasible_trial   = trial_violation < FEAS_TOL

        improved = (
            (feasible_trial & ~feasible_current) |
            (feasible_trial & feasible_current & (trial_fitness < fitness)) |
            (~feasible_trial & ~feasible_current & (trial_violation < violation))
        )
        positions[improved] = trials[improved]
        fitness[improved]   = trial_fitness[improved]
        violation[improved] = trial_violation[improved]

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
        if it % 2 == 0:
            cross_fitness, cross_violation = robustness_batch(cross_trials)

            feasible_current = violation < 1e-6
            feasible_trial   = cross_violation < 1e-6

            improved = (
                (feasible_trial & ~feasible_current) |
                (feasible_trial & feasible_current & (cross_fitness < fitness)) |
                (~feasible_trial & ~feasible_current & (cross_violation < violation))
            )

            positions[improved] = cross_trials[improved]
            fitness[improved]   = cross_fitness[improved]
            violation[improved] = cross_violation[improved]

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
            onlooker_fitness, onlooker_violation = robustness_batch(onlooker_trials)

            feasible_current = violation < 1e-6
            feasible_trial   = onlooker_violation < 1e-6

            improved = onlooker_mask & (
                (feasible_trial & ~feasible_current) |
                (feasible_trial & feasible_current & (onlooker_fitness < fitness)) |
                (~feasible_trial & ~feasible_current & (onlooker_violation < violation))
            )

            positions[improved] = onlooker_trials[improved]
            fitness[improved]   = onlooker_fitness[improved]
            violation[improved] = onlooker_violation[improved]

        # Mirror Learning — batch eval
        mirrored       = mirror_population(positions, lb, ub)
        mirror_fitness, mirror_violation = robustness_batch(mirrored)

        feasible_current = violation < 1e-6
        feasible_trial   = mirror_violation < FEAS_TOL

        improved = (
            (feasible_trial & ~feasible_current) |
            (feasible_trial & feasible_current & (mirror_fitness < fitness)) |
            (~feasible_trial & ~feasible_current & (mirror_violation < violation))
        )

        positions[improved] = mirrored[improved]
        fitness[improved]   = mirror_fitness[improved]
        violation[improved] = mirror_violation[improved]


        # Elite Reinforcement 
        if t > 0.8:
            positions += 0.2 * (best_pos - positions)

        for ei in elites_idx:
            if np.random.rand() < 0.7:
                if np.random.rand() < 0.3:
                    positions[ei] += np.random.normal(0, 0.01 * (ub - lb), size=dims)

        if np.random.rand() < 0.3:
            positions[0] = best_pos.copy()

        switch_counter += 1

        # Plot convergence
        if it % 5 == 0:
            line.set_data(range(len(fitness_history)), fitness_history)
            if it % 5 == 0:
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