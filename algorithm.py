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



# Scalar Fitness Projection


def scalar_score(v):
    v = np.array(v)
    w = np.ones(len(v)) / len(v)   # equal weighting automatically
    return np.sum(w * v)


# Adaptive Bee Optimization Live Engine

def adaptive_bee_optimization_live(objective_functions,
                                   min_bounds,
                                   max_bounds,
                                   num_bees=300,
                                   max_iterations=500,
                                   switch_iterations=20,
                                   patience=20, #can be changed up to 40
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

    current_func = objective_functions[0]

    # Initial swarm fitness
    fitness = np.array([
        scalar_score(current_func(p))
        for p in positions
    ])

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

    levy_prob = 0.35 #can be up to 0.3 or 0.4
    scout_ratio = 0.25 #can be up to 0.3

    # Convergence plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))
    line, = ax.plot([], [])

    ax.set_xlim(0, max_iterations)
    ax.set_yscale("log")
    ax.set_title("Adaptive Bee Optimization - Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")

    cardiac_trace = []
    resp_trace = []
    neural_trace = []
    metabolic_trace = []

    # Second plot window must be created externally
    fig2, ax2 = None, None
    lines = None

    # Main Optimization Loop

    for it in range(max_iterations):

        t = it / max(1, max_iterations-1)
        radius = initial_radius*(1-t) + final_radius*t

        # Swarm evaluation
        fitness = np.array([
            scalar_score(current_func(p))
            for p in positions
        ])

        idx = np.argmin(fitness)

        # Global best update
        if fitness[idx] < best_fit:

            best_fit = fitness[idx]
            best_pos = positions[idx].copy()

            no_improve_counter = 0

            if archive_size > 0:
                archive_positions.append(best_pos.copy())
                archive_fitness.append(best_fit)

                if len(archive_positions) > archive_size:

                    combined = list(zip(
                        archive_positions,
                        archive_fitness
                    ))

                    combined.sort(key=lambda x: x[1])
                    combined = combined[:archive_size]

                    archive_positions = [c[0] for c in combined]
                    archive_fitness = [c[1] for c in combined]

        else:
            no_improve_counter += 1

        fitness_history.append(best_fit)

        # Probability normalization safety block
        inv = 1.0 / (1.0 + fitness - np.min(fitness))
        denom = np.sum(inv)

        if denom == 0 or np.isnan(denom):
            probs = np.ones_like(inv) / len(inv)
        else:
            probs = inv / denom

        order = np.argsort(fitness)
        elites_idx = order[:elite_count]

        # Exploration Phase
        for i in range(num_bees):

            if i in elites_idx:
                continue

            if np.random.rand() < levy_prob:

                step = levy_flight(dims)*(ub-lb)*0.05
                trial = positions[i] + step

            else:

                j = np.random.choice(num_bees, p=probs)
                neighbor = positions[j]

                phi = np.random.normal(0,1,size=dims)*radius
                trial = positions[i] + phi*(positions[i]-neighbor)

                trial += np.random.normal(
                    0,0.01*(ub-lb), size=dims)

            trial = reflect_bounds(trial, lb, ub)

            f_trial = scalar_score(current_func(trial))

            if f_trial < fitness[i]:
                positions[i] = trial
                fitness[i] = f_trial

        # Onlooker Phase
        num_onlookers = int((1-scout_ratio)*num_bees)

        for _ in range(num_onlookers):

            k = np.random.choice(num_bees, p=probs)

            if k in elites_idx:
                continue

            phi = np.random.normal(0,1,size=dims)*radius
            trial = positions[k] + phi*(best_pos-positions[k])

            trial = reflect_bounds(trial, lb, ub)

            f_trial = scalar_score(current_func(trial))

            if f_trial < fitness[k]:
                positions[k] = trial
                fitness[k] = f_trial

        # Mirror Learning
        mirrored = mirror_population(positions, lb, ub)

        for i in range(num_bees):

            f_mirror = scalar_score(current_func(mirrored[i]))

            if f_mirror < fitness[i]:
                positions[i] = mirrored[i]
                fitness[i] = f_mirror

        # Elite Reinforcement
        for rank_idx, ei in enumerate(elites_idx):
            if np.random.rand() < 0.7:
                positions[rank_idx] = positions[ei] + np.random.normal(
                    0, 0.02*(ub-lb),
                    size=dims
        )

        positions[0] = best_pos.copy()

        switch_counter += 1

        # Plot convergence
        if it % 1 == 0:

            line.set_data(
                range(len(fitness_history)),
                fitness_history
            )

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"Iter {it:04d}: "
                f"Best fitness = {best_fit:.5e} "
                f"| Archive size = {len(archive_positions)}"
            )

    plt.ioff()
    plt.show(block=False)
    plt.close(fig)

    return best_pos, best_fit, fitness_history