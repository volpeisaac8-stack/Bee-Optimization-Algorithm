import numpy as np

def particle_swarm_optimization(
    objective_function,
    min_bounds,
    max_bounds,
    num_particles=30,
    max_iterations=100,
    w=0.7,
    c1=1.5,
    c2=1.5,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dim = len(min_bounds)
    lb = np.array(min_bounds)
    ub = np.array(max_bounds)

    print("\n==============================")
    print("   PARTICLE SWARM OPTIMIZER")
    print("==============================")
    print(f"Particles      : {num_particles}")
    print(f"Dimensions     : {dim}")
    print(f"Iterations     : {max_iterations}")
    print(f"Bounds         : {lb} → {ub}")
    print("==============================\n")

    # Initialize particles
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))

    # Evaluate initial swarm
    pbest_pos = positions.copy()
    pbest_fit = np.array([objective_function(p) for p in positions])

    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    print(f"Initial best fitness: {gbest_fit:.6f}\n")

    history = []

    for t in range(max_iterations):

        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)

        # Velocity update
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        # Position update
        positions = positions + velocities
        positions = np.clip(positions, lb, ub)

        # Evaluate
        fitness = np.array([objective_function(p) for p in positions])

        # Update personal bests
        better_mask = fitness < pbest_fit
        pbest_pos[better_mask] = positions[better_mask]
        pbest_fit[better_mask] = fitness[better_mask]

        # Update global best
        current_best_idx = np.argmin(pbest_fit)
        if pbest_fit[current_best_idx] < gbest_fit:
            gbest_fit = pbest_fit[current_best_idx]
            gbest_pos = pbest_pos[current_best_idx].copy()

        history.append(gbest_fit)

        # 🔥 PRINT PROGRESS EVERY 10 ITERATIONS + FIRST
        if t % 2 == 0 or t == max_iterations - 1:
            print(f"Iteration {t+1:4d} | Best Fitness: {gbest_fit:.6f}")

    print("\n==============================")
    print("PSO COMPLETE")
    print(f"Final Best Fitness: {gbest_fit:.6f}")
    print("==============================\n")

    return gbest_pos, gbest_fit, history