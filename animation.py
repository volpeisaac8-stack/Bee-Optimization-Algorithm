import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def demo_animation():
    # Synthetic 3D Landscape
  
    def landscape(x, y):
        return (
            np.sin(2*x) * np.cos(2*y)
            + 0.5 * np.exp(-(x**2 + y**2))
            + 0.2 * np.sin(3*x)
        )

    # Create surface grid
    grid_size = 80
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = landscape(X, Y)

# Synthetic 3D Landscape
def landscape(x, y):
    return (
        np.sin(2*x) * np.cos(2*y)
        + 0.5 * np.exp(-(x**2 + y**2))
        + 0.2 * np.sin(3*x)
    )

# Create surface grid
grid_size = 80
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)
Z = landscape(X, Y)


# Initialize Bee Swarm
num_bees = 50
positions = np.random.uniform(-3, 3, (num_bees, 2))


def swarm_step(pos):
    new_pos = pos.copy()

    for i in range(len(pos)):
        # Attraction toward center basin
        attraction = -0.05 * pos[i]

        # Random exploration
        noise = np.random.normal(0, 0.1, 2)

        new_pos[i] += attraction + noise

    return np.clip(new_pos, -3, 3)


fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.35)

scatter = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    landscape(positions[:, 0], positions[:, 1]),
    s=50
)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(np.min(Z)-1, np.max(Z)+1)

ax.set_title("3D Bee Swarm Search Demonstration")
ax.set_xlabel("Parameter X")
ax.set_ylabel("Parameter Y")
ax.set_zlabel("Fitness")


def update(frame):
    global positions

    positions[:] = swarm_step(positions)

    z_vals = landscape(positions[:, 0], positions[:, 1])

    scatter._offsets3d = (
        positions[:, 0],
        positions[:, 1],
        z_vals
    )

    return scatter,

ani = FuncAnimation(
    fig,
    update,
    frames=150,
    interval=80,
    blit=False
)

plt.show()

demo_animation()