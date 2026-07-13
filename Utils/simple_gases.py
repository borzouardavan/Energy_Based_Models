import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

## set the color themes of the figures
import requests
style = open("plot_style.py").read()
exec(style)

def create_gas_animation():
    n_particles = 50
    # Random initial positions and velocities
    pos = np.random.rand(n_particles, 2)
    vel = (np.random.rand(n_particles, 2) - 0.5) * 0.1

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Gas Molecules")

    scatter = ax.scatter(pos[:, 0], pos[:, 1], s=40, c='red', alpha=0.6, edgecolors='none')

    def update(frame):
        nonlocal pos, vel
        pos += vel
        
        # Bounce off walls
        for i in range(2):
            mask_min = pos[:, i] < 0
            mask_max = pos[:, i] > 1
            pos[mask_min, i] = -pos[mask_min, i]
            vel[mask_min, i] *= -1
            pos[mask_max, i] = 2 - pos[mask_max, i]
            vel[mask_max, i] *= -1
            
        scatter.set_offsets(pos)
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

    # Save to the current directory (Utils folder)
    save_path = os.path.join(os.path.dirname(__file__), 'gases.gif')
    ani.save(save_path, writer='pillow', fps=20)
    print(f"Animation successfully saved to {save_path}")

if __name__ == "__main__":
    create_gas_animation()
