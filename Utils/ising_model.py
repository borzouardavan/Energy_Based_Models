import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def create_ising_animation():
    nrows, ncols = 20, 20
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    ax.set_title("Ising Model (Spin Flips)")
    
    # Initialize random spins (-1 or 1)
    grid = np.random.choice([-1, 1], size=(nrows, ncols))
    
    # Create the initial image using a colormap (e.g. black and white or coolwarm)
    img = ax.imshow(grid, cmap='gray', interpolation='nearest', vmin=-1, vmax=1)

    def update(frame):
        nonlocal grid
        # Randomly select a fraction of spins to flip
        flip_mask = np.random.rand(nrows, ncols) < 0.1
        grid[flip_mask] *= -1
            
        img.set_data(grid)
        return img,

    ani = animation.FuncAnimation(fig, update, frames=50, interval=150, blit=True)

    # Save to the current directory (Utils folder)
    save_path = os.path.join(os.path.dirname(__file__), 'Ising_model.gif')
    ani.save(save_path, writer='pillow', fps=10)
    print(f"Animation successfully saved to {save_path}")

if __name__ == "__main__":
    create_ising_animation()