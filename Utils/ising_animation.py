import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def create_ising_animation():
    nrows, ncols = 10, 10
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    
    # Initialize random spins (-1 or 1)
    grid = np.random.choice([-1, 1], size=(nrows, ncols))
    
    # Create the initial image
    img = ax.imshow(grid, cmap='coolwarm', interpolation='nearest')

    def update(frame):
        nonlocal grid
        # Randomly flip a few spins
        num_flips = 5
        for _ in range(num_flips):
            i, j = np.random.randint(0, nrows), np.random.randint(0, ncols)
            grid[i, j] *= -1
            
        img.set_data(grid)
        return img,

    ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)

    # Save to the current directory (Utils folder)
    save_path = os.path.join(os.path.dirname(__file__), 'Ising_model.gif')
    ani.save(save_path, writer='pillow', fps=5)
    print(f"Animation successfully saved to {save_path}")

if __name__ == "__main__":
    create_ising_animation()