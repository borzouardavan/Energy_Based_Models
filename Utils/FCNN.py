import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_s_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
## set the color themes of the figures
import requests
style = requests.get("https://compu-flair.com/notes/jupyter-color-style/raw").text
exec(style)

def draw_fully_connected_1_50_1() -> None:
    """Plot a fully connected neural network with architecture 1 -> 50 -> 1."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # X positions for layers
    x_input, x_hidden, x_output = 0.0, 1.0, 2.0

    # Y positions for nodes
    y_input = [0.5]  # x
    y_hidden = [i / 49 for i in range(50)]  # 50 hidden neurons
    y_output = [0.5]  # y

    # Draw all fully connected edges (input -> hidden)
    for yi in y_input:
        for yh in y_hidden:
            ax.plot([x_input, x_hidden], [yi, yh], color="#00FFFF", alpha=0.3, linewidth=0.8)

    # Draw all fully connected edges (hidden -> output)
    for yh in y_hidden:
        for yo in y_output:
            ax.plot([x_hidden, x_output], [yh, yo], color="#FFD700", alpha=0.3, linewidth=0.8)

    # Draw nodes
    node_style = dict(s=120, edgecolors="white", linewidths=0.8, zorder=3)
    ax.scatter([x_input] * len(y_input), y_input, color="#00FF00", **node_style)
    ax.scatter([x_hidden] * len(y_hidden), y_hidden, color="#FF00FF", s=55, edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter([x_output] * len(y_output), y_output, color="#FF3333", **node_style)

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.12, 1.1)
    ax.axis("off")
    plt.tight_layout()
    import os
    plt.savefig(os.path.join(os.path.dirname(__file__), "FCNN.png"), bbox_inches="tight", dpi=150, transparent=True)
    plt.show()


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def make_energy_function():
    """Return E(f_theta(x), y) = 1/2 * (f_theta(x) - y)^2."""
    def energy(f_theta_x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * (f_theta_x - y) ** 2

    return energy


def plot_energy_surface(
    model: nn.Module,
    x_mean: float,
    x_std: float,
    y_mean: float,
    y_std: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Plot 3D E(x, y) by evaluating f_theta(x) on a grid of all x-y combinations."""
    model.eval()

    x_vals = np.linspace(x_min, x_max, 180, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, 180, dtype=np.float32)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluate network only on x, then broadcast across all y values.
    x_vals_norm = ((x_vals - x_mean) / x_std).reshape(-1, 1)
    with torch.no_grad():
        x_tensor = torch.tensor(x_vals_norm, dtype=torch.float32)
        f_norm = model(x_tensor).cpu().numpy().squeeze()
    f_x = f_norm * y_std + y_mean

    F = np.tile(f_x, (len(y_vals), 1))
    energy = make_energy_function()
    E = energy(F, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, E, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)

    ax.set_title("Energy Landscape: E(x, y) = 1/2 * (f_theta(x) - y)^2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("E")
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=16, pad=0.1, label="Energy")
    plt.tight_layout()
    import os
    plt.savefig(os.path.join(os.path.dirname(__file__), "FCNN_energy_3d.png"), bbox_inches="tight", dpi=180, transparent=True)
    plt.show()

def train_fcnn():
    # 1. Generate a 2-column regression dataset: [x, y]
    # make_s_curve returns a continuous target t; we use one coordinate as x and t as y.
    X_s, t = make_s_curve(n_samples=1000, noise=0.05, random_state=42)
    X = X_s[:, [0]].astype(np.float32)
    y = t.astype(np.float32)
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(y.min()), float(y.max())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize using training statistics only
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # 2. Initialize the model, loss function, and optimizer
    model = FCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred = test_outputs.cpu().numpy().squeeze()
        y_true = y_test_tensor.cpu().numpy().squeeze()
        rmse = float(np.sqrt(mean_squared_error(y_true, test_pred)))
        r2 = float(r2_score(y_true, test_pred))
        print(f"Test RMSE (normalized y): {rmse:.4f}")
        print(f"Test R^2 (normalized y): {r2:.4f}")

    return (
        model,
        float(x_mean.squeeze()),
        float(x_std.squeeze()),
        float(y_mean),
        float(y_std),
        x_min,
        x_max,
        y_min,
        y_max,
    )




if __name__ == "__main__":
    draw_fully_connected_1_50_1()
    trained_model, x_mean, x_std, y_mean, y_std, x_min, x_max, y_min, y_max = train_fcnn()
    plot_energy_surface(trained_model, x_mean, x_std, y_mean, y_std, x_min, x_max, y_min, y_max)

