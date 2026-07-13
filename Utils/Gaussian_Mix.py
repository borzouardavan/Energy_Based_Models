import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## set the color themes of the figures
import requests
style = open("plot_style.py").read()
exec(style)

# Parameters for the Gaussian mixture as given in Example 2.1
p = 0.7
mu1 = 0
mu2 = 5
sigma1 = 1
sigma2 = 0.5

# Generate x values for plotting
x = np.linspace(-3, 7, 500)

# Calculate the exact PDF of the Gaussian Mixture
pdf = p * norm.pdf(x, mu1, sigma1) + (1 - p) * norm.pdf(x, mu2, sigma2)

# Calculate the associated Energy (Negative Log-Likelihood)
# Note: EBM represents PDF as proportional to exp(-U). Thus U = -log(PDF).
U = -np.log(pdf)

# Generate synthetic samples to represent the "Data" histogram
np.random.seed(42)
n_samples = 5000
component_choices = np.random.rand(n_samples) < p
samples = np.where(component_choices, 
                   np.random.normal(mu1, sigma1, n_samples), 
                   np.random.normal(mu2, sigma2, n_samples))

import os

# Probability Plot
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.hist(samples, bins=50, density=True, color='gray', alpha=0.7, label='Data')
ax1.plot(x, pdf, 'r-', lw=2, label=r'$p$')
ax1.set_title('Gaussian Mixture and Sampled Data')
ax1.legend()
fig1.tight_layout()
prob_path = os.path.join(os.path.dirname(__file__), 'gaussian_prob.png')
fig1.savefig(prob_path, dpi=300)

# Energy Plot
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(x, U, 'r-', lw=2, label=r'$U_\theta$')
ax2.set_xlabel('x')
ax2.set_ylabel('U')
ax2.set_title('Energy')
ax2.legend()
fig2.tight_layout()
energy_path = os.path.join(os.path.dirname(__file__), 'gaussian_energy.png')
fig2.savefig(energy_path, dpi=300)

print(f"Saved {prob_path} and {energy_path}")