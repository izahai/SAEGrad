import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider

# Define the Sigmoid / Logistic Schedule formula
def sigmoid_schedule(t, t0, k):
    return 1 / (1 + np.exp(-k * (t - t0)))

# Create an interactive function for the sliders to control
def update_plot(t0, k):
    # Fixed timestep range for visualization consistency
    t = np.linspace(0, 1000, 1000)
    alpha_t = sigmoid_schedule(t, t0, k)
    
    # Clear the previous figure to avoid overlapping plots
    plt.figure(figsize=(10, 6))
    
    # Plot the dynamic curve
    plt.plot(t, alpha_t, label=f'k = {k}, t0 = {t0}', color='royalblue', linewidth=2)
    
    # Visual cues for the midpoint
    plt.axvline(x=t0, color='gray', linestyle=':', alpha=0.7, label=f'Midpoint (t0={t0})')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    
    # Configure plot labels and styling
    plt.title('Interactive Sigmoid / Logistic Schedule $\\alpha(t)$', fontsize=14)
    plt.xlabel('Timestep ($t$)', fontsize=12)
    plt.ylabel('$\\alpha(t)$', fontsize=12)
    plt.xlim(0, 1000)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=11)
    plt.show()

# Set up the interactive sliders
interact(
    update_plot,
    t0=IntSlider(min=100, max=900, step=10, value=500, description='Midpoint ($t_0$):'),
    k=FloatSlider(min=0.001, max=0.05, step=0.001, value=0.01, description='Sharpness ($k$):', readout_format='.3f')
);