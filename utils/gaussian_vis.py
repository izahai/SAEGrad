import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Setup total diffusion timesteps
total_timesteps = 50
t = np.arange(0, total_timesteps)

# --- FIXED: Adjusted initial values to fit within 50 steps ---
init_center = 25.0  # Placed right in the middle (50 / 2)
init_sigma = 10.0   # Scaled down so the curve fits nicely on screen

def compute_bell_weight(t, mu, sigma):
    """Mathematical formulation: exp(- (t - mu)^2 / (2 * sigma^2))"""
    return np.exp(-((t - mu) ** 2) / (2 * (sigma ** 2)))

# 2. Initialize the plot layout
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)  

# 3. Plot the initial state
initial_weight = compute_bell_weight(t, init_center, init_sigma)
line, = ax.plot(t, initial_weight, linewidth=2, color='royalblue', label="Dynamic Gaussian")

# Styling the graph
ax.set_title("Interactive Gaussian (Bell) Smooth Function", fontsize=12, pad=15)
ax.set_xlabel("Timestep (t)", fontsize=10)
ax.set_ylabel(r"Weight ($\alpha$)", fontsize=10)
ax.set_xlim(-1, total_timesteps) # Keep x-axis locked to your step range
ax.set_ylim(-0.05, 1.05) 
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")

# 4. Define positions for the sliders
ax_center = plt.axes([0.15, 0.12, 0.65, 0.03])
ax_sigma  = plt.axes([0.15, 0.06, 0.65, 0.03])

# 5. Create the Slider widgets
# --- FIXED: Adjusted valmax and valmin ranges for a 50-step scale ---
slider_center = Slider(
    ax=ax_center,
    label='Center (t)',
    valmin=0,
    valmax=total_timesteps,
    valinit=init_center,
    valfmt='%0.0f'
)

slider_sigma = Slider(
    ax=ax_sigma,
    label='Sigma (σ)',
    valmin=1,          # Prevent sigma from hitting 0 (which causes division by zero)
    valmax=30,         # Maximum width tailored for a 50-step window
    valinit=init_sigma,
    valfmt='%0.0f'
)

# 6. Callback function triggered whenever the sliders move
def update(val):
    current_center = slider_center.val
    current_sigma = slider_sigma.val
    
    new_weight = compute_bell_weight(t, current_center, current_sigma)
    line.set_ydata(new_weight)
    fig.canvas.draw_idle()

# Register the update function
slider_center.on_changed(update)
slider_sigma.on_changed(update)

plt.show()