import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Setup total diffusion timesteps (0 to 49)
total_timesteps = 50
t = np.arange(0, total_timesteps)

# Initial configurations matching your PyTorch defaults
init_mid = 24.5
init_k = 0.15

def compute_sigmoid_weights(t, t_mid, k):
    """
    Mathematical formulation:
    alpha = 1 / (1 + exp(-k * (t - t_mid)))
    beta = 1 - alpha
    """
    alpha = 1.0 / (1.0 + np.exp(-k * (t - t_mid)))
    beta = 1.0 - alpha
    return alpha, beta

# 2. Initialize the plot layout
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(bottom=0.25)  # Leave room for sliders at the bottom

# 3. Plot the initial state (both Alpha and Beta)
initial_alpha, initial_beta = compute_sigmoid_weights(t, init_mid, init_k)
line_alpha, = ax.plot(t, initial_alpha, linewidth=2.5, color='royalblue', label=r"Alpha ($\alpha$) - Large $t$ loss weight")
line_beta,  = ax.plot(t, initial_beta, linewidth=2.5, color='crimson', linestyle='--', label=r"Beta ($\beta$) - Small $t$ loss weight")

# Styling the graph
ax.set_title("Interactive Sigmoid Loss Weight Function (Steps 0-49)", fontsize=12, pad=15)
ax.set_xlabel("Timestep Index (t)", fontsize=10)
ax.set_ylabel("Weight Value", fontsize=10)
ax.set_xlim(-1, total_timesteps) 
ax.set_ylim(-0.05, 1.05) 
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

# 4. Define positions for the sliders [left, bottom, width, height]
ax_mid = plt.axes([0.15, 0.12, 0.65, 0.03])
ax_k   = plt.axes([0.15, 0.06, 0.65, 0.03])

# 5. Create the Slider widgets
slider_mid = Slider(
    ax=ax_mid,
    label='Center (t_mid)',
    valmin=0,
    valmax=total_timesteps - 1,
    valinit=init_mid,
    valfmt='%0.1f'
)

slider_k = Slider(
    ax=ax_k,
    label='Steepness (k)',
    valmin=0.01,   # Keep it positive to avoid flipping or flattening completely
    valmax=1.0,    # 1.0 makes it look almost like a step function
    valinit=init_k,
    valfmt='%0.2f'
)

# 6. Callback function triggered whenever the sliders move
def update(val):
    current_mid = slider_mid.val
    current_k = slider_k.val
    
    # Recalculate and update both curves
    new_alpha, new_beta = compute_sigmoid_weights(t, current_mid, current_k)
    line_alpha.set_ydata(new_alpha)
    line_beta.set_ydata(new_beta)
    
    fig.canvas.draw_idle()

# Register the update function to both sliders
slider_mid.on_changed(update)
slider_k.on_changed(update)

plt.show()