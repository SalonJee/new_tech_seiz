import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================================================
# Model Constants
# =========================================================
tau = 0.01  # Time constant (seconds)


def sigmoid(V, gain=2.5):
    # Firing rate function. High gain represents the 'excitable' brain state
    return 1 / (1 + np.exp(-gain * (V - 2.0)))


# Weights representing the Thalamocortical Loop connectivity
W_loop = np.array([
    [0,  -25,  35,   0],  # PY
    [25,   0,   0,   0],  # IN
    [25,   0,   0,  -45], # TC
    [25,   0,  25,   0]   # RE
])


# =========================================================
# 1. Suffczynski Model (Random Noise)
# =========================================================
def suffczynski_ode(t, y):
    u_noise = 2.0 + 1.5 * np.random.normal()  # Random input
    F = sigmoid(y, gain=1.0)  # Normal lower gain
    dVdt = (-y + np.dot(W_loop, F)) / tau
    dVdt[2] += u_noise
    return dVdt


# =========================================================
# 2. Haghighi Model (Deterministic Sine Wave)
# =========================================================
def haghighi_ode(t, y, freq):
    u_sine = 2.0 + 12.0 * np.cos(2 * np.pi * freq * t)
    F = sigmoid(y, gain=2.5)  # High gain state
    dVdt = (-y + np.dot(W_loop, F)) / tau
    dVdt[2] += u_sine
    return dVdt


# =========================================================
# Run Simulations (60-second interval)
# =========================================================
interval_seconds = 60.0
steps = 60000  # 1 ms resolution over 60 seconds

t_eval = np.linspace(0, interval_seconds, steps)
y0 = np.zeros(4)

# A. Suffczynski: Background Noise
sol_suff = solve_ivp(suffczynski_ode, [0, interval_seconds], y0, t_eval=t_eval)

# B. Haghighi: Quiet State (Non-Resonant Frequency 2Hz)
sol_hagh_quiet = solve_ivp(lambda t, y: haghighi_ode(t, y, freq=2.0), [0, interval_seconds], y0, t_eval=t_eval)

# C. Haghighi: Seizure State (Resonant Frequency 10Hz)
sol_hagh_seiz = solve_ivp(lambda t, y: haghighi_ode(t, y, freq=10.0), [0, interval_seconds], y0, t_eval=t_eval)


# =========================================================
# Visualization
# =========================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

ax1.plot(sol_suff.t, sol_suff.y[0], color='blue')
ax1.set_title("1. Suffczynski Model: Stochastic Background Noise (Normal Activity)")
ax1.set_ylabel("PY Potential")
ax1.grid(True)

ax2.plot(sol_hagh_quiet.t, sol_hagh_quiet.y[0], color='green')
ax2.set_title("2. Haghighi Model: Deterministic Input at 2Hz (Quiet / Interictal)")
ax2.set_ylabel("PY Potential")
ax2.grid(True)

ax3.plot(sol_hagh_seiz.t, sol_hagh_seiz.y[0], color='red')
ax3.set_title("3. Haghighi Model: Deterministic Input at 10Hz (Ictal / Seizure 'Jump')")
ax3.set_xlabel("Time (seconds)")
ax3.set_ylabel("PY Potential")
ax3.grid(True)

plt.tight_layout()
os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/scuff_hagigi_60.png", dpi=150)
plt.show()
