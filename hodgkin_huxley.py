import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================================================
# Hodgkin-Huxley Model Parameters
# =========================================================

Cm = 1.0      # membrane capacitance (uF/cm^2)

gNa = 120.0   # maximum sodium conductance
gK  = 36.0    # maximum potassium conductance
gL  = 0.3     # leak conductance

ENa = 50.0    # sodium reversal potential (mV)
EK  = -77.0   # potassium reversal potential (mV)
EL  = -54.387 # leak reversal potential (mV)

# =========================================================
# External Current
# =========================================================

def external_current(t):

    # Apply current only between 10 ms and 40 ms
    if 10 <= t <= 40:
        return 10.0

    return 0.0

# =========================================================
# Gating Variable Rate Functions
# =========================================================

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    return 4 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))

# =========================================================
# Hodgkin-Huxley Differential Equations
# =========================================================

def hodgkin_huxley(t, y):

    V, m, h, n = y

    # External stimulus current
    I_ext = external_current(t)

    # Ionic currents
    INa = gNa * (m**3) * h * (V - ENa)
    IK  = gK  * (n**4) * (V - EK)
    IL  = gL * (V - EL)

    # Membrane voltage equation
    dVdt = (I_ext - INa - IK - IL) / Cm

    # Gating variable equations
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dVdt, dmdt, dhdt, dndt]

# =========================================================
# Initial Conditions
# =========================================================

V0 = -65.0

m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

y0 = [V0, m0, h0, n0]

# =========================================================
# Time Setup
# =========================================================

t_start = 0
t_end = 50

t_points = np.linspace(t_start, t_end, 5000)

# =========================================================
# Solve ODE System
# =========================================================

solution = solve_ivp(
    hodgkin_huxley,
    [t_start, t_end],
    y0,
    t_eval=t_points,
    method='RK45'
)

# =========================================================
# Extract Results
# =========================================================

t = solution.t

V = solution.y[0]
m = solution.y[1]
h = solution.y[2]
n = solution.y[3]

# =========================================================
# Plot Membrane Potential
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(t, V)

plt.title("Hodgkin-Huxley Neuron Simulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")

plt.grid(True)

plt.savefig("graphs/PV vs Time general (HH).jpeg")
plt.show()


# =========================================================
# Plot Gating Variables
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(t, m, label='m')
plt.plot(t, h, label='h')
plt.plot(t, n, label='n')

plt.title("Gating Variables")
plt.xlabel("Time (ms)")
plt.ylabel("Probability")

plt.legend()
plt.grid(True)

plt.savefig("graphs/Time vs Probability (HH).jpeg")
plt.show()
