"""
real_model.py
=============
A Python port of CxTh_Suff_Neurosci_2004.mdl (Suffczynski, Kalitzin & Lopes
da Silva, 2004), reverse-engineered directly from the Simulink model file
rather than approximated. Every constant below was read (or algebraically
solved) from the .mdl's actual Block definitions -- see the accompanying
chat message for the block-by-block trace. This REPLACES the placeholder
constants used in the earlier model_core.py.

MODEL STRUCTURE
---------------
4 membrane potentials: V_PY, V_IN, V_TC, V_RE
9 synaptic conductance filters, each a real 2nd-order transfer function
    (converted here to a 2-state ODE in controllable canonical form:
     for G(s) = b0 / (s^2 + a1*s + a0):
         x1' = x2
         x2' = -a0*x1 - a1*x2 + u(t)
         y   = b0*x1
     which is an exact, lossless conversion of the Simulink TransferFcn
     block -- not an approximation.)
2 thalamic burst-firing pathways (TC, RE), each with its own m_inf/n_inf
    activation/inactivation sigmoids and a shared delay-kernel filter,
    exactly Eqs. 6-7 of the paper.

KNOWN SIMPLIFICATION (flagged, not hidden):
The two 8 ms Transport Delay blocks (on the PY-output and TC-burst-output
feedback paths) are NOT implemented -- those feedback signals are used
instantaneously here. Implementing them properly needs a delay-
differential-equation approach (method of steps), which is a further,
separate piece of work. Everything else below is a direct, lossless port.
"""

import numpy as np
from scipy.integrate import solve_ivp

# =====================================================================
# State vector layout (26 states)
# =====================================================================
V_PY, V_IN, V_TC, V_RE = 0, 1, 2, 3
# each synapse filter occupies 2 consecutive slots (x1, x2)
S_AMPA_PY   = (4, 5)     # "2" block:      C12*u_cortical + C15*FB_TC  -> PY
S_AMPA_IN   = (6, 7)     # "AMPA1" block:  C11*F_PY + C8minus*FB_TC   -> IN
S_GABAA_PY  = (8, 9)     # "GABAA" block:  F_IN                       -> PY
S_GABAB_PY  = (10, 11)   # "GABAb" block:  B(F_IN)*F_IN                -> PY
S_AMPA_TC   = (12, 13)   # "AMPA2" block:  C14*F_PY + C3*u_sensory    -> TC
S_GABAA_TC  = (14, 15)   # "GABAa" block:  FB_RE                       -> TC
S_GABAB_TC  = (16, 17)   # "GABAb1" block: B(FB_RE)*FB_RE               -> TC
S_AMPA_RE   = (18, 19)   # "AMPA5" block:  C7*FB_TC + C16*F_PY         -> RE
S_GABAA_RE  = (20, 21)   # "GABAA1" block: phi_RE (tonic, constant)    -> RE
S_LTS_TC    = (22, 23)   # h_n(t) filter inside "TC bursts"
S_LTS_RE    = (24, 25)   # h_n(t) filter inside "RE spikes"
N_STATES = 26


def default_params():
    """All values below are read/solved directly from the .mdl file."""
    return dict(
        g_leak=40.0, V_leak=0.0,           # gL / gL / C6 / C10 / C17 all = 40
        V_AMPA=60.0, V_AMPA_RE=72.0,       # AMPA5 uses a distinct reversal
        V_GABAA=-15.0, V_GABAA_RE_tonic=-3.0,  # GABAA1 (tonic RE) uses -3
        V_GABAB=-30.0,
        # synaptic filter (b0, a1, a0) for TF = b0/(s^2+a1 s+a0)
        AMPA=(23950.0, 2605.0, 262500.0),
        GABAA=(120750.0, 2585.0, 212500.0),
        GABAB=(375.0, 60.0, 800.0),
        LTS=(5200.0, 170.0, 5200.0),
        # GABA_B activation nonlinearity B(F), Eq. 5
        B_theta=8.0, B_k=100.0,
        # cortical firing-rate sigmoid F(V), Eq. 4 (shared by PY, IN)
        F_Gmax=50.0, F_theta=7.0, F_k=2.0,
        # thalamic burst gating sigmoids (Eq. 6), TC and RE differ
        TC_m_theta=6.0, TC_m_k=2.0, TC_n_theta=-16.0, TC_n_k=6.0,
        RE_m_theta=16.0, RE_m_k=2.0, RE_n_theta=-6.0, RE_n_k=6.0,
        C_TC=800.0, C_RE=800.0,
        # coupling gains (the c1-c13-style constants)
        C1=8.0, C2=8.0, C3=6.0, C5=5.0, C7=14.0, C8=1.0, C8m=0.1,
        C9=1.0, C11=10.0, C12=6.0, C14=5.0, C15=2.0, C16=20.0,
    )


def _filter2_deriv(x1, x2, u, b0, a1, a0):
    """One 2-state synaptic/LTS filter's derivative + its output y=b0*x1."""
    dx1 = x2
    dx2 = -a0 * x1 - a1 * x2 + u
    y = b0 * x1
    return dx1, dx2, y


def _F(V, Gmax, theta, k):
    """Firing-rate sigmoid, Eq. 4."""
    return Gmax / (1.0 + np.exp((V - theta) / (-k)))


def _m_inf(V, theta, k):
    return 1.0 / (1.0 + np.exp((V - theta) / (-k)))


def _n_inf(V, theta, k):
    return 1.0 / (1.0 + np.exp((V - theta) / k))


def _B(F, theta, k):
    """GABA_B activation nonlinearity, Eq. 5."""
    return 1.0 / (1.0 + np.exp((F - theta) * (-k)))


def rhs(t, y, p, u_cortical_fn, u_sensory_fn, phi_RE):
    """
    Full 26-state right-hand side.
    u_cortical_fn(t), u_sensory_fn(t): callables -> the two time-varying
    external inputs (Eqs. 8-9). phi_RE: constant (Eq. 10).
    """
    Vpy, Vin, Vtc, Vre = y[V_PY], y[V_IN], y[V_TC], y[V_RE]

    # --- burst-firing pathways (Eq. 6-7) ---
    m_tc = _m_inf(Vtc, p["TC_m_theta"], p["TC_m_k"])
    n_tc = _n_inf(Vtc, p["TC_n_theta"], p["TC_n_k"])
    m_re = _m_inf(Vre, p["RE_m_theta"], p["RE_m_k"])
    n_re = _n_inf(Vre, p["RE_n_theta"], p["RE_n_k"])

    x1, x2 = y[S_LTS_TC[0]], y[S_LTS_TC[1]]
    d_lts_tc1, d_lts_tc2, h_n_tc = _filter2_deriv(x1, x2, n_tc, *p["LTS"])
    FB_TC = p["C_TC"] * m_tc * h_n_tc

    x1, x2 = y[S_LTS_RE[0]], y[S_LTS_RE[1]]
    d_lts_re1, d_lts_re2, h_n_re = _filter2_deriv(x1, x2, n_re, *p["LTS"])
    FB_RE = p["C_RE"] * m_re * h_n_re

    # --- cortical firing rates (Eq. 4) ---
    F_py = _F(Vpy, p["F_Gmax"], p["F_theta"], p["F_k"])
    F_in = _F(Vin, p["F_Gmax"], p["F_theta"], p["F_k"])

    # --- external inputs ---
    u_cortical = u_cortical_fn(t)
    u_sensory = u_sensory_fn(t)

    # --- synaptic filters ---
    x1, x2 = y[S_AMPA_PY[0]], y[S_AMPA_PY[1]]
    in_ampa_py = p["C12"] * u_cortical + p["C15"] * FB_TC
    d_ap1, d_ap2, g_ampa_py = _filter2_deriv(x1, x2, in_ampa_py, *p["AMPA"])

    x1, x2 = y[S_AMPA_IN[0]], y[S_AMPA_IN[1]]
    in_ampa_in = p["C11"] * F_py + p["C8m"] * FB_TC
    d_ai1, d_ai2, g_ampa_in = _filter2_deriv(x1, x2, in_ampa_in, *p["AMPA"])

    x1, x2 = y[S_GABAA_PY[0]], y[S_GABAA_PY[1]]
    d_gap1, d_gap2, g_gabaA_py = _filter2_deriv(x1, x2, F_in, *p["GABAA"])

    x1, x2 = y[S_GABAB_PY[0]], y[S_GABAB_PY[1]]
    b_in = _B(F_in, p["B_theta"], p["B_k"])
    d_gbp1, d_gbp2, g_gabaB_py = _filter2_deriv(x1, x2, b_in * F_in, *p["GABAB"])

    x1, x2 = y[S_AMPA_TC[0]], y[S_AMPA_TC[1]]
    in_ampa_tc = p["C14"] * F_py + p["C3"] * u_sensory
    d_at1, d_at2, g_ampa_tc = _filter2_deriv(x1, x2, in_ampa_tc, *p["AMPA"])

    x1, x2 = y[S_GABAA_TC[0]], y[S_GABAA_TC[1]]
    d_gat1, d_gat2, g_gabaA_tc = _filter2_deriv(x1, x2, FB_RE, *p["GABAA"])

    x1, x2 = y[S_GABAB_TC[0]], y[S_GABAB_TC[1]]
    b_re = _B(FB_RE, p["B_theta"], p["B_k"])
    d_gbt1, d_gbt2, g_gabaB_tc = _filter2_deriv(x1, x2, b_re * FB_RE, *p["GABAB"])

    x1, x2 = y[S_AMPA_RE[0]], y[S_AMPA_RE[1]]
    in_ampa_re = p["C7"] * FB_TC + p["C16"] * F_py
    d_ar1, d_ar2, g_ampa_re = _filter2_deriv(x1, x2, in_ampa_re, *p["AMPA"])

    x1, x2 = y[S_GABAA_RE[0]], y[S_GABAA_RE[1]]
    d_gar1, d_gar2, g_gabaA_re = _filter2_deriv(x1, x2, phi_RE, *p["GABAA"])

    # --- membrane equations: dV/dt = sum(g*(Vsyn - V)) - g_leak*(V-Vleak) ---
    dV_py = (g_ampa_py * (p["V_AMPA"] - Vpy)
             + p["C2"] * g_gabaA_py * (p["V_GABAA"] - Vpy)
             + p["C1"] * g_gabaB_py * (p["V_GABAB"] - Vpy)
             - p["g_leak"] * (Vpy - p["V_leak"]))

    dV_in = (g_ampa_in * (p["V_AMPA"] - Vin)
             - p["g_leak"] * (Vin - p["V_leak"]))

    dV_tc = (g_ampa_tc * (p["V_AMPA"] - Vtc)
             + p["C8"] * g_gabaA_tc * (p["V_GABAA"] - Vtc)
             + p["C9"] * g_gabaB_tc * (p["V_GABAB"] - Vtc)
             - p["g_leak"] * (Vtc - p["V_leak"]))

    dV_re = (g_ampa_re * (p["V_AMPA_RE"] - Vre)
             + p["C5"] * g_gabaA_re * (p["V_GABAA_RE_tonic"] - Vre)
             - p["g_leak"] * (Vre - p["V_leak"]))

    dydt = np.empty(N_STATES)
    dydt[V_PY], dydt[V_IN], dydt[V_TC], dydt[V_RE] = dV_py, dV_in, dV_tc, dV_re
    dydt[S_AMPA_PY[0]], dydt[S_AMPA_PY[1]] = d_ap1, d_ap2
    dydt[S_AMPA_IN[0]], dydt[S_AMPA_IN[1]] = d_ai1, d_ai2
    dydt[S_GABAA_PY[0]], dydt[S_GABAA_PY[1]] = d_gap1, d_gap2
    dydt[S_GABAB_PY[0]], dydt[S_GABAB_PY[1]] = d_gbp1, d_gbp2
    dydt[S_AMPA_TC[0]], dydt[S_AMPA_TC[1]] = d_at1, d_at2
    dydt[S_GABAA_TC[0]], dydt[S_GABAA_TC[1]] = d_gat1, d_gat2
    dydt[S_GABAB_TC[0]], dydt[S_GABAB_TC[1]] = d_gbt1, d_gbt2
    dydt[S_AMPA_RE[0]], dydt[S_AMPA_RE[1]] = d_ar1, d_ar2
    dydt[S_GABAA_RE[0]], dydt[S_GABAA_RE[1]] = d_gar1, d_gar2
    dydt[S_LTS_TC[0]], dydt[S_LTS_TC[1]] = d_lts_tc1, d_lts_tc2
    dydt[S_LTS_RE[0]], dydt[S_LTS_RE[1]] = d_lts_re1, d_lts_re2
    return dydt


def initial_state():
    return np.zeros(N_STATES)


def simulate(t_total, params, u_cortical_fn, u_sensory_fn, phi_RE,
             y0=None, max_step=0.001, rtol=1e-6, atol=1e-8, n_eval=4000):
    if y0 is None:
        y0 = initial_state()
    t_eval = np.linspace(0, t_total, n_eval)
    sol = solve_ivp(rhs, (0, t_total), y0, args=(params, u_cortical_fn, u_sensory_fn, phi_RE),
                     method="RK45", max_step=max_step, rtol=rtol, atol=atol,
                     t_eval=t_eval, dense_output=False)
    return sol


class _FixedStepResult:
    """Mimics the small subset of scipy's OdeResult interface this
    project uses (.t, .y, .success), so callers don't need to branch."""
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.success = True


def simulate_fixed_step(t_total, params, u_cortical_fn, u_sensory_fn, phi_RE,
                         y0=None, dt=0.0005, n_eval=4000):
    """
    Fixed-step RK4 integrator. NEEDED for noise-driven inputs: a zero-
    order-hold random signal is discontinuous every sample, which makes
    adaptive solvers (RK45/etc.) repeatedly reject steps near every
    jump -- 10s of noise-driven simulation can take 40+ seconds with
    solve_ivp's adaptive stepping. Fixed-step marches straight through
    discontinuities in constant time, matching how the original
    Simulink model was actually integrated (fixed-step ode5, dt=0.001).
    Safe for smooth sinusoidal inputs too, just less adaptive.
    """
    if y0 is None:
        y0 = initial_state()
    n_steps = int(np.round(t_total / dt))
    y = y0.copy()
    t = 0.0
    t_out = np.empty(n_steps + 1)
    y_out = np.empty((N_STATES, n_steps + 1))
    t_out[0] = 0.0
    y_out[:, 0] = y

    for i in range(1, n_steps + 1):
        k1 = rhs(t, y, params, u_cortical_fn, u_sensory_fn, phi_RE)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1, params, u_cortical_fn, u_sensory_fn, phi_RE)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2, params, u_cortical_fn, u_sensory_fn, phi_RE)
        k4 = rhs(t + dt, y + dt * k3, params, u_cortical_fn, u_sensory_fn, phi_RE)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        t_out[i] = t
        y_out[:, i] = y

    # resample onto n_eval points for a consistent output size
    t_eval = np.linspace(0, t_total, n_eval)
    y_eval = np.empty((N_STATES, n_eval))
    for row in range(N_STATES):
        y_eval[row] = np.interp(t_eval, t_out, y_out[row])
    return _FixedStepResult(t_eval, y_eval)
