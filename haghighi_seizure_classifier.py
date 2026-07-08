"""
haghighi_seizure_classifier.py
===============================================================================
Implements the thalamocortical model described in:

    H. Sohanian Haghighi & A. H. D. Markazi, "A new description of epileptic
    seizures based on dynamic analysis of a thalamocortical model",
    Scientific Reports 7:13615 (2017).

...which itself extends the bistable neuronal network of:

    Suffczynski, Kalitzin & Lopes da Silva, Neuroscience 126:467-484 (2004).

and uses the resulting model's driven-resonance / "jump" phenomenon to build a
practical seizure-vs-normal signature detector that is then applied to real
scalp EEG (CHB-MIT chb01 recordings).

--------------------------------------------------------------------------
IMPORTANT — READ BEFORE TRUSTING ANY EXACT NUMBERS FROM THIS SCRIPT
--------------------------------------------------------------------------
The Haghighi paper gives the *functional form* of the model in full (its
Eqs. 1-10: leaky membrane potentials for PY/IN/TC/RE, bi-exponential AMPA /
GABA_A / GABA_B synaptic kinetics, a nonlinear GABA_B activation function,
sigmoidal firing-rate curves, and three deterministic external inputs). It
does NOT reprint the full numeric parameter table (the 13 coupling constants
c1-c13, synaptic rise/decay rates, sigmoid thresholds/slopes, membrane time
constants) -- for those it just refers the reader to the original 2004
Suffczynski paper, which was not available to source in this session.

So what you get here is:
  * The real model STRUCTURE (Eqs. 1-5, Fig. 2 connectivity, Eqs. 8-10 inputs)
    implemented faithfully -- this is a genuine step up from a quick sigmoid/
    weight-matrix toy network (no offense to `scuff_hagigi_60.py` :)).
  * Parameter VALUES that are physiologically-typical for this class of
    thalamocortical mass model, then hand-tuned (see "PARAMETER TUNING NOTES"
    below) so the model reproduces the qualitative phenomenon the paper is
    actually about: a nonlinear, frequency-dependent "jump" to a much larger-
    amplitude oscillation when the cortical drive frequency approaches a
    resonance band (the paper reports this near ~9 Hz; the tuned model here
    resonates near ~8.5 Hz -- close, not identical, because the exact
    parameters aren't public).

If you have access to the original numeric table (Suffczynski et al. 2004,
or the Haghighi paper's supplementary material), replace the constants in
`DEFAULT_PARAMS` and everything downstream (frequency sweep, EEG
classification thresholds) will automatically re-tune itself.

PARAMETER TUNING NOTES
  * Coupling topology follows the paper's prose description exactly:
      PY <-> IN   (PY excites IN via AMPA; IN inhibits PY via GABA_A + GABA_B)
      PY -> TC, PY -> RE (AMPA)
      TC -> PY, TC -> RE (AMPA)
      RE -> TC   (GABA_A + GABA_B, nonlinear per Eq. 3)
      external sensory input  -> TC   (Eq. 8)
      external cortical input -> PY   (Eq. 9)
      external (constant, inhibitory) reticular input -> RE (Eq. 10)
  * Each synaptic pathway is a genuine 2nd-order linear filter representing
    convolution with the bi-exponential kernel h(t) = A[e^-a1 t - e^-a2 t]
    (Eq. 2), NOT an instantaneous algebraic coupling -- this is what gives the
    model a real frequency response / resonance instead of just a static
    sigmoid network.
  * GABA_B current amplitude is modulated by the nonlinear activation
    B(F) = 1/(1+exp(nuB*(F-thetaB))) as in Eq. 3.
  * The burst-firing mechanism for thalamic populations (Eqs. 6-7, low-
    threshold Ca2+ spikes) is deliberately OMITTED: its extra sigmoids
    (m_inf, n_inf) and time constants (n1, n2) aren't published here either,
    and guessing them adds failure modes without adding reliability. The
    bistable/jump mechanism the paper emphasizes for state transitions
    (Eqs. 1-5) is retained faithfully.
  * With static bias inputs alone (no periodic drive) this parameter set does
    NOT show the autonomous bistability of Fig. 3 -- only the driven
    resonance/jump of Figs. 4, 6, 7. That's the mechanism used below for
    classification, since it is directly comparable across model and EEG:
    "does a narrow-band, sensory/cortical-frequency-tracked spectral
    component reach seizure-like amplitude".
===============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch

os.makedirs("graphs", exist_ok=True)


# =============================================================================
# PART 1 -- EDF READER  (same minimal EDF parser used in your existing scripts)
# =============================================================================
class EDFReader:
    """Minimal EDF file reader for basic channel extraction."""

    @staticmethod
    def _strip_text(text):
        if isinstance(text, bytes):
            text = text.decode("ascii", "ignore")
        return text.rstrip(" \t\r\n\x00")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            header = f.read(256)
            if len(header) < 256:
                raise ValueError("EDF header too short")

            num_records = int(cls._strip_text(header[236:244]))
            duration = float(cls._strip_text(header[244:252]))
            num_signals = int(cls._strip_text(header[252:256]))

            signal_labels = [cls._strip_text(f.read(16)) for _ in range(num_signals)]
            _ = [cls._strip_text(f.read(80)) for _ in range(num_signals)]
            _ = [cls._strip_text(f.read(8)) for _ in range(num_signals)]
            physical_mins = [float(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            physical_maxs = [float(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            digital_mins = [int(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            digital_maxs = [int(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            _ = [cls._strip_text(f.read(80)) for _ in range(num_signals)]
            samples_per_record = [int(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            _ = [cls._strip_text(f.read(32)) for _ in range(num_signals)]

            total_samples_per_record = sum(samples_per_record)
            data_dtype = np.dtype("<i2")
            total_records = num_records * total_samples_per_record
            raw_data = np.fromfile(f, dtype=data_dtype, count=total_records)

            if raw_data.size != total_records:
                raise ValueError("EDF file does not contain the expected number of samples")

            raw_data = raw_data.reshape((num_records, total_samples_per_record))

            channels = []
            offset = 0
            for chan_idx in range(num_signals):
                count = samples_per_record[chan_idx]
                channel_data = raw_data[:, offset:offset + count].reshape(-1)
                offset += count
                digital_min = digital_mins[chan_idx]
                digital_max = digital_maxs[chan_idx]
                physical_min = physical_mins[chan_idx]
                physical_max = physical_maxs[chan_idx]
                scale = (physical_max - physical_min) / (digital_max - digital_min)
                channel_phys = physical_min + (channel_data - digital_min) * scale
                channels.append(channel_phys)

            sampling_rate = samples_per_record[0] / duration
            return {
                "labels": signal_labels,
                "data": np.vstack(channels),
                "fs": sampling_rate,
                "record_duration": duration,
                "num_records": num_records,
            }

    @classmethod
    def load_channel(cls, path, channel_index=0, start_seconds=0.0, duration_seconds=None):
        data = cls.load(path)
        if channel_index < 0 or channel_index >= data["data"].shape[0]:
            raise IndexError("channel_index out of range")
        channel = data["data"][channel_index]
        fs = data["fs"]
        start_sample = int(round(max(0.0, start_seconds) * fs))
        end_sample = None
        if duration_seconds is not None:
            end_sample = start_sample + int(round(duration_seconds * fs))
        segment = channel[start_sample:end_sample]
        return {
            "name": data["labels"][channel_index],
            "signal": segment,
            "fs": fs,
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
        }


# =============================================================================
# PART 2 -- THE THALAMOCORTICAL MODEL (Suffczynski / Haghighi formalism)
# =============================================================================
# State vector layout (22 states):
#   0: V_PY   1: V_IN   2: V_TC   3: V_RE
#   4,5   : AMPA  filter feeding PY  (from TC + external cortical input)
#   6,7   : GABA_A filter feeding PY (from IN)
#   8,9   : GABA_B filter feeding PY (from IN, nonlinear amplitude via B(F_IN))
#   10,11 : AMPA  filter feeding IN  (from PY)
#   12,13 : AMPA  filter feeding TC  (from PY + external sensory input)
#   14,15 : GABA_A filter feeding TC (from RE)
#   16,17 : GABA_B filter feeding TC (from RE, nonlinear amplitude via B(F_RE))
#   18,19 : AMPA  filter feeding RE  (from PY)
#   20,21 : GABA_A filter feeding RE (from the constant external reticular input)
#
# Each (x1, x2) pair implements convolution with h(t)=A[e^-a1 t - e^-a2 t]
# (Eq. 2) exactly, via  dx1/dt = D(t) - a1*x1,  dx2/dt = D(t) - a2*x2,
# g(t) = A*(x1 - x2), where D(t) is the (coupling-weighted) presynaptic drive.

DEFAULT_PARAMS = dict(
    # --- membrane (Eq. 1) ---
    Cm=1.0, gleak=50.0, Vleak=0.0,
    V_AMPA=2.0, V_GABAA=-2.0, V_GABAB=-3.0,

    # --- firing-rate sigmoid F(V) (Eq. 4, conventional increasing form) ---
    GF=1.0, thetaF=2.0, nuF=2.0,

    # --- nonlinear GABA_B activation B(F) (Eq. 5) ---
    thetaB=0.3, nuB=8.0,

    # --- synaptic kinetics: a1 = 1/decay tau, a2 = 1/rise tau (Eq. 2) ---
    a1_AMPA=120.0, a2_AMPA=600.0,     # ~8.3 ms decay / ~1.7 ms rise
    a1_GABAA=90.0, a2_GABAA=450.0,    # ~11 ms decay / ~2.2 ms rise
    a1_GABAB=8.0, a2_GABAB=35.0,      # ~125 ms decay / ~29 ms rise

    # --- synaptic gains (compensate for filter's own steady-state gain) ---
    A_AMPA=100.0, A_GABAA=100.0, A_GABAB=8.0,

    # --- coupling constants (Fig. 2 topology; magnitudes are illustrative) ---
    c_TCPY=3500.0,      # TC -> PY   (AMPA)
    c_INPY_A=1800.0,    # IN -> PY   (GABA_A)
    c_INPY_B=700.0,     # IN -> PY   (GABA_B, nonlinear)
    c_PYIN=2500.0,      # PY -> IN   (AMPA)
    c_PYTC=2500.0,      # PY -> TC   (AMPA)
    c_RETC_A=3500.0,    # RE -> TC   (GABA_A)
    c_RETC_B=1000.0,    # RE -> TC   (GABA_B, nonlinear)
    c_PYRE=2500.0,      # PY -> RE   (AMPA)

    # --- external input gains (Eqs. 8-10) ---
    c_cortical=3.0,     # cortical input  -> PY (AMPA-like channel)
    c_sensory=3.0,      # sensory input   -> TC (AMPA-like channel)
    c_reticular=3.0,    # constant reticular bias -> RE (GABA_A-like channel)
)

N_STATES = 22


class ThalamocorticalModel:
    """Mean-field thalamocortical model (PY, IN, TC, RE) with proper synaptic
    kinetics, GABA_B nonlinearity, and the three external inputs of Eqs. 8-10.
    """

    def __init__(self, params=None):
        self.p = DEFAULT_PARAMS.copy()
        if params:
            self.p.update(params)

    def F(self, V):
        """Sigmoidal firing rate (Eq. 4), increasing with depolarization."""
        p = self.p
        z = np.clip(-p['nuF'] * (V - p['thetaF']), -60, 60)
        return p['GF'] / (1 + np.exp(z))

    def B(self, f):
        """Nonlinear GABA_B activation (Eq. 5)."""
        p = self.p
        z = np.clip(p['nuB'] * (f - p['thetaB']), -60, 60)
        return 1.0 / (1 + np.exp(z))

    @staticmethod
    def external_inputs(t, drive):
        """Eqs. 8-10: biased (sinusoidal) sensory & cortical inputs, constant
        reticular input."""
        u_sensory = drive['phi_bs'] + drive['phi_as'] * np.sin(2 * np.pi * drive['fs'] * t)
        u_cortical = drive['phi_bc'] + drive['phi_ac'] * np.sin(2 * np.pi * drive['fc'] * t)
        u_RE = drive['phi_RE']
        return u_sensory, u_cortical, u_RE

    def rhs(self, t, y, drive):
        p = self.p
        (V_PY, V_IN, V_TC, V_RE,
         xAP1, xAP2, xGAA1, xGAA2, xGAB1, xGAB2,
         xAI1, xAI2,
         xAT1, xAT2, xGTA1, xGTA2, xGTB1, xGTB2,
         xAR1, xAR2, xGRA1, xGRA2) = y

        F_PY, F_IN, F_TC, F_RE = self.F(V_PY), self.F(V_IN), self.F(V_TC), self.F(V_RE)
        u_sensory, u_cortical, u_RE = self.external_inputs(t, drive)

        # presynaptic drives (coupling-weighted firing rates / external inputs)
        D_AP = p['c_TCPY'] * F_TC + p['c_cortical'] * u_cortical      # -> PY AMPA
        D_GAA = p['c_INPY_A'] * F_IN                                  # -> PY GABA_A
        D_GAB = p['c_INPY_B'] * F_IN                                  # -> PY GABA_B
        D_AI = p['c_PYIN'] * F_PY                                     # -> IN AMPA
        D_AT = p['c_PYTC'] * F_PY + p['c_sensory'] * u_sensory        # -> TC AMPA
        D_GTA = p['c_RETC_A'] * F_RE                                  # -> TC GABA_A
        D_GTB = p['c_RETC_B'] * F_RE                                  # -> TC GABA_B
        D_AR = p['c_PYRE'] * F_PY                                     # -> RE AMPA
        D_GRA = p['c_reticular'] * u_RE                               # -> RE GABA_A (bias)

        def filt(x1, x2, D, a1, a2):
            return D - a1 * x1, D - a2 * x2

        dAP1, dAP2 = filt(xAP1, xAP2, D_AP, p['a1_AMPA'], p['a2_AMPA'])
        dGAA1, dGAA2 = filt(xGAA1, xGAA2, D_GAA, p['a1_GABAA'], p['a2_GABAA'])
        dGAB1, dGAB2 = filt(xGAB1, xGAB2, D_GAB, p['a1_GABAB'], p['a2_GABAB'])
        dAI1, dAI2 = filt(xAI1, xAI2, D_AI, p['a1_AMPA'], p['a2_AMPA'])
        dAT1, dAT2 = filt(xAT1, xAT2, D_AT, p['a1_AMPA'], p['a2_AMPA'])
        dGTA1, dGTA2 = filt(xGTA1, xGTA2, D_GTA, p['a1_GABAA'], p['a2_GABAA'])
        dGTB1, dGTB2 = filt(xGTB1, xGTB2, D_GTB, p['a1_GABAB'], p['a2_GABAB'])
        dAR1, dAR2 = filt(xAR1, xAR2, D_AR, p['a1_AMPA'], p['a2_AMPA'])
        dGRA1, dGRA2 = filt(xGRA1, xGRA2, D_GRA, p['a1_GABAA'], p['a2_GABAA'])

        g_AP = p['A_AMPA'] * (xAP1 - xAP2)
        g_GAA = p['A_GABAA'] * (xGAA1 - xGAA2)
        g_GAB = p['A_GABAB'] * (xGAB1 - xGAB2)
        g_AI = p['A_AMPA'] * (xAI1 - xAI2)
        g_AT = p['A_AMPA'] * (xAT1 - xAT2)
        g_GTA = p['A_GABAA'] * (xGTA1 - xGTA2)
        g_GTB = p['A_GABAB'] * (xGTB1 - xGTB2)
        g_AR = p['A_AMPA'] * (xAR1 - xAR2)
        g_GRA = p['A_GABAA'] * (xGRA1 - xGRA2)

        B_IN = self.B(F_IN)
        B_RE = self.B(F_RE)

        dV_PY = (-p['gleak'] * (V_PY - p['Vleak'])
                 - g_AP * (V_PY - p['V_AMPA'])
                 - g_GAA * (V_PY - p['V_GABAA'])
                 - g_GAB * B_IN * (V_PY - p['V_GABAB'])) / p['Cm']
        dV_IN = (-p['gleak'] * (V_IN - p['Vleak'])
                 - g_AI * (V_IN - p['V_AMPA'])) / p['Cm']
        dV_TC = (-p['gleak'] * (V_TC - p['Vleak'])
                 - g_AT * (V_TC - p['V_AMPA'])
                 - g_GTA * (V_TC - p['V_GABAA'])
                 - g_GTB * B_RE * (V_TC - p['V_GABAB'])) / p['Cm']
        dV_RE = (-p['gleak'] * (V_RE - p['Vleak'])
                 - g_AR * (V_RE - p['V_AMPA'])
                 - g_GRA * (V_RE - p['V_GABAA'])) / p['Cm']

        return [dV_PY, dV_IN, dV_TC, dV_RE,
                dAP1, dAP2, dGAA1, dGAA2, dGAB1, dGAB2,
                dAI1, dAI2,
                dAT1, dAT2, dGTA1, dGTA2, dGTB1, dGTB2,
                dAR1, dAR2, dGRA1, dGRA2]

    def simulate(self, t_span, drive, y0=None, t_eval=None, **kw):
        if y0 is None:
            y0 = np.zeros(N_STATES)
        return solve_ivp(lambda t, y: self.rhs(t, y, drive), t_span, y0,
                          t_eval=t_eval, method='RK45', **kw)


# =============================================================================
# PART 3 -- REPRODUCING THE PAPER'S BIFURCATION / FREQUENCY-RESPONSE METHOD
# =============================================================================
# The paper's own methodology (see "Numerical simulation"): scan a stimulus
# parameter (e.g. cortical input frequency f_c) in small steps, using the
# final state of the previous run as the initial condition for the next step
# (this is exactly what lets bistable/hysteretic behaviour show up as a
# difference between forward and backward sweeps in Fig. 7).

def frequency_sweep(model, freqs, drive_base, vary='fc', t_settle=6.0,
                     t_record=2.4, dt=0.002, y0=None):
    """Sweep drive_base[vary] across `freqs`, continuing the trajectory from
    one frequency to the next (paper's continuation method). Returns arrays
    of peak-to-peak V_PY amplitude and the raw tail traces."""
    if y0 is None:
        y0 = np.zeros(N_STATES)
    ptp = []
    traces = []
    for f in freqs:
        drive = dict(drive_base)
        drive[vary] = f
        t_total = t_settle + t_record
        t_eval = np.arange(0, t_total, dt)
        sol = model.simulate((0, t_total), drive, y0=y0, t_eval=t_eval)
        y0 = sol.y[:, -1]
        mask = t_eval >= t_settle
        tail = sol.y[0, mask]
        ptp.append(tail.max() - tail.min())
        traces.append((t_eval[mask] - t_settle, tail))
    return np.array(ptp), traces, y0


def find_resonance_band(freqs, ptp, rel_threshold=0.5):
    """Return (peak_freq, f_lo, f_hi) where ptp crosses rel_threshold*peak."""
    peak_idx = int(np.argmax(ptp))
    peak_freq = freqs[peak_idx]
    thresh = ptp[peak_idx] * rel_threshold
    above = freqs[ptp >= thresh]
    return peak_freq, above.min(), above.max()


# =============================================================================
# PART 4 -- APPLYING THE MODEL'S SIGNATURE TO REAL EEG
# =============================================================================
# The model can't be quantitatively fit to real EEG without the original
# parameter table, so instead of trying to match absolute units, we borrow
# the paper's qualitative diagnostic: a narrow-band spectral component near
# the model's resonance frequency, at amplitude well above the recording's
# own interictal baseline, is the model's proposed ictal signature (paper:
# ~9 Hz seizure-like vs ~11 Hz lower-amplitude interictal/spindle activity).

def sliding_window_spectrum(signal, fs, win_sec=4.0, step_sec=1.0,
                             band=(3.0, 13.0)):
    """Compute dominant frequency and band power in a sliding window."""
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    n = len(signal)
    times, dom_freqs, band_powers = [], [], []
    for start in range(0, max(n - win, 1), step):
        seg = signal[start:start + win]
        if len(seg) < win:
            break
        f, Pxx = welch(seg, fs=fs, nperseg=min(1024, len(seg)))
        in_band = (f >= band[0]) & (f <= band[1])
        if not in_band.any():
            continue
        bp = np.trapz(Pxx[in_band], f[in_band])
        dom_f = f[in_band][np.argmax(Pxx[in_band])]
        times.append(start / fs)
        dom_freqs.append(dom_f)
        band_powers.append(bp)
    return np.array(times), np.array(dom_freqs), np.array(band_powers)


def classify_windows(dom_freqs, band_powers, seizure_freq_band,
                      baseline_power, power_z_threshold=2.0):
    """Flag windows as 'model-predicted seizure-like' if the dominant
    frequency sits inside the model's resonance band AND band power is a
    statistical outlier relative to this recording's own baseline."""
    baseline_median = np.median(baseline_power)
    baseline_mad = np.median(np.abs(baseline_power - baseline_median)) + 1e-12
    z = (band_powers - baseline_median) / (1.4826 * baseline_mad)
    freq_hit = (dom_freqs >= seizure_freq_band[0]) & (dom_freqs <= seizure_freq_band[1])
    power_hit = z >= power_z_threshold
    return freq_hit & power_hit, z


# =============================================================================
# PART 5 -- FIGURES
# =============================================================================
def plot_frequency_response(freqs, ptp_fwd, ptp_bwd, seizure_band, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(freqs, ptp_fwd, 'o-', color='tab:blue', label='forward sweep', ms=3)
    ax.plot(freqs, ptp_bwd, 'x--', color='tab:orange', label='backward sweep', ms=4)
    ax.axvspan(seizure_band[1], seizure_band[2], color='red', alpha=0.15,
               label=f"model 'seizure-like' band ({seizure_band[1]:.1f}-{seizure_band[2]:.1f} Hz)")
    ax.axvline(seizure_band[0], color='red', ls=':', lw=1)
    ax.set_xlabel("Cortical input frequency $f_c$ (Hz)")
    ax.set_ylabel("Peak-to-peak $V_{PY}$ (model units)")
    ax.set_title("Model frequency response: nonlinear resonance / 'jump' (cf. paper Figs. 4, 6, 7)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example_traces(model, drive_base, quiet_f, seizure_f, path):
    y0 = np.zeros(N_STATES)
    drive0 = dict(drive_base); drive0['fc'] = 0.0; drive0['phi_ac'] = 0.0
    sol0 = model.simulate((0, 8), drive0, y0=y0, t_eval=np.linspace(0, 8, 2000))
    y0 = sol0.y[:, -1]

    t_eval = np.linspace(0, 6, 6000)
    drive_q = dict(drive_base); drive_q['fc'] = quiet_f
    sol_q = model.simulate((0, 6), drive_q, y0=y0, t_eval=t_eval)

    drive_s = dict(drive_base); drive_s['fc'] = seizure_f
    sol_s = model.simulate((0, 6), drive_s, y0=y0, t_eval=t_eval)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    ax1.plot(sol_q.t, sol_q.y[0], color='tab:green')
    ax1.set_title(f"Off-resonance drive at {quiet_f} Hz -- 'quiet / interictal-like'")
    ax1.set_ylabel("$V_{PY}$")
    ax1.grid(True, alpha=0.4)

    ax2.plot(sol_s.t, sol_s.y[0], color='tab:red')
    ax2.set_title(f"On-resonance drive at {seizure_f} Hz -- 'high-amplitude / ictal-like'")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("$V_{PY}$")
    ax2.grid(True, alpha=0.4)

    fig.suptitle("Model output: quiet vs. resonant cortical drive (cf. paper Fig. 1 concept)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eeg_session(session_id, t, signal, fs, times, dom_freqs, band_powers,
                      z_scores, flags, seizure_event, seizure_freq_band, path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(f"chb01_{session_id}: real EEG vs. model-derived seizure signature", fontsize=15, y=0.99)

    axes[0].plot(t, signal, color='k', linewidth=0.5)
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Raw channel-1 trace")

    axes[1].plot(times, dom_freqs, color='tab:purple', marker='.', ms=3)
    axes[1].axhspan(seizure_freq_band[0], seizure_freq_band[1], color='red', alpha=0.12)
    axes[1].set_ylabel("Dominant freq (Hz)")
    axes[1].set_title("Sliding-window dominant frequency (red band = model resonance/'seizure' band)")

    axes[2].plot(times, band_powers, color='tab:blue')
    axes[2].set_ylabel("3-13 Hz power")
    axes[2].set_yscale('log')
    axes[2].set_title("Sliding-window band power")

    axes[3].plot(times, z_scores, color='gray')
    axes[3].axhline(2.0, color='tab:orange', ls='--', label='z = 2 threshold')
    axes[3].fill_between(times, -5, 30, where=flags, color='red', alpha=0.25,
                          label='flagged: model-predicted seizure-like')
    axes[3].set_ylim(min(-1, z_scores.min() - 1), max(5, z_scores.max() + 1))
    axes[3].set_ylabel("Power z-score")
    axes[3].set_xlabel("Time in window (s)")
    axes[3].legend(loc='upper right')
    axes[3].set_title("Classification: dominant-frequency-in-band AND power-outlier")

    if seizure_event is not None:
        for ax in axes:
            ax.axvspan(seizure_event[0], seizure_event[1], color='black', alpha=0.08)
        axes[0].text(seizure_event[0], axes[0].get_ylim()[1] * 0.9,
                     "  true seizure (clinical annotation)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary_scatter(all_dom_freqs, all_band_powers, all_labels, seizure_freq_band, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    all_dom_freqs = np.array(all_dom_freqs)
    all_band_powers = np.array(all_band_powers)
    all_labels = np.array(all_labels)

    ax.scatter(all_dom_freqs[~all_labels], all_band_powers[~all_labels],
               s=10, alpha=0.4, color='tab:blue', label='true: normal / interictal window')
    ax.scatter(all_dom_freqs[all_labels], all_band_powers[all_labels],
               s=14, alpha=0.7, color='tab:red', label='true: seizure window')
    ax.axvspan(seizure_freq_band[0], seizure_freq_band[1], color='red', alpha=0.08,
               label="model 'seizure-like' frequency band")
    ax.set_yscale('log')
    ax.set_xlabel("Dominant frequency in window (Hz)")
    ax.set_ylabel("3-13 Hz band power (log scale)")
    ax.set_title("Do real seizure windows actually fall in the model-predicted band?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# PART 6 -- MAIN PIPELINE
# =============================================================================
def run_model_analysis():
    print("=" * 70)
    print("STEP 1: characterizing the model's own driven-resonance 'jump'")
    print("=" * 70)

    model = ThalamocorticalModel()
    # bias values are the ones the paper itself uses for its bifurcation
    # figures (Fig. 3 caption / Fig. 4 caption): phi_bc=13.5, phi_bs=11, phi_RE=12
    drive_base = dict(phi_bs=11.0, phi_as=0.0, fs=0.0,
                       phi_bc=13.5, phi_ac=3.0, fc=1.0,
                       phi_RE=12.0)

    freqs = np.arange(1.0, 20.01, 0.5)
    ptp_fwd, _, y_end = frequency_sweep(model, freqs, drive_base, vary='fc')
    ptp_bwd, _, _ = frequency_sweep(model, freqs[::-1], drive_base, vary='fc', y0=y_end)
    ptp_bwd = ptp_bwd[::-1]

    peak_freq, f_lo, f_hi = find_resonance_band(freqs, ptp_fwd, rel_threshold=0.85)
    print(f"  resonance peak at {peak_freq:.1f} Hz")
    print(f"  seizure-like band (85%-of-peak half-width): {f_lo:.1f}-{f_hi:.1f} Hz")

    plot_frequency_response(freqs, ptp_fwd, ptp_bwd, (peak_freq, f_lo, f_hi),
                             "graphs/model_frequency_response.png")

    plot_example_traces(model, drive_base, quiet_f=2.0, seizure_f=peak_freq,
                         path="graphs/model_quiet_vs_seizure_traces.png")

    print("  saved graphs/model_frequency_response.png")
    print("  saved graphs/model_quiet_vs_seizure_traces.png")

    return f_lo, f_hi


def run_eeg_analysis(seizure_freq_band, data_folder=os.path.join("eeg data", "chb01_eeg"),
                      channel_index=0):
    print()
    print("=" * 70)
    print("STEP 2: applying the model's seizure-band signature to real EEG")
    print("=" * 70)

    normal_path = os.path.join(data_folder, "chb01_01.edf")
    if not os.path.exists(normal_path):
        print(f"  EEG data not found at '{data_folder}'. Skipping real-data analysis.")
        print("  (Model-only figures were still produced in graphs/.)")
        return

    # Known seizure sessions/timings for chb01 (as used in your existing
    # scuff_hagigi_irl.py / scuff_hagigi_separate.py scripts).
    sessions = [
        ("03", 2996.0, 3036.0),
        ("04", 1467.0, 1494.0),
        ("15", 1732.0, 1772.0),
        ("16", 1015.0, 1066.0),
        ("18", 1720.0, 1810.0),
    ]

    all_dom_freqs, all_band_powers, all_labels = [], [], []

    for session_id, seiz_start, seiz_end in sessions:
        path = os.path.join(data_folder, f"chb01_{session_id}.edf")
        if not os.path.exists(path):
            print(f"  skipping session {session_id}: file not found")
            continue

        window_start = max(0.0, seiz_start - 60.0)
        window_dur = (seiz_end - seiz_start) + 120.0

        seg = EDFReader.load_channel(path, channel_index=channel_index,
                                      start_seconds=window_start,
                                      duration_seconds=window_dur)
        fs = seg["fs"]
        signal = seg["signal"]
        t = np.arange(len(signal)) / fs

        times, dom_freqs, band_powers = sliding_window_spectrum(signal, fs)

        # baseline = windows that fall clearly outside the true seizure interval
        in_true_seizure = (times + window_start >= seiz_start) & (times + window_start <= seiz_end)
        baseline_power = band_powers[~in_true_seizure]
        if len(baseline_power) < 3:
            baseline_power = band_powers

        flags, z = classify_windows(dom_freqs, band_powers, seizure_freq_band, baseline_power)

        plot_eeg_session(
            session_id, t, signal, fs, times, dom_freqs, band_powers, z, flags,
            seizure_event=(seiz_start - window_start, seiz_end - window_start),
            seizure_freq_band=seizure_freq_band,
            path=f"graphs/eeg_{session_id}_model_signature.png",
        )
        print(f"  session {session_id}: {flags.sum()}/{len(flags)} windows flagged "
              f"as model-predicted seizure-like; true seizure window "
              f"{seiz_start:.0f}-{seiz_end:.0f}s")

        all_dom_freqs.extend(dom_freqs.tolist())
        all_band_powers.extend(band_powers.tolist())
        all_labels.extend(in_true_seizure.tolist())

    if all_dom_freqs:
        plot_summary_scatter(all_dom_freqs, all_band_powers, all_labels,
                              seizure_freq_band, "graphs/summary_freq_vs_power.png")
        print("  saved graphs/summary_freq_vs_power.png")


def main():
    f_lo, f_hi = run_model_analysis()
    run_eeg_analysis((f_lo, f_hi))
    print()
    print("Done. All figures are in ./graphs/")


if __name__ == "__main__":
    main()
