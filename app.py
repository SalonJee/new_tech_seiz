"""
app.py -- Thalamocortical Loop Simulator (Streamlit)
=======================================================
Flow:
  1. Choose input model: Haghighi (deterministic sinusoid) or
     Suffczynski (stochastic noise) -- one page, toggle switches panels.
  2. Set that model's input parameters, click "Create input" -> shows
     a readable parameter summary + a plot of the raw input signal(s).
  3. Click "Send to neural mass model ->" -> runs the real, reverse-
     engineered thalamocortical model (real_model.py) with that input,
     and shows the output (EEG-proxy V_PY) plus feature info on the
     right half of the screen.
  4. Export the run -> saved to runs/ as a self-describing .npz, for
     compare_with_real_eeg.py to pick up on the command line.

Run with:  streamlit run app.py
"""

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import real_model as rm
import seizure_features as sf

RUNS_DIR = Path("runs")
ICTAL_THRESHOLD = 2.0  # same threshold used in the "Interpretation" line below

st.set_page_config(page_title="Thalamocortical Loop Simulator",
                    layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
# session state
# =====================================================================
defaults = dict(input_created=False, sim_done=False,
                input_t=None, input_cortical=None, input_sensory=None,
                input_params=None, input_kind=None, input_f_s=None,
                u_cortical_fn=None, u_sensory_fn=None, phi_RE=None, duration=None,
                sim_t=None, sim_y=None, sim_features=None)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================================
# small helpers
# =====================================================================

def make_noise_fn(bias, noise_amp, duration, dt=0.001, seed=0):
    rng = np.random.default_rng(seed)
    n = int(np.ceil(duration / dt)) + 2
    vals = rng.normal(0, 1, n)

    def u(t):
        idx = min(int(t / dt), n - 1)
        return bias + noise_amp * vals[idx]
    return u, vals, dt


def eval_fn_over(fn, t_arr):
    return np.array([fn(tt) for tt in t_arr])


# =====================================================================
# header
# =====================================================================
st.markdown(
    "<h2 style='margin-bottom:0'>Thalamocortical Loop Simulator</h2>"
    "<p style='color:#666; margin-top:2px'>"
    "Configure an input on the left, then send it through the "
    "real-parameter thalamocortical model to see the output on the right."
    "</p>", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

# =====================================================================
# LEFT COLUMN -- input configuration
# =====================================================================
with col_left:
    st.subheader("1 · Choose the input model")
    model_choice = st.radio(
        "Input model", ["Haghighi (deterministic sinusoid)", "Suffczynski (stochastic noise)"],
        horizontal=True, label_visibility="collapsed")
    is_haghighi = model_choice.startswith("Haghighi")

    st.subheader("2 · Set input parameters")

    duration = st.slider("Duration (s)", 2.0, 20.0, 8.0, 0.5)

    if is_haghighi:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cortical input** → drives PY")
            phi_bc = st.slider("bias φ_bc", 0.0, 40.0, 13.5, 0.5)
            phi_ac = st.slider("sine amplitude φ_ac", 0.0, 20.0, 0.0, 0.5)
            f_c = st.slider("frequency f_c (Hz)", 0.0, 30.0, 0.0, 0.5)
        with c2:
            st.markdown("**Sensory input** → drives TC")
            phi_bs = st.slider("bias φ_bs", 0.0, 40.0, 11.0, 0.5)
            phi_as = st.slider("sine amplitude φ_as", 0.0, 20.0, 0.0, 0.5)
            f_s = st.slider("frequency f_s (Hz)", 0.0, 30.0, 0.0, 0.5)
        phi_RE = st.slider("reticular bias φ_RE (constant)", 0.0, 30.0, 12.0, 0.5)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cortical input** → drives PY")
            phi_bc = st.slider("mean bias", 0.0, 40.0, 13.5, 0.5, key="s_bc")
            noise_c = st.slider("noise amplitude", 0.0, 20.0, 3.0, 0.5, key="s_nc")
        with c2:
            st.markdown("**Sensory input** → drives TC")
            phi_bs = st.slider("mean bias", 0.0, 40.0, 11.0, 0.5, key="s_bs")
            noise_s = st.slider("noise amplitude", 0.0, 20.0, 3.0, 0.5, key="s_ns")
        phi_RE = st.slider("reticular bias φ_RE (constant)", 0.0, 30.0, 12.0, 0.5, key="s_re")
        seed = st.number_input("random seed", 0, 9999, 42, help="Change this to get a different noise draw.")

    create_clicked = st.button("Create input", type="primary", width="stretch")

    if create_clicked:
        t_plot = np.linspace(0, duration, 2000)
        if is_haghighi:
            u_cortical_fn = lambda t: phi_bc + phi_ac * np.sin(2 * np.pi * f_c * t)
            u_sensory_fn = lambda t: phi_bs + phi_as * np.sin(2 * np.pi * f_s * t)
            params_display = {
                "Cortical bias φ_bc": phi_bc, "Cortical sine amp φ_ac": phi_ac, "Cortical freq f_c (Hz)": f_c,
                "Sensory bias φ_bs": phi_bs, "Sensory sine amp φ_as": phi_as, "Sensory freq f_s (Hz)": f_s,
                "Reticular bias φ_RE": phi_RE, "Duration (s)": duration,
            }
            st.session_state.input_f_s = f_s
        else:
            u_cortical_fn, _, _ = make_noise_fn(phi_bc, noise_c, duration, seed=seed)
            u_sensory_fn, _, _ = make_noise_fn(phi_bs, noise_s, duration, seed=seed + 1)
            params_display = {
                "Cortical mean bias": phi_bc, "Cortical noise amp": noise_c,
                "Sensory mean bias": phi_bs, "Sensory noise amp": noise_s,
                "Reticular bias φ_RE": phi_RE, "Duration (s)": duration, "Seed": seed,
            }
            st.session_state.input_f_s = None

        st.session_state.input_t = t_plot
        st.session_state.input_cortical = eval_fn_over(u_cortical_fn, t_plot)
        st.session_state.input_sensory = eval_fn_over(u_sensory_fn, t_plot)
        st.session_state.u_cortical_fn = u_cortical_fn
        st.session_state.u_sensory_fn = u_sensory_fn
        st.session_state.phi_RE = phi_RE
        st.session_state.duration = duration
        st.session_state.input_params = params_display
        st.session_state.input_kind = model_choice
        st.session_state.input_created = True
        st.session_state.sim_done = False  # invalidate any stale output

    if st.session_state.input_created:
        st.markdown("##### Input summary")
        p = st.session_state.input_params
        cols = st.columns(4)
        for i, (label, val) in enumerate(p.items()):
            cols[i % 4].metric(label, f"{val:.3g}" if isinstance(val, (int, float)) else val)

        st.markdown("##### Input signal")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Cortical input u(t) → PY", "Sensory input u(t) → TC"))
        fig.add_trace(go.Scatter(x=st.session_state.input_t, y=st.session_state.input_cortical,
                                  line=dict(color="#D85A30", width=1), name="cortical"), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.input_t, y=st.session_state.input_sensory,
                                  line=dict(color="#378ADD", width=1), name="sensory"), row=2, col=1)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        fig.update_xaxes(title_text="time (s)", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

        send_clicked = st.button("Send to neural mass model  →", type="primary",
                                  width="stretch", key="send_btn")

        if send_clicked:
            with st.spinner("Running the thalamocortical model..."):
                params = rm.default_params()
                sol = rm.simulate_fixed_step(
                    st.session_state.duration, params,
                    st.session_state.u_cortical_fn, st.session_state.u_sensory_fn,
                    st.session_state.phi_RE, dt=0.0007, n_eval=4000)
                fs_out = 4000 / st.session_state.duration
                st.session_state.sim_t = sol.t
                st.session_state.sim_y = sol.y
                st.session_state.sim_features = sf.feature_summary(sol.y[rm.V_PY], fs_out)
                st.session_state.sim_fs = fs_out
                st.session_state.sim_done = True

# =====================================================================
# RIGHT COLUMN -- neural mass model output
# =====================================================================
with col_right:
    st.subheader("3 · Neural mass model output")

    if not st.session_state.sim_done:
        st.info("Configure an input on the left, click **Create input**, "
                "then **Send to neural mass model** to see the output here.")
    else:
        t = st.session_state.sim_t
        y = st.session_state.sim_y
        feats = st.session_state.sim_features

        show_all = st.checkbox("Show all 4 populations (PY / IN / TC / RE)", value=False)

        fig2 = go.Figure()
        if show_all:
            names = ["PY (cortex)", "IN (cortex)", "TC (thalamus)", "RE (thalamus)"]
            colors = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E"]
            fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=names)
            for i in range(4):
                fig2.add_trace(go.Scatter(x=t, y=y[i], line=dict(color=colors[i], width=1)),
                                row=i + 1, col=1)
            fig2.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            fig2.update_xaxes(title_text="time (s)", row=4, col=1)
        else:
            fig2.add_trace(go.Scatter(x=t, y=y[rm.V_PY], line=dict(color="#378ADD", width=1)))
            fig2.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="time (s)", yaxis_title="V_PY (EEG-proxy)")
        st.plotly_chart(fig2, width="stretch")

        st.markdown("##### Output features")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean amplitude", f"{feats['mean_amplitude']:.3g}")
        m2.metric("Max amplitude", f"{feats['max_amplitude']:.3g}")
        m3.metric("Dominant frequency", f"{feats['dominant_freq_hz']:.2f} Hz")
        m4.metric("Band power (3–12Hz)", f"{feats['band_power']:.3g}")

        # simple, transparent interpretation -- not a black-box classifier
        state_label = "ictal-like (large-amplitude oscillation)" if feats["max_amplitude"] > ICTAL_THRESHOLD \
            else "interictal-like (small-amplitude / resting)"
        st.markdown(f"**Interpretation:** based on a simple amplitude threshold "
                    f"(peak-to-peak > {ICTAL_THRESHOLD}), this output looks **{state_label}**. "
                    f"This is a transparent heuristic, not a validated clinical classifier — "
                    f"see the project README for how to validate this properly against real EEG. "
                    f"**This threshold is *not* used to tag exported runs below** -- that's a "
                    f"manual choice you make yourself, since this heuristic isn't reliable enough "
                    f"to trust as ground truth (e.g. it currently calls the default resting bias "
                    f"'ictal-like' too, which is wrong).")

        st.caption("Note: V_PY is a proxy for the pyramidal population's aggregate membrane "
                   "potential -- the dominant contributor to EEG -- not a biophysically complete "
                   "EEG signal (no volume conduction / scalp filtering / source mixing).")

        st.divider()
        st.markdown("##### 4 · Export this run for comparison against real EEG")
        st.caption("Saves everything `compare_with_real_eeg.py` needs as a single .npz file "
                   "under `runs/`. The filename auto-includes the ictal/interictal tag, max "
                   "amplitude, and dominant frequency, so you can tell runs apart with `ls` "
                   "alone -- no need to open the file.")
        run_name = st.text_input("Optional label (e.g. 'high_bias_test1')", value="",
                                  key="run_name_input")
        tag_choice = st.radio(
            "How do you want to tag this run?",
            ["Interictal (baseline / resting params)", "Ictal (seizure-like params)"],
            horizontal=True, key="tag_choice",
            help="This is YOUR call based on the input params you chose, not the app's guess. "
                 "The amplitude-threshold heuristic above is unreliable as a classifier -- see "
                 "the README -- so tagging is manual here rather than automatic.")
        export_clicked = st.button("Export this run", width="stretch", key="export_btn")
        if export_clicked:
            RUNS_DIR.mkdir(exist_ok=True)
            kind_slug = "haghighi" if st.session_state.input_kind.startswith("Haghighi") else "suffczynski"

            tag = "ictal" if tag_choice.startswith("Ictal") else "interictal"
            meanamp = f"{feats['mean_amplitude']:.2f}".rstrip('0').rstrip('.')
            domfreq = f"{feats['dominant_freq_hz']:.2f}".rstrip('0').rstrip('.')
            label_part = f"-{run_name.strip().replace(' ', '_')}" if run_name.strip() else ""
            fs_label = ""
            if st.session_state.input_f_s is not None:
                f_s_val = f"{st.session_state.input_f_s:.2f}".rstrip('0').rstrip('.')
                fs_label = f"-f_s={f_s_val}Hz"

            fname = RUNS_DIR / f"{kind_slug}-{tag}-meanamp{meanamp}-domfreq{domfreq}hz{fs_label}{label_part}.npz"
            if fname.exists():
                fname = RUNS_DIR / f"{kind_slug}-{tag}-meanamp{meanamp}-domfreq{domfreq}hz{fs_label}{label_part}-{time.strftime('%Y%m%d_%H%M%S')}.npz"

            np.savez(
                fname,
                t=st.session_state.sim_t,
                y=st.session_state.sim_y,          # shape (4, n_eval): PY, IN, TC, RE
                v_py=st.session_state.sim_y[rm.V_PY],
                fs=st.session_state.sim_fs,
                duration=st.session_state.duration,
                kind=st.session_state.input_kind,
                params_json=json.dumps(st.session_state.input_params),
                features_json=json.dumps(st.session_state.sim_features),
            )
            st.success(f"Saved to `{fname}`. Point `compare_with_real_eeg.py --runs {fname}` at it.")