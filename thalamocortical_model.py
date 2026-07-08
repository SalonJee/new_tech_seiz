
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ─────────────────────────────────────────────
#  Neural model
# ─────────────────────────────────────────────

NODES = ["PY (cortex)", "IN (cortex)", "TC (thalamus)", "RE (thalamus)"]
NODE_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E"]

DEFAULT_W = [
    [ 0,  -25,  35,   0],  # row = target PY
    [25,    0,   0,   0],  # row = target IN
    [25,    0,   0, -45],  # row = target TC
    [25,    0,  25,   0],  # row = target RE
]


def sigmoid(V, gain=2.5):
    """Firing-rate function. Returns values in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-gain * (V - 2.0)))


def suffczynski_ode(t, y, W, tau, gain, noise_amp, bias):
    """Stochastic ODE — noise drives the TC cell."""
    u = bias + noise_amp * np.random.normal()
    F = sigmoid(y, gain=gain)
    dVdt = (-y + np.dot(W, F)) / tau
    dVdt[2] += u / tau
    return dVdt


def haghighi_ode(t, y, W, tau, gain, freq, sine_amp, bias):
    """Deterministic sine-wave ODE — periodic input drives TC."""
    u = bias + sine_amp * np.sin(2.0 * np.pi * freq * t)
    F = sigmoid(y, gain=gain)
    dVdt = (-y + np.dot(W, F)) / tau
    dVdt[2] += u / tau
    return dVdt


def run_simulation(model, W, tau, duration, steps, **kwargs):
    """Integrate the chosen ODE and return time + 4 traces."""
    t_span = (0.0, duration)
    t_eval = np.linspace(0.0, duration, steps)
    y0 = np.zeros(4)

    if model == "suffczynski":
        sol = solve_ivp(
            lambda t, y: suffczynski_ode(
                t, y, W, tau,
                kwargs["gain"], kwargs["noise_amp"], kwargs["bias"]
            ),
            t_span, y0, t_eval=t_eval, method="RK45"
        )
    else:
        sol = solve_ivp(
            lambda t, y: haghighi_ode(
                t, y, W, tau,
                kwargs["gain"], kwargs["freq"], kwargs["sine_amp"], kwargs["bias"]
            ),
            t_span, y0, t_eval=t_eval, method="RK45"
        )

    return sol.t, sol.y  # y shape: (4, steps)


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thalamocortical Loop Simulator")
        self.resizable(True, True)
        self.configure(bg="#f5f5f0")

        # ── internal state ──
        self.model_var = tk.StringVar(value="suffczynski")

        # Shared
        self.tau_var      = tk.DoubleVar(value=10.0)   # milliseconds (converted on use)
        self.duration_var = tk.DoubleVar(value=1.0)

        # Suffczynski
        self.noise_var  = tk.DoubleVar(value=5.0)
        self.sgain_var  = tk.DoubleVar(value=1.5)
        self.sbias_var  = tk.DoubleVar(value=2.0)

        # Haghighi
        self.freq_var   = tk.DoubleVar(value=10.0)
        self.samp_var   = tk.DoubleVar(value=12.0)
        self.hgain_var  = tk.DoubleVar(value=2.5)
        self.hbias_var  = tk.DoubleVar(value=2.0)

        # Weight matrix (4×4 StringVars)
        self.w_vars = [
            [tk.StringVar(value=str(DEFAULT_W[r][c])) for c in range(4)]
            for r in range(4)
        ]

        self._build_ui()
        self._on_run()   # run once on startup so plot is populated

    # ── layout ──────────────────────────────────

    def _build_ui(self):
        # Left panel (controls) + right panel (plot)
        left = tk.Frame(self, bg="#f5f5f0", padx=12, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = tk.Frame(self, bg="#ffffff")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,0))

        self._build_controls(left)
        self._build_plot(right)

    def _section(self, parent, title):
        frame = tk.LabelFrame(
            parent, text=title,
            bg="#f5f5f0", fg="#444441",
            font=("Helvetica", 10, "bold"),
            padx=8, pady=6, relief=tk.GROOVE, bd=1
        )
        frame.pack(fill=tk.X, pady=(0, 8))
        return frame

    def _slider_row(self, parent, label, var, from_, to, resolution, fmt="{:.1f}"):
        row = tk.Frame(parent, bg="#f5f5f0")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, bg="#f5f5f0", fg="#5F5E5A",
                 font=("Helvetica", 9), width=18, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, textvariable=var, bg="#f5f5f0", fg="#2C2C2A",
                           font=("Helvetica", 9, "bold"), width=5, anchor="e")
        val_lbl.pack(side=tk.RIGHT)
        sl = ttk.Scale(row, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL)
        sl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        # Round display to resolution precision
        def _trace(*_):
            rounded = round(var.get() / resolution) * resolution
            var.set(round(rounded, 6))
        var.trace_add("write", _trace)

    def _build_controls(self, parent):
        # ── Model selector ──
        sec = self._section(parent, "Model")
        for label, val in [("Suffczynski (stochastic)", "suffczynski"),
                            ("Haghighi (deterministic)", "haghighi")]:
            tk.Radiobutton(
                sec, text=label, variable=self.model_var, value=val,
                bg="#f5f5f0", fg="#2C2C2A", font=("Helvetica", 9),
                activebackground="#f5f5f0",
                command=self._toggle_panels
            ).pack(anchor="w")

        # ── Suffczynski params ──
        self.suff_frame = self._section(parent, "Suffczynski parameters")
        self._slider_row(self.suff_frame, "Noise amplitude",   self.noise_var,  0,   15,  0.5)
        self._slider_row(self.suff_frame, "Sigmoid gain",      self.sgain_var,  0.5, 5.0, 0.1)
        self._slider_row(self.suff_frame, "Bias (u₀)",         self.sbias_var,  0,   6.0, 0.5)

        # ── Haghighi params ──
        self.hagh_frame = self._section(parent, "Haghighi parameters")
        self._slider_row(self.hagh_frame, "Frequency (Hz)",    self.freq_var,   1,   30,  1.0)
        self._slider_row(self.hagh_frame, "Sine amplitude",    self.samp_var,   1,   25,  0.5)
        self._slider_row(self.hagh_frame, "Sigmoid gain",      self.hgain_var,  0.5, 6.0, 0.1)
        self._slider_row(self.hagh_frame, "Bias (u₀)",         self.hbias_var,  0,   6.0, 0.5)

        # ── Shared params ──
        shared = self._section(parent, "Shared parameters")
        self._slider_row(shared, "Time const τ (ms)", self.tau_var,      1,  50, 1.0)
        self._slider_row(shared, "Duration (s)",       self.duration_var, 0.5, 3.0, 0.5)

        # ── Synaptic weight matrix ──
        wframe = self._section(parent, "Synaptic weight matrix W")
        tk.Label(wframe, text="Rows = source node, cols = target node",
                 bg="#f5f5f0", fg="#888780", font=("Helvetica", 8)).pack(anchor="w", pady=(0,4))
        header_row = tk.Frame(wframe, bg="#f5f5f0")
        header_row.pack(fill=tk.X)
        tk.Label(header_row, text="", bg="#f5f5f0", width=4).pack(side=tk.LEFT)
        for node in ["PY", "IN", "TC", "RE"]:
            tk.Label(header_row, text=f"→{node}", bg="#f5f5f0", fg="#888780",
                     font=("Helvetica", 8, "bold"), width=6).pack(side=tk.LEFT)
        for r in range(4):
            row_frame = tk.Frame(wframe, bg="#f5f5f0")
            row_frame.pack(fill=tk.X, pady=1)
            tk.Label(row_frame, text=["PY","IN","TC","RE"][r]+":",
                     bg="#f5f5f0", fg="#5F5E5A", font=("Helvetica", 8, "bold"),
                     width=4, anchor="e").pack(side=tk.LEFT)
            for c in range(4):
                e = tk.Entry(row_frame, textvariable=self.w_vars[r][c],
                             width=6, font=("Helvetica", 9),
                             bg="#ffffff", fg="#2C2C2A",
                             relief=tk.FLAT, bd=1,
                             highlightbackground="#d3d1c7",
                             highlightthickness=1)
                e.pack(side=tk.LEFT, padx=2)

        # ── Run button ──
        tk.Button(
            parent, text="▶  Run simulation",
            command=self._on_run,
            bg="#185FA5", fg="white",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, padx=12, pady=6,
            activebackground="#0C447C", activeforeground="white",
            cursor="hand2"
        ).pack(fill=tk.X, pady=(4, 0))

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(parent, textvariable=self.status_var,
                 bg="#f5f5f0", fg="#888780", font=("Helvetica", 8)).pack(pady=4)

        self._toggle_panels()

    def _build_plot(self, parent):
        self.fig = Figure(figsize=(9, 7), dpi=100, facecolor="#ffffff")
        self.axes = [self.fig.add_subplot(4, 1, i+1) for i in range(4)]
        self.fig.subplots_adjust(hspace=0.45, left=0.08, right=0.97,
                                  top=0.94, bottom=0.07)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── logic ────────────────────────────────────

    def _toggle_panels(self):
        m = self.model_var.get()
        if m == "suffczynski":
            self.suff_frame.pack(fill=tk.X, pady=(0, 8), before=self.hagh_frame)
            self.hagh_frame.pack_forget()
        else:
            self.hagh_frame.pack(fill=tk.X, pady=(0, 8), before=self.suff_frame)
            self.suff_frame.pack_forget()

    def _read_weights(self):
        W = np.zeros((4, 4))
        for r in range(4):
            for c in range(4):
                try:
                    W[r][c] = float(self.w_vars[r][c].get())
                except ValueError:
                    messagebox.showerror("Weight error",
                        f"Invalid weight at row {r+1}, col {c+1}. Using 0.")
        return W

    def _on_run(self):
        self.status_var.set("Running…")
        self.update_idletasks()
        try:
            W    = self._read_weights()
            tau  = self.tau_var.get() / 1000.0   # ms → seconds
            dur  = self.duration_var.get()
            steps = int(dur * 2000)
            model = self.model_var.get()

            if model == "suffczynski":
                kwargs = dict(
                    gain      = self.sgain_var.get(),
                    noise_amp = self.noise_var.get(),
                    bias      = self.sbias_var.get(),
                )
            else:
                kwargs = dict(
                    gain     = self.hgain_var.get(),
                    freq     = self.freq_var.get(),
                    sine_amp = self.samp_var.get(),
                    bias     = self.hbias_var.get(),
                )

            t, y = run_simulation(model, W, tau, dur, steps, **kwargs)
            self._update_plot(t, y, model)
            self.status_var.set("Done.")
        except Exception as ex:
            self.status_var.set(f"Error: {ex}")
            messagebox.showerror("Simulation error", str(ex))

    def _update_plot(self, t, y, model):
        model_label = "Suffczynski — stochastic" if model == "suffczynski" \
                      else f"Haghighi — {self.freq_var.get():.0f} Hz sine input"

        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.plot(t, y[i], color=NODE_COLORS[i], linewidth=0.9)
            ax.set_ylabel(NODES[i].split(" ")[0], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            mn, mx = y[i].min(), y[i].max()
            ax.set_ylim(mn - 0.05*abs(mn or 1), mx + 0.05*abs(mx or 1))
            if i == 0:
                ax.set_title(f"Model: {model_label}   τ={self.tau_var.get():.0f} ms",
                             fontsize=9, pad=4)
            if i == 3:
                ax.set_xlabel("Time (s)", fontsize=8)
            else:
                ax.set_xticklabels([])

        self.fig.canvas.draw()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()