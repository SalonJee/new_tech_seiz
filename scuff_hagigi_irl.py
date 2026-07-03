import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch


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


TAU = 0.01
W_LOOP = np.array([
    [0, -25, 35, 0],
    [25, 0, 0, 0],
    [25, 0, 0, -45],
    [25, 0, 25, 0],
])


def sigmoid(V, gain=2.5):
    return 1.0 / (1.0 + np.exp(-gain * (V - 2.0)))


def suffczynski_ode(t, y, gain=1.0, noise_amp=1.5, bias=2.0):
    u_noise = bias + noise_amp * np.random.normal()
    F = sigmoid(y, gain=gain)
    dVdt = (-y + np.dot(W_LOOP, F)) / TAU
    dVdt[2] += u_noise
    return dVdt


def haghighi_ode(t, y, freq=10.0, gain=2.5, sine_amp=12.0, bias=2.0):
    u_sine = bias + sine_amp * np.cos(2 * np.pi * freq * t)
    F = sigmoid(y, gain=gain)
    dVdt = (-y + np.dot(W_LOOP, F)) / TAU
    dVdt[2] += u_sine
    return dVdt


def run_model(model, duration=10.0, steps=5000, **kwargs):
    t_span = (0.0, duration)
    t_eval = np.linspace(0.0, duration, steps)
    y0 = np.zeros(4)
    if model == "suffczynski":
        sol = solve_ivp(lambda t, y: suffczynski_ode(t, y, **kwargs), t_span, y0, t_eval=t_eval, method="RK45")
    elif model == "haghighi":
        sol = solve_ivp(lambda t, y: haghighi_ode(t, y, **kwargs), t_span, y0, t_eval=t_eval, method="RK45")
    else:
        raise ValueError(f"Unknown model: {model}")
    return sol.t, sol.y[0]


def plot_time_series(ax, time, signal, label, color):
    ax.plot(time, signal, label=label, color=color, linewidth=1)
    ax.set_ylabel("Voltage")
    ax.grid(True)
    ax.legend()


def plot_psd(ax, signal, fs, label, color, nperseg=None):
    nperseg = nperseg or min(2048, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    ax.semilogy(f, Pxx, label=label, color=color)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()


def trapezoidal_integral(y, x):
    if len(x) < 2 or len(y) < 2:
        return 0.0
    dx = np.diff(x)
    return np.sum((y[:-1] + y[1:]) * dx * 0.5)


def power_band(signal, fs, low, high):
    f, Pxx = welch(signal, fs=fs, nperseg=min(2048, len(signal)))
    idx = np.logical_and(f >= low, f <= high)
    return trapezoidal_integral(Pxx[idx], f[idx])


def process_session(session_id, seizure_start, seizure_end, normal_path, data_folder, channel_index, normal_window, seizure_window):
    filename = f"chb01_{session_id}.edf"
    seizure_path = os.path.join(data_folder, filename)
    if not os.path.exists(seizure_path):
        raise FileNotFoundError(f"EEG session file not found: {seizure_path}")

    seizure_segment_start = max(0.0, seizure_start - 20.0)
    normal = EDFReader.load_channel(
        normal_path,
        channel_index=channel_index,
        start_seconds=seizure_segment_start,
        duration_seconds=normal_window,
    )
    seizure = EDFReader.load_channel(
        seizure_path,
        channel_index=channel_index,
        start_seconds=seizure_segment_start,
        duration_seconds=seizure_window,
    )

    fs = normal["fs"]
    t_normal = np.arange(len(normal["signal"])) / fs
    t_seizure = np.arange(len(seizure["signal"])) / fs

    os.makedirs("graphs", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"EEG Channel 1 Comparison: Normal vs Seizure (session {session_id})", fontsize=16, y=0.98)
    plot_time_series(axes[0], t_normal, normal["signal"], f"Normal EEG (chb01_01) — window {int(seizure_segment_start)}s to {int(seizure_segment_start + normal_window)}s", "tab:blue")
    plot_time_series(axes[0], t_seizure, seizure["signal"], f"Seizure EEG (chb01_{session_id}) — window {int(seizure_segment_start)}s to {int(seizure_segment_start + seizure_window)}s", "tab:red")
    axes[0].set_title(f"Time series of EEG Channel 1 ({normal['name']}) — same 60s window")
    axes[0].set_xlabel("Relative time in window (s)")

    plot_psd(axes[1], normal["signal"], fs, "Normal EEG (chb01_01)", "tab:blue")
    plot_psd(axes[1], seizure["signal"], fs, f"Seizure EEG (chb01_{session_id})", "tab:red")
    axes[1].set_title("Power Spectral Density of EEG Channel 1")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path_time = f"graphs/eeg_channel1_normal_vs_seizure_{session_id}.png"
    fig.savefig(path_time, dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig2.suptitle(f"Power Spectral Density Comparison: EEG and Model (session {session_id})", fontsize=16, y=0.98)
    plot_psd(axes2[0], normal["signal"], fs, "Normal EEG (chb01_01)", "tab:blue")
    axes2[0].set_title("EEG Channel 1 PSD — normal data")
    plot_psd(axes2[1], seizure["signal"], fs, f"Seizure EEG (chb01_{session_id})", "tab:red")
    axes2[1].set_title("EEG Channel 1 PSD — seizure data")
    plot_psd(axes2[2], seizure["signal"], fs, f"Seizure EEG (chb01_{session_id})", "tab:red")
    axes2[2].set_title("Seizure segment PSD — model-independent view")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    path_psd = f"graphs/model_and_eeg_psd_{session_id}.png"
    fig2.savefig(path_psd, dpi=150)
    plt.close(fig2)

    normal_band = power_band(normal["signal"], fs, 3, 12)
    seizure_band = power_band(seizure["signal"], fs, 3, 12)

    return {
        "session_id": session_id,
        "session_path": seizure_path,
        "seizure_window": (seizure_segment_start, seizure_segment_start + seizure_window),
        "seizure_event": (seizure_start, seizure_end),
        "normal_band": normal_band,
        "seizure_band": seizure_band,
        "time_path": path_time,
        "psd_path": path_psd,
    }


def main():
    data_folder = os.path.join("eeg data", "chb01_eeg")
    normal_path = os.path.join(data_folder, "chb01_01.edf")
    if not os.path.exists(normal_path):
        raise FileNotFoundError(
            "EEG data file not found. Place chb01_01.edf in eeg data/chb01_eeg/."
        )

    channel_index = 0
    normal_window = 60.0
    seizure_window = 60.0
    sessions = [
        ("03", 2996.0, 3036.0),
        ("04", 1467.0, 1494.0),
        ("15", 1732.0, 1772.00),
        ("16", 1015.0, 1066.0),
        ("18", 1720.0, 1810.0),
    ]

    os.makedirs("graphs", exist_ok=True)

    results = []
    for session_id, seizure_start, seizure_end in sessions:
        result = process_session(
            session_id,
            seizure_start,
            seizure_end,
            normal_path,
            data_folder,
            channel_index,
            normal_window,
            seizure_window,
        )
        results.append(result)

    fig_model, axes_model = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig_model.suptitle("Thalamocortical Model Time Traces (Suffczynski and Haghighi)", fontsize=16, y=0.98)
    t_suff, v_suff = run_model("suffczynski", duration=10.0, steps=5000, gain=1.0, noise_amp=2.0, bias=2.0)
    t_hagh_quiet, v_hagh_quiet = run_model("haghighi", duration=10.0, steps=5000, freq=2.0, gain=2.5, sine_amp=8.0, bias=2.0)
    t_hagh_seiz, v_hagh_seiz = run_model("haghighi", duration=10.0, steps=5000, freq=10.0, gain=2.5, sine_amp=12.0, bias=2.0)
    plot_time_series(axes_model[0], t_suff, v_suff, "Suffczynski model: stochastic normal", "tab:blue")
    axes_model[0].set_title("Suffczynski model output (stochastic input) — assumed normal-like state")
    plot_time_series(axes_model[1], t_hagh_quiet, v_hagh_quiet, "Haghighi model: quiet 2 Hz", "tab:green")
    axes_model[1].set_title("Haghighi model output (2 Hz sinusoidal input) — quiet/interictal-like state")
    plot_time_series(axes_model[2], t_hagh_seiz, v_hagh_seiz, "Haghighi model: seizure 10 Hz", "tab:red")
    axes_model[2].set_title("Haghighi model output (10 Hz sinusoidal input) — seizure/ictal-like state")
    axes_model[2].set_xlabel("Time (s)")
    fig_model.tight_layout(rect=[0, 0, 1, 0.96])
    fig_model.savefig("graphs/model_traces.png", dpi=150)
    plt.close(fig_model)

    print("=== Validation summary ===")
    print(f"Normal file: {normal_path}")
    print(f"Using EDF channel index: {channel_index}")
    for result in results:
        print("---")
        print(f"Session: {result['session_id']}")
        print(f"  Record file: {result['session_path']}")
        print(f"  Seizure event: {result['seizure_event'][0]:.0f}s to {result['seizure_event'][1]:.0f}s")
        print(f"  Plot window: {result['seizure_window'][0]:.0f}s to {result['seizure_window'][1]:.0f}s (same for normal and seizure)")
        print(f"  Normal 3-12 Hz power: {result['normal_band']:.3e}")
        print(f"  Seizure 3-12 Hz power: {result['seizure_band']:.3e}")
        print(f"  PSD ratio: {result['seizure_band'] / (result['normal_band'] + 1e-12):.2f}")
        print(f"  Saved time series: {result['time_path']}")
        print(f"  Saved PSD: {result['psd_path']}")


if __name__ == "__main__":
    main()
