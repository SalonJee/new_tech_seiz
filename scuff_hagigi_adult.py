import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch


TAU = 0.01
W_LOOP = np.array([
    [0, -25, 35, 0],
    [25, 0, 0, 0],
    [25, 0, 0, -45],
    [25, 0, 25, 0],
])


class EDFReader:
    """Minimal EDF reader for single-channel extraction."""

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
            dtype = np.dtype("<i2")
            total_records = num_records * total_samples_per_record
            raw_data = np.fromfile(f, dtype=dtype, count=total_records)

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


def save_model_traces(output_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Thalamocortical model traces for adult-data pipeline", fontsize=16, y=0.98)

    t_suff, v_suff = run_model("suffczynski", duration=10.0, steps=5000, gain=1.0, noise_amp=2.0, bias=2.0)
    t_hagh_quiet, v_hagh_quiet = run_model("haghighi", duration=10.0, steps=5000, freq=2.0, gain=2.5, sine_amp=8.0, bias=2.0)
    t_hagh_seiz, v_hagh_seiz = run_model("haghighi", duration=10.0, steps=5000, freq=10.0, gain=2.5, sine_amp=12.0, bias=2.0)

    axes[0].plot(t_suff, v_suff, color="tab:blue", linewidth=1)
    axes[0].set_title("Suffczynski model output (stochastic input)")
    axes[0].set_ylabel("Voltage")
    axes[0].grid(True)

    axes[1].plot(t_hagh_quiet, v_hagh_quiet, color="tab:green", linewidth=1)
    axes[1].set_title("Haghighi model output (2 Hz sinusoidal input)")
    axes[1].set_ylabel("Voltage")
    axes[1].grid(True)

    axes[2].plot(t_hagh_seiz, v_hagh_seiz, color="tab:red", linewidth=1)
    axes[2].set_title("Haghighi model output (10 Hz sinusoidal input)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Voltage")
    axes[2].grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def parse_time_to_seconds(text):
    text = str(text).strip()
    patterns = re.findall(r"(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})", text)
    if patterns:
        h, m, s = map(int, patterns[0])
        return h * 3600 + m * 60 + s

    parts = [int(p) for p in re.findall(r"\d+", text)]
    if len(parts) >= 3:
        h, m, s = parts[0], parts[1], parts[2]
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 1:
        return parts[0]
    raise ValueError(f"Could not parse time value: {text}")


def parse_seizure_events(annotation_path, target_filename=None):
    events = []
    current = None
    with open(annotation_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("Seizure n"):
                if current is not None:
                    events.append(current)
                current = {"label": raw}
            elif current is not None and raw.startswith("File name:"):
                current["file_name"] = raw.split(":", 1)[1].strip()
            elif current is not None and raw.startswith("Registration start time:"):
                current["recording_start"] = parse_time_to_seconds(raw.split(":", 1)[1])
            elif current is not None and raw.startswith("Registration end time:"):
                current["recording_end"] = parse_time_to_seconds(raw.split(":", 1)[1])
            elif current is not None and raw.startswith("Seizure start time:"):
                current["start"] = parse_time_to_seconds(raw.split(":", 1)[1])
                current["start_seconds"] = current["start"] - current.get("recording_start", 0)
            elif current is not None and raw.startswith("Seizure end time:"):
                current["end"] = parse_time_to_seconds(raw.split(":", 1)[1])
                current["end_seconds"] = current["end"] - current.get("recording_start", 0)

    if current is not None:
        events.append(current)

    if target_filename is None:
        return events

    return [ev for ev in events if ev.get("file_name") == target_filename]


def save_eeg_comparison(edf_path, output_dir, channel_index=0, window_duration=60.0, baseline_padding=60.0):
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"Adult EDF file not found: {edf_path}")

    annotation_path = os.path.join(os.path.dirname(edf_path), "Seizures-list-PN10.txt")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    events = parse_seizure_events(annotation_path, target_filename=os.path.basename(edf_path))
    if not events:
        raise ValueError("No matching seizure events were found in the annotation file")

    output_paths = []
    for event in events:
        event_start = event.get("start_seconds", event.get("start", 0))
        event_end = event.get("end_seconds", event.get("end", 0))
        event_window_start = max(0.0, event_start - 20.0)
        baseline_start = max(0.0, event_window_start - window_duration)

        baseline = EDFReader.load_channel(
            edf_path,
            channel_index=channel_index,
            start_seconds=baseline_start,
            duration_seconds=window_duration,
        )
        seizure = EDFReader.load_channel(
            edf_path,
            channel_index=channel_index,
            start_seconds=event_window_start,
            duration_seconds=window_duration,
        )

        if baseline["fs"] != seizure["fs"]:
            raise ValueError("Sampling rates differ between baseline and seizure segments")

        fs = baseline["fs"]
        t_base = np.arange(len(baseline["signal"])) / fs
        t_seiz = np.arange(len(seizure["signal"])) / fs

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        fig.suptitle(f"Adult EEG comparison: {os.path.basename(edf_path)}", fontsize=16, y=0.98)
        plot_time_series(axes[0], t_base, baseline["signal"], f"Baseline window ({int(baseline_start)}s to {int(baseline_start + window_duration)}s)", "tab:blue")
        plot_time_series(axes[0], t_seiz, seizure["signal"], f"Seizure window ({int(event_window_start)}s to {int(event_window_start + window_duration)}s)", "tab:red")
        axes[0].set_title(f"Time series of channel {channel_index + 1}: {baseline['name']}")
        axes[0].set_xlabel("Relative time in window (s)")

        plot_psd(axes[1], baseline["signal"], fs, "Baseline EEG", "tab:blue")
        plot_psd(axes[1], seizure["signal"], fs, "Seizure EEG", "tab:red")
        axes[1].set_title("PSD comparison of baseline and seizure windows")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        event_label = re.sub(r"[^A-Za-z0-9]+", "_", event["label"]).strip("_").lower()
        output_path = os.path.join(output_dir, f"adult_irl_{event_label}.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        output_paths.append(output_path)

        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig2.suptitle(f"Separate EEG windows: {os.path.basename(edf_path)}", fontsize=16, y=0.98)
        axes2[0].plot(t_base, baseline["signal"], color="tab:blue", linewidth=0.8)
        axes2[0].set_title("Baseline window")
        axes2[0].set_ylabel("Voltage")
        axes2[0].grid(True)

        axes2[1].plot(t_seiz, seizure["signal"], color="tab:red", linewidth=0.8)
        axes2[1].set_title(f"Seizure window (event {event_start} s to {event_end} s)")
        axes2[1].set_xlabel("Relative time in window (s)")
        axes2[1].set_ylabel("Voltage")
        axes2[1].grid(True)

        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        separate_path = os.path.join(output_dir, f"adult_separate_{event_label}.png")
        fig2.savefig(separate_path, dpi=150)
        plt.close(fig2)
        output_paths.append(separate_path)

        baseline_band = power_band(baseline["signal"], fs, 3, 12)
        seizure_band = power_band(seizure["signal"], fs, 3, 12)
        print(f"Event {event_label}: baseline 3-12 Hz power={baseline_band:.3e}, seizure 3-12 Hz power={seizure_band:.3e}, ratio={seizure_band / (baseline_band + 1e-12):.2f}")

    return output_paths


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "adult_eeg_data")
    output_dir = os.path.join(base_dir, "adults_graphs")
    os.makedirs(output_dir, exist_ok=True)

    edf_path = os.path.join(data_dir, "PN10-4.5.6.edf")
    model_path = os.path.join(output_dir, "adult_model_traces.png")
    save_model_traces(model_path)
    print(f"Saved model traces to {model_path}")

    eeg_paths = save_eeg_comparison(edf_path, output_dir, channel_index=0, window_duration=60.0)
    print("Saved adult EEG/IRL plots:")
    for path in eeg_paths:
        print(f"- {path}")

    summary_path = os.path.join(output_dir, "adult_pipeline_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Adult EEG pipeline summary\n")
        f.write(f"EDF file: {edf_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write("Generated files:\n")
        for path in [model_path, *eeg_paths, summary_path]:
            f.write(f"- {path}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
