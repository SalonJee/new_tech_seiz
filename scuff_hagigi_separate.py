import os
import numpy as np
import matplotlib.pyplot as plt


class EDFReader:
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


def plot_pair_window(normal_path, seizure_path, channel_index, seizure_start, seizure_end, window_duration=60.0):
    window_start = max(0.0, seizure_start - 20.0)
    normal_segment = EDFReader.load_channel(
        normal_path,
        channel_index=channel_index,
        start_seconds=window_start,
        duration_seconds=window_duration,
    )
    seizure_segment = EDFReader.load_channel(
        seizure_path,
        channel_index=channel_index,
        start_seconds=window_start,
        duration_seconds=window_duration,
    )

    if normal_segment["fs"] != seizure_segment["fs"]:
        raise ValueError("Sampling rates differ between normal and seizure files")

    fs = normal_segment["fs"]
    t = np.arange(len(normal_segment["signal"])) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Channel 1 Voltage vs Time: normal {os.path.basename(normal_path)} and seizure {os.path.basename(seizure_path)}",
        fontsize=16,
        y=0.98,
    )

    axes[0].plot(t, normal_segment["signal"], color="tab:blue", linewidth=0.7)
    axes[0].set_title(f"Normal EEG: {os.path.basename(normal_path)} — window {window_start:.0f}s to {window_start + window_duration:.0f}s")
    axes[0].set_ylabel("Voltage")
    axes[0].grid(True)

    axes[1].plot(t, seizure_segment["signal"], color="tab:red", linewidth=0.7)
    axes[1].set_title(f"Seizure EEG: {os.path.basename(seizure_path)} — event {seizure_start:.0f}s to {seizure_end:.0f}s")
    axes[1].set_ylabel("Voltage")
    axes[1].set_xlabel("Relative time in window (s)")
    axes[1].grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    data_folder = os.path.join("eeg data", "chb01_eeg")
    pairs = [
        ("01", "03", 2996.0, 3036.0),
        ("02", "04", 1467.0, 1494.0),
        ("14", "15", 1732.0, 1772.0),
        ("17", "18", 1720.0, 1810.0),
    ]

    os.makedirs("graphs", exist_ok=True)

    for normal_id, seizure_id, seizure_start, seizure_end in pairs:
        normal_path = os.path.join(data_folder, f"chb01_{normal_id}.edf")
        seizure_path = os.path.join(data_folder, f"chb01_{seizure_id}.edf")
        if not os.path.exists(normal_path) or not os.path.exists(seizure_path):
            print(f"Skipping pair {normal_id}/{seizure_id}: file missing")
            continue

        fig = plot_pair_window(
            normal_path,
            seizure_path,
            channel_index=0,
            seizure_start=seizure_start,
            seizure_end=seizure_end,
            window_duration=60.0,
        )
        output_name = f"graphs/normal_{normal_id}_vs_seizure_{seizure_id}.png"
        fig.savefig(output_name, dpi=150)
        plt.close(fig)
        print(f"Saved {output_name}")


if __name__ == "__main__":
    main()
