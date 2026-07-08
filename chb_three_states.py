"""
CHB EEG Three States Visualization Script
Displays 1-second windows of:
1. Normal brain activity (baseline)
2. Interictal activity (before seizure with 2Hz-like characteristics)
3. Ictal activity (during seizure with 10Hz-like characteristics)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


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


def create_three_state_plot(output_path="adults_graphs/chb_three_states.png", channel_index=0):
    """
    Creates a 3-plot visualization of CHB EEG data showing:
    1. Normal activity (baseline from file without seizures)
    2. Interictal activity (before seizure onset)
    3. Ictal activity (during seizure)
    
    All plots show exactly 1 second of data.
    
    Known seizure times in CHB-MIT dataset:
    - chb01_03.edf: seizure ~43-90 seconds
    - chb01_04.edf: seizure ~95-106 seconds
    """
    
    data_dir = os.path.join(os.path.dirname(__file__), "eeg data", "chb01_eeg")
    
    # File paths
    normal_file = os.path.join(data_dir, "chb01_01.edf")  # No seizures in this file
    interictal_file = os.path.join(data_dir, "chb01_03.edf")  # Has seizures
    seizure_file = os.path.join(data_dir, "chb01_03.edf")  # Same file with seizure
    
    # Verify files exist
    for f in [normal_file, interictal_file, seizure_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"EEG file not found: {f}")
    
    # Load 1-second segments
    # 1. Normal activity: from beginning of normal file (no seizures)
    normal_seg = EDFReader.load_channel(
        normal_file,
        channel_index=channel_index,
        start_seconds=10.0,  # Start at 10 seconds to avoid any artifacts
        duration_seconds=1.0
    )
    
    # 2. Interictal activity: ~10 seconds before seizure onset in chb01_03
    # Seizure starts around 43 seconds, so interictal at 30 seconds
    interictal_seg = EDFReader.load_channel(
        interictal_file,
        channel_index=channel_index,
        start_seconds=30.0,
        duration_seconds=1.0
    )
    
    # 3. Ictal activity: during seizure in chb01_03 (seizure ~43-90s, use mid-point at 60s)
    seizure_seg = EDFReader.load_channel(
        seizure_file,
        channel_index=channel_index,
        start_seconds=60.0,
        duration_seconds=1.0
    )
    
    # Create time arrays
    fs = normal_seg["fs"]
    t_normal = np.arange(len(normal_seg["signal"])) / fs
    t_interictal = np.arange(len(interictal_seg["signal"])) / fs
    t_seizure = np.arange(len(seizure_seg["signal"])) / fs
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("CHB EEG Data: Three Brain States (1-second windows)", fontsize=16, y=0.98)
    
    # Plot 1: Normal Activity
    axes[0].plot(t_normal, normal_seg["signal"], color="blue", linewidth=0.8)
    axes[0].set_title("1. Normal Activity (Baseline)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Voltage (µV)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1.0)
    
    # Plot 2: Interictal Activity
    axes[1].plot(t_interictal, interictal_seg["signal"], color="green", linewidth=0.8)
    axes[1].set_title("2. Interictal Activity (Before Seizure, ~2Hz-like)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Voltage (µV)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1.0)
    
    # Plot 3: Ictal Activity
    axes[2].plot(t_seizure, seizure_seg["signal"], color="red", linewidth=0.8)
    axes[2].set_title("3. Ictal Activity (During Seizure, ~10Hz-like 'Jump')", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Voltage (µV)", fontsize=11)
    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 1.0)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    
    # Print data info
    print(f"\nData Summary:")
    print(f"  Channel: {normal_seg['name']}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Normal activity window: {normal_file.split('/')[-1]} @ 10-11 seconds")
    print(f"  Interictal window: {interictal_file.split('/')[-1]} @ 30-31 seconds (pre-seizure)")
    print(f"  Ictal window: {seizure_file.split('/')[-1]} @ 60-61 seconds (during seizure)")
    
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    output = create_three_state_plot()
    print(f"\n✓ Complete! Image saved to: {output}")
