"""
edf_reader.py
=============
Minimal, dependency-free EDF reader. Only needs numpy (already a project
dependency) -- no pyedflib / mne required. Written against, and tested
against, the CHB-MIT files (chb01_14.edf, chb01_15.edf): 23 channels,
256 Hz, 1-second data records, standard EDF (not EDF+) header.

If your EDF files differ wildly (annotation channels, varying
samples-per-record across channels, etc.) this simple reader may need
extending -- it covers the common single-rate case, which is what
CHB-MIT uses.
"""

import numpy as np


def read_edf_header(path):
    with open(path, "rb") as f:
        main = f.read(256)
        nsig = int(main[252:256])
        n_records = int(main[236:244])
        record_dur = float(main[244:252])

        labels = [f.read(16).decode("ascii", "ignore").strip() for _ in range(nsig)]
        _transducer = [f.read(80) for _ in range(nsig)]
        phys_dim = [f.read(8).decode("ascii", "ignore").strip() for _ in range(nsig)]
        phys_min = [float(f.read(8)) for _ in range(nsig)]
        phys_max = [float(f.read(8)) for _ in range(nsig)]
        dig_min = [float(f.read(8)) for _ in range(nsig)]
        dig_max = [float(f.read(8)) for _ in range(nsig)]
        _prefilter = [f.read(80) for _ in range(nsig)]
        samples_per_record = [int(f.read(8)) for _ in range(nsig)]
        header_bytes = 256 + nsig * 256

    return dict(
        nsig=nsig, n_records=n_records, record_dur=record_dur,
        labels=labels, phys_dim=phys_dim, phys_min=phys_min, phys_max=phys_max,
        dig_min=dig_min, dig_max=dig_max, samples_per_record=samples_per_record,
        header_bytes=header_bytes,
    )


def read_edf_channel(path, channel_name):
    """Read one channel's full time series, in physical units (e.g. uV).

    Returns (signal: np.ndarray, fs: float).
    """
    hdr = read_edf_header(path)
    if channel_name not in hdr["labels"]:
        raise ValueError(f"Channel {channel_name!r} not found. Available: {hdr['labels']}")
    ch = hdr["labels"].index(channel_name)

    spr = hdr["samples_per_record"]
    fs = spr[ch] / hdr["record_dur"]
    scale = (hdr["phys_max"][ch] - hdr["phys_min"][ch]) / (hdr["dig_max"][ch] - hdr["dig_min"][ch])
    offset = hdr["phys_max"][ch] - scale * hdr["dig_max"][ch]

    record_len_samples = sum(spr)  # samples per record, all channels combined
    with open(path, "rb") as f:
        f.seek(hdr["header_bytes"])
        raw = np.fromfile(f, dtype="<i2")

    n_full_records = len(raw) // record_len_samples
    raw = raw[: n_full_records * record_len_samples].reshape(n_full_records, record_len_samples)

    start = sum(spr[:ch])
    end = start + spr[ch]
    digital = raw[:, start:end].reshape(-1).astype(np.float64)
    physical = digital * scale + offset
    return physical, fs


def list_channels(path):
    return read_edf_header(path)["labels"]