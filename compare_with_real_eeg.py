"""
compare_with_real_eeg.py
=========================
Compares:
  (a) real EEG features -- ictal segment vs. interictal segment, pulled
      straight out of the CHB-MIT .edf files using the seizure time in
      the accompanying .seizures annotation file, and
  (b) simulated features -- one or more runs exported from app.py
      (see the "Export this run" button added to app.py).

using the exact same feature functions from seizure_features.py, so the
two sides are computed identically.

USAGE
-----
    # just look at the real EEG (no model run yet)
    python compare_with_real_eeg.py --edf chb01_15.edf --seizures chb01_15_edf.seizures

    # compare against one or more exported model runs
    python compare_with_real_eeg.py --edf chb01_15.edf --seizures chb01_15_edf.seizures \
        --runs runs/haghighi_2026....npz runs/suffczynski_2026....npz

    # use chb01_14.edf (no seizures in this file) as a second, independent
    # interictal reference
    python compare_with_real_eeg.py --edf chb01_15.edf --seizures chb01_15_edf.seizures \
        --baseline-edf chb01_14.edf

IMPORTANT CAVEAT -- READ BEFORE TRUSTING ANY NUMBER THIS PRINTS
----------------------------------------------------------------
V_PY (the model output) is in arbitrary model units, NOT microvolts, and
has no scalp/volume-conduction filtering applied. So comparing raw
mean_amplitude or max_amplitude between the model and real EEG directly
is not meaningful (e.g. "12.4" vs "340uV" tells you nothing on its own).

What IS meaningful, and what this script reports front-and-center:
  - the RATIO of ictal-like to interictal-like amplitude/power, computed
    separately within each dataset (real EEG ratio vs. model ratio) --
    this cancels out the arbitrary unit/scale problem.
  - the dominant frequency shift between ictal and interictal epochs --
    frequency is unit-free, so it is directly comparable.
  - whether the *shape* of the change (does amplitude go up during the
    seizure? does dominant frequency shift into a particular band?)
    matches between model and real data -- not whether the absolute
    numbers match.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from edf_reader import read_edf_channel, list_channels
import seizure_features as sf


# ---------------------------------------------------------------------
# .seizures annotation parsing
# ---------------------------------------------------------------------
def read_seizure_time(seizures_path):
    """Parse a CHB-MIT .seizures annotation file.

    Reverse-engineered by comparing raw bytes against the published
    chb01-summary.txt seizure times (not a documented format -- there is
    no public spec for this binary file). Verified against chb01_15.edf
    (recovers seizure_start=1732s, seizure_end=1772s, matching the
    official summary file exactly). Only handles a single seizure per
    file, which covers chb01_15 and most CHB-MIT files but not files
    with multiple seizures.
    """
    b = Path(seizures_path).read_bytes()
    start = (b[38] << 8) | b[41]
    duration = b[49]
    return start, start + duration


# ---------------------------------------------------------------------
# feature extraction from a raw EEG segment
# ---------------------------------------------------------------------
def eeg_segment_features(signal, fs, t0, t1, label):
    seg = signal[int(t0 * fs):int(t1 * fs)]
    feats = sf.feature_summary(seg, fs)
    feats["label"] = label
    feats["t0"], feats["t1"] = t0, t1
    return feats


def print_feature_row(feats):
    print(f"  {feats['label']:<28s} "
          f"mean_amp={feats['mean_amplitude']:>10.3g}  "
          f"max_amp={feats['max_amplitude']:>10.3g}  "
          f"dom_freq={feats['dominant_freq_hz']:>6.2f} Hz  "
          f"band_power(3-12Hz)={feats['band_power']:>10.3g}")


def ratio_summary(ictal_feats, interictal_feats):
    r_amp = ictal_feats["mean_amplitude"] / max(interictal_feats["mean_amplitude"], 1e-12)
    r_pow = ictal_feats["band_power"] / max(interictal_feats["band_power"], 1e-12)
    d_freq = ictal_feats["dominant_freq_hz"] - interictal_feats["dominant_freq_hz"]
    return r_amp, r_pow, d_freq


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf", required=True, help="EDF file containing the seizure")
    ap.add_argument("--seizures", required=True, help=".seizures annotation file for --edf")
    ap.add_argument("--channel", default=None,
                     help="Channel to use (default: first available)")
    ap.add_argument("--baseline-edf", default=None,
                     help="Optional second EDF with no seizures, for an independent baseline")
    ap.add_argument("--runs", nargs="*", default=[],
                     help="One or more .npz files exported from app.py's 'Export this run' button")
    args = ap.parse_args()

    channel = args.channel or list_channels(args.edf)[0]
    print(f"Using channel: {channel}\n")

    signal, fs = read_edf_channel(args.edf, channel)
    t_start, t_end = read_seizure_time(args.seizures)
    print(f"Seizure window from annotation file: {t_start}s -- {t_end}s "
          f"({t_end - t_start}s long)\n")

    ictal = eeg_segment_features(signal, fs, t_start, t_end, "REAL ictal")

    # interictal reference: a same-length window comfortably before the
    # seizure (start of recording), from the SAME file
    baseline_len = t_end - t_start
    interictal_same_file = eeg_segment_features(
        signal, fs, 30, 30 + baseline_len, "REAL interictal (same file)")

    print("Real EEG features:")
    print_feature_row(ictal)
    print_feature_row(interictal_same_file)

    interictal_for_ratio = interictal_same_file

    if args.baseline_edf:
        baseline_signal, baseline_fs = read_edf_channel(args.baseline_edf, channel)
        interictal_other_file = eeg_segment_features(
            baseline_signal, baseline_fs, 30, 30 + baseline_len,
            f"REAL interictal ({Path(args.baseline_edf).name})")
        print_feature_row(interictal_other_file)

    r_amp, r_pow, d_freq = ratio_summary(ictal, interictal_for_ratio)
    print(f"\n  --> REAL EEG: ictal/interictal amplitude ratio = {r_amp:.2f}x, "
          f"band-power ratio = {r_pow:.2f}x, "
          f"dominant-freq shift = {d_freq:+.2f} Hz\n")

    if not args.runs:
        print("No --runs given -- pass one or more exported model runs "
              "(.npz from app.py) to compare against the numbers above.")
        return

    print("Simulated model runs:")
    for run_path in args.runs:
        data = np.load(run_path, allow_pickle=True)
        v_py = data["v_py"]
        fs_sim = float(data["fs"])
        kind = str(data["kind"])
        duration = float(data["duration"])

        # split the simulated run in half: first half as "baseline",
        # second half as "test" -- adjust this if your input has a
        # clear onset/offset structure (e.g. a step-change bias) instead
        half = duration / 2
        sim_first = eeg_segment_features(v_py, fs_sim, 0, half, f"[{kind}] first half")
        sim_second = eeg_segment_features(v_py, fs_sim, half, duration, f"[{kind}] second half")
        print_feature_row(sim_first)
        print_feature_row(sim_second)

        r_amp_sim, r_pow_sim, d_freq_sim = ratio_summary(sim_second, sim_first)
        print(f"  --> {Path(run_path).name}: "
              f"amplitude ratio = {r_amp_sim:.2f}x, band-power ratio = {r_pow_sim:.2f}x, "
              f"dominant-freq shift = {d_freq_sim:+.2f} Hz\n")

    print("Compare the RATIOS/SHIFTS above (real vs. simulated), not the raw "
          "amplitude numbers -- see the module docstring for why.")


if __name__ == "__main__":
    main()