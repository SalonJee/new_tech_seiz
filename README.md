# Tech Seizures — EEG Analysis & Neural Modeling

A project for analyzing EEG data and simulating neural models to understand seizure patterns and brain activity dynamics.

## 📋 Project Overview

This repository contains:
- **Neural Models**: Hodgkin-Huxley and Suffczynski models for simulating neuronal behavior
- **EEG Analysis**: Tools for processing and analyzing EEG data
- **Seizure Detection**: Scripts for identifying and visualizing seizure events
- **Visualizations**: Graphs and analysis plots (PSDs, voltage-time plots, etc.)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd tech_seizures
```

### 2. Download EEG Data
The EEG data is stored in a shared Google Drive folder. Download and extract it into the project directory:

**📥 Google Drive Link:** `https://drive.google.com/drive/folders/1WiNZ_-sYyue5iaHRdLeP-M7eSIOC4ZMP?usp=drive_link`

After downloading, extract the data so your directory structure looks like:
```
tech_seizures/
├── eeg data/
│   └── chb01_eeg/
│       ├── chb01_01.edf
│       ├── chb01_02.edf
│       ├── ...
│       └── *.edf.seizures  (annotation files)
├── graphs/
├── hodgkin_huxley.py
├── scuffzynski.py
├── ...
└── README.md
```

### 3. Set Up Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run Analysis Scripts
```bash
# Run Hodgkin-Huxley model
python hodgkin_huxley.py

# Run Suffczynski model variants
python scuffzynski.py
python scuff_hagigi_60.py
python scuff_hagigi_irl.py
python scuff_hagigi_separate.py
```

Graphs will be saved to the `graphs/` directory.

## 📁 Directory Structure

```
tech_seizures/
├── hodgkin_huxley.py         # Hodgkin-Huxley neural model simulation
├── scuffzynski.py            # Suffczynski model with random noise
├── scuff_hagigi_60.py        # 60 Hz variant
├── scuff_hagigi_irl.py       # In-vivo recording variant
├── scuff_hagigi_separate.py  # Separated components analysis
├── eeg data/                 # EEG recordings (CHB-MIT dataset)
│   └── chb01_eeg/
│       ├── *.edf             # EDF format EEG files
│       └── *.edf.seizures    # Seizure annotations
├── graphs/                   # Output visualizations
│   ├── channel 1 only/
│   ├── normal_vs_seizure(separate)/
│   ├── psd_vs_frequency/
│   └── voltage_vs_time/
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 📦 Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:
- `numpy` — numerical computing
- `scipy` — scientific algorithms (ODE solvers)
- `matplotlib` — plotting and visualization
- `pyedflib` or `mne` — EEG data reading (if needed)

## 🔬 How to Use

1. **View EEG Data**: Data is stored in `.edf` format with `.edf.seizures` annotation files
2. **Run Models**: Execute Python scripts to generate neural simulations
3. **Analyze Results**: Check `graphs/` folder for visualizations
4. **Modify Parameters**: Edit model files to adjust neural parameters and test hypotheses

## 🛠️ Development

To contribute or modify the code:

```bash
# Activate environment
source venv/bin/activate

# Make your changes, then test
python <script_name>.py

# Stage and commit changes
git add <files>
git commit -m "Description of changes"
git push origin <your-branch>
```

## ❓ Questions?

For issues or questions, please open an issue or contact the repository maintainer.

---

**Last Updated**: 2026-07-03  
**Authors**: @salon-timsina
