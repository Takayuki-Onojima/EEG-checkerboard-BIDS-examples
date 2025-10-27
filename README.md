# Code release for the manuscript: "High-resolution EEG dataset during resting and localized visual stimulation in the lower-left visual field"

The `./code/` directory contains sample preprocessing and group-analysis scripts intended for public release alongside the manuscript.

## Scripts Overview

### 1. Preprocessing Pipeline
- `preprocessing.py` - Main preprocessing script for VEP analysis
  - Band-pass filtering (0.1–30 Hz)
  - Resampling to specified frequency (default 1000 Hz)
  - ICA-based EOG correction for blinks and horizontal eye movements
  - Outputs cleaned FIF files to `derivatives/preproc_vep/`

### 2. Group-Level Analysis
- `compute_group_erp.py` - Computes group-level ERP averages and plots
- `compute_group_tfr.py` - Computes group-level time-frequency representations using Morlet wavelets

---

## Usage Examples

### Preprocessing
```bash
python preprocessing.py --subjects 01 02 --runs 1 2 3 --bids_root .. --right_ear X2 --sfreq 1000
```

### Group ERP Analysis
```bash
python compute_group_erp.py --subjects 01 02 03 --runs 1 2 3 --tmin -0.3 --tmax 0.8 --baseline_tmin -0.2 --baseline_tmax -0.05
```

### Group TFR Analysis
```bash
python compute_group_tfr.py --subjects 01 02 03 --runs 1 2 3 --tmin -0.5 --tmax 1.0 --baseline_min -0.2 --baseline_max -0.05
```

---

## Implementation Details

### Preprocessing (`preprocessing.py`)
- Loads raw data from BIDS using `read_raw_bids`
- Optional re-referencing to linked earlobes
- ICA-based EOG correction using both VEOG and HEOG channels
- Robust ICA fitting strategy with fallback options
- Saves cleaned data with JSON metadata

### Group ERP (`compute_group_erp.py`)
- Maps event descriptions to intensity labels (high, mid1, mid2, low)
- Concatenates epochs across runs per subject
- Computes group mean and SEM for selected channels (O1, Oz, O2)
- Handles baseline correction consistently across runs

### Group TFR (`compute_group_tfr.py`)
- Computes time-frequency maps using Morlet wavelets (4-40 Hz)
- Per-subject baseline conversion to dB
- Averages across subjects for group-level analysis

---

## Requirements

Install required packages:
```bash
pip install mne pandas numpy matplotlib seaborn
```

