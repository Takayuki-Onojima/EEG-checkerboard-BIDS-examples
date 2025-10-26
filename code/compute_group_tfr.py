"""Compute group-average time-frequency representation (TFR) for posterior ROI.

Usage: python .\code\compute_group_tfr.py --subjects 01 02 03 --channels O1,Oz,O2
"""

from pathlib import Path
import argparse
import mne
from mne_bids import BIDSPath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_fif(subject, run, root, bids_root):
    """Find preprocessed .fif file for given subject and run."""
    bids_path = BIDSPath(subject=subject, task="visstim", run=f"{int(run):02d}", datatype="eeg", root=bids_root)
    fif_path = bids_path.copy().update(root=root, suffix="eeg", extension=".fif").fpath
    return fif_path if fif_path.exists() else None


def load_events_from_bids(subject, run, bids_root):
    """Load BIDS events.tsv file for given subject and run."""
    bids_path = BIDSPath(subject=subject, task="visstim", run=f"{int(run):02d}", datatype="eeg", root=bids_root)
    events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
    if events_tsv.exists():
        try:
            return pd.read_csv(events_tsv, sep='\t')
        except Exception as e:
            print(f"⚠️ Failed to read {events_tsv}: {e}")
    return None


TFR_INTENSITIES = ['non-target_high', 'non-target_low']


def get_events_for_name(subject, run, raw, bids_root):
    """Load events from BIDS and convert to MNE events format."""
    df = load_events_from_bids(subject, run, bids_root)
    if df is not None and 'onset' in df.columns and len(df) > 0:
        try:
            times = df['onset'].astype(float).values
            durations = df['duration'].fillna(0).astype(float).values if 'duration' in df.columns else [0]*len(times)
            descriptions = df['trial_type'].fillna('stim').values
            annotations = mne.Annotations(onset=times, duration=durations, description=descriptions)
            raw.set_annotations(annotations)
            return mne.events_from_annotations(raw)
        except Exception as e:
            print(f"⚠️ Failed to build annotations from events.tsv: {e}")
    
    # Fallback to existing annotations
    if len(raw.annotations) > 0:
        try:
            return mne.events_from_annotations(raw)
        except Exception as e:
            print(f"⚠️ Failed to get events from raw annotations: {e}")
    
    return None, {}


def compute_subject_tfr(subject, runs, tmin, tmax, freqs, n_cycles, bids_root, deriv_vep, channels, intensities, args):
    """Compute TFR for each intensity per subject."""
    subject_epochs = {intensity: {ch: [] for ch in channels} for intensity in intensities}
    trial_counts = {intensity: 0 for intensity in intensities}
    times_ref = None
    
    for run in runs:
        fif_path = find_fif(subject, run, deriv_vep, bids_root)
        if fif_path is None:
            continue
            
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=False, verbose='ERROR')
            events, event_map = get_events_for_name(subject, run, raw, bids_root)
            
            if events is None or not event_map:
                continue
                
            for event_name, event_id in event_map.items():
                if event_name not in TFR_INTENSITIES:
                    continue
                
                intensity = event_name.split('_')[-1]  # 'non-target_high' → 'high'
                
                # Create epochs
                epochs = mne.Epochs(
                    raw, events, event_id={event_name: event_id},
                    tmin=tmin - args.padding, tmax=tmax + args.padding, 
                    preload=True, verbose='ERROR'
                )
                
                if len(epochs) == 0:
                    continue
                
                print(f"  Subject {subject} Run {run}: {event_name} -> {intensity} ({len(epochs)} trials)")
                trial_counts[intensity] += len(epochs)
                
                # Store epochs for each channel
                for ch in channels:
                    if ch not in epochs.ch_names:
                        continue
                    
                    epochs_ch = epochs.copy().pick([ch])
                    subject_epochs[intensity][ch].append(epochs_ch)
                    
                    if times_ref is None:
                        times_ref = epochs_ch.times
                        
        except Exception as e:
            print(f"⚠️ Failed to process {subject} run {run}: {e}")
            continue
    
    # Pool all epochs per subject and compute TFR
    results = {intensity: {} for intensity in intensities}
    
    for intensity in intensities:
        for ch in channels:
            if subject_epochs[intensity][ch]:
                # Concatenate all epochs for this subject/intensity/channel
                all_epochs = mne.concatenate_epochs(subject_epochs[intensity][ch])
                
                try:
                    # Compute TFR on all pooled epochs
                    power = mne.time_frequency.tfr_morlet(
                        all_epochs, freqs=freqs, n_cycles=n_cycles, 
                        use_fft=True, return_itc=False, average=True  # Average across trials
                    )
                    
                    # Extract averaged data
                    data_avg = power.data[0]  # (n_freqs, n_times)
                    results[intensity][ch] = (data_avg, power.times)
                    
                except Exception as e:
                    print(f"⚠️ TFR failed for subject {subject} ch {ch} intensity {intensity}: {e}")
                    continue
    
    print(f"  Subject {subject} trial counts: {trial_counts}")
    return results


def main(args):
    # Initialize paths
    bids_root = Path(args.bids_root)
    deriv_vep = bids_root / "derivatives" / "preproc_vep"
    tfr_plots = deriv_vep / "tfr_plots"
    tfr_plots.mkdir(parents=True, exist_ok=True)
    
    subjects = args.subjects
    runs = args.runs if args.runs else list(range(1, 7))
    channels = [c.strip() for c in args.channels.split(',')]
    intensities = ['high', 'low']
    
    # TFR parameters
    freqs = np.arange(args.freq_min, args.freq_max + 1, 1)  
    n_cycles = freqs / 2.0
    
    # Collect TFR data across subjects
    group_results = {intensity: {ch: [] for ch in channels} for intensity in intensities}
    times_ref = None
    
    print(f"Processing {len(subjects)} subjects with intensities: {intensities}")
    print(f"Target events: {TFR_INTENSITIES}")
    
    for subject in subjects:
        print(f"Processing subject {subject}...")
        subject_results = compute_subject_tfr(
            subject, runs, args.tmin, args.tmax, freqs, n_cycles, 
            bids_root, deriv_vep, channels, intensities, args
        )
        
        for intensity in intensities:
            for ch in channels:
                if ch in subject_results[intensity]:
                    data, times = subject_results[intensity][ch]
                    group_results[intensity][ch].append(data)
                    if times_ref is None:
                        times_ref = times
    
    # Report data collection summary
    for intensity in intensities:
        for ch in channels:
            count = len(group_results[intensity][ch])
            print(f"Collected {count} TFR datasets for {intensity} intensity, channel {ch}")
    
    if times_ref is None:
        print("⚠️ No TFR data collected across subjects")
        return
    
    # Apply baseline correction and compute group averages
    group_db = {intensity: {} for intensity in intensities}
    bmask = (times_ref >= args.baseline_min) & (times_ref <= args.baseline_max)
    
    for intensity in intensities:
        for ch in channels:
            subj_data = group_results[intensity][ch]
            if not subj_data:
                continue
                
            # Stack subject data
            arr = np.stack(subj_data, axis=0)  # (n_subjects, n_freqs, n_times)
            
            # Baseline correction in dB
            if np.any(bmask):
                baseline_mean = np.mean(arr[:, :, bmask], axis=2, keepdims=True)
                baseline_mean[baseline_mean == 0] = np.finfo(float).eps
                arr_db = 10 * np.log10(arr / baseline_mean)
            else:
                arr_db = 10 * np.log10(arr)
                
            group_db[intensity][ch] = np.mean(arr_db, axis=0)
    
    # Prepare display window
    dt_mask = (times_ref >= args.display_tmin) & (times_ref <= args.display_tmax)
    times_disp = times_ref[dt_mask]
    
    # Determine color scale
    all_vals = []
    for intensity in intensities:
        for ch in channels:
            if ch in group_db[intensity]:
                all_vals.append(group_db[intensity][ch][:, dt_mask])
    
    if not all_vals:
        print("⚠️ No data to plot")
        return
        
    all_data = np.concatenate([d.flatten() for d in all_vals])
    absmax = np.nanmax(np.abs(all_data))
    vmin, vmax = -absmax, absmax
    
    # Create plot: 2 rows (intensities) x N cols (channels)
    n_channels = len(channels)
    fig, axes = plt.subplots(2, n_channels, figsize=(4*n_channels, 6), 
                            sharex=True, sharey=True)
    if n_channels == 1:
        axes = axes[:, np.newaxis]
    
    for row, intensity in enumerate(intensities):
        for col, ch in enumerate(channels):
            ax = axes[row, col]
            
            if ch in group_db[intensity]:
                data = group_db[intensity][ch][:, dt_mask]
                im = ax.imshow(data, aspect='auto', origin='lower',
                             extent=[times_disp[0], times_disp[-1], freqs[0], freqs[-1]],
                             cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.axvline(0, color='k', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center')
            
            if row == 1:
                ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'{ch} ({intensity})')
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
    cbar.set_label('Power (dB)')
    
    # Save figure
    channel_str = '_'.join(channels)
    out_tiff = tfr_plots / f"group_tfr_channels-{channel_str}_{args.freq_min}-{args.freq_max}Hz.tiff"
    plt.savefig(out_tiff, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved TFR plot: {out_tiff}")
    
    # Save caption
    caption = "Transient alpha-band (8-13 Hz) suppression (ERD) observed during stimulus presentation (non-target trials only)."
    caption_file = tfr_plots / f"group_tfr_channels-{channel_str}_caption.txt"
    with open(caption_file, 'w', encoding='utf-8') as f:
        f.write(caption)
    print(f"Saved caption: {caption_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group TFR analysis (time-frequency representation)')
    
    # Base preprocessing options
    parser.add_argument('--subjects', nargs='+', default=["01","02","03","04","05","06","07","08"], help='subject list')
    parser.add_argument('--runs', nargs='+', type=int, default=[1,2,3,4,5,6], help='run numbers')
    parser.add_argument('--bids_root', type=str, default='.', help='BIDS root path')
    
    # TFR-specific options
    parser.add_argument('--tmin', type=float, default=-0.5, help='Epoch start time (s)')
    parser.add_argument('--tmax', type=float, default=1.0, help='Epoch end time (s)')
    parser.add_argument('--channels', type=str, default='O1,Oz,O2', help='Comma-separated channel names for TFR analysis')
    parser.add_argument('--baseline_min', type=float, default=-0.2, help='Baseline start time (s)')
    parser.add_argument('--baseline_max', type=float, default=-0.05, help='Baseline end time (s)')
    parser.add_argument('--display_tmin', type=float, default=-0.5, help='Display window start (s)')
    parser.add_argument('--display_tmax', type=float, default=1.0, help='Display window end (s)')
    parser.add_argument('--freq_min', type=int, default=4, help='Minimum frequency (Hz)')
    parser.add_argument('--freq_max', type=int, default=40, help='Maximum frequency (Hz)')
    parser.add_argument('--padding', type=float, default=0.5, help='Padding to avoid edge effects (s)')
    
    args = parser.parse_args()
    main(args)
