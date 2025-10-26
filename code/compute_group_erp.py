"""Compute group-average ERP for specified channels across subjects.

Usage: python .\code\compute_group_erp.py --subjects 01 02 03 --channels O1,Oz,O2
"""

from pathlib import Path
import argparse
import mne
from mne_bids import BIDSPath
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





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


ERP_INTENSITIES = ['non-target_high', 'non-target_mid1', 'non-target_mid2', 'non-target_low']


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


def compute_subject_evokeds(subject, runs, tmin, tmax, bids_root, deriv_vep, args):
    """Compute per-subject combined evokeds grouped by stimulus intensity."""
    intensities = ['high', 'mid1', 'mid2', 'low']
    epochs_by_intensity = {k: [] for k in intensities}
    
    for run in runs:
        fif_vep = find_fif(subject, run, deriv_vep, bids_root)
        if fif_vep is None:
            continue
        
        try:
            raw_v = mne.io.read_raw_fif(fif_vep, preload=False, verbose='ERROR')
            events_v, map_v = get_events_for_name(subject, run, raw_v, bids_root)
            
            if events_v is None or not map_v:
                continue
                
            for name, id_ in map_v.items():
                if name not in ERP_INTENSITIES:
                    continue
                    
                epochs = mne.Epochs(
                    raw_v, events_v, event_id={name: id_},
                    tmin=tmin, tmax=tmax, detrend=args.detrend, preload=True
                )
                
                if args.baseline_tmin is not None and args.baseline_tmax is not None:
                    epochs.apply_baseline((args.baseline_tmin, args.baseline_tmax))
                
                intensity = name.split('_')[-1]
                epochs_by_intensity[intensity].append(epochs)
                
        except Exception as e:
            print(f"⚠️ Failed to process {subject} run {run}: {e}")
            continue
    
    # Combine epochs by intensity
    combined = {}
    for k in intensities:
        epochs_list = epochs_by_intensity.get(k, [])
        if epochs_list:
            try:
                all_epochs = mne.concatenate_epochs(epochs_list)
                combined[k] = all_epochs.average()
            except Exception:
                # Fallback: combine evokeds
                evokeds = [ep.average() for ep in epochs_list]
                combined[k] = mne.combine_evoked(evokeds, weights='equal') if evokeds else None
        else:
            combined[k] = None
    
    return combined


def main(args):
    # Initialize paths based on command line arguments
    bids_root = Path(args.bids_root)
    deriv_vep = bids_root / "derivatives" / "preproc_vep"
    erp_plots = deriv_vep / "erp_plots"
    erp_plots.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] BIDS_ROOT={bids_root}, DERIV_VEP={deriv_vep}, ERP_PLOTS={erp_plots}")
    
    subjects = args.subjects
    runs = args.runs if args.runs else list(range(1, 7))
    # collect per-intensity evokeds across subjects
    intensities = ['high', 'mid1', 'mid2', 'low']
    collected = {k: [] for k in intensities}
    for subj in subjects:
        print(f"Processing subject {subj}...")
        subj_combined = compute_subject_evokeds(subj, runs, args.tmin, args.tmax, bids_root, deriv_vep, args)
        if not subj_combined:
            continue
        for k in intensities:
            ev = subj_combined.get(k)
            if ev is not None:
                collected[k].append(ev)

    if not any(collected[k] for k in intensities):
        print("⚠️ No data found across subjects for any intensity.")
        return

    # For each intensity, we want the list of Evokeds per subject (already in collected)
    # Compute group mean and SEM per timepoint for requested channels
    # Default to O1,Oz,O2 in horizontal columns for a 2x3 figure (top: high vs low, bottom: mid1 vs mid2)
    channels = [c.strip() for c in args.channels.split(',')] 

    colors = {'high': 'C2', 'mid1': 'C1', 'mid2': 'C3', 'low': 'C0'}

    # Precompute mean/sem per (channel,intensity)
    stats = {ch: {} for ch in channels}
    for ch in channels:
        for k in intensities:
            lst = collected.get(k, [])
            if not lst:
                continue
            data_list = []
            times_ref = None
            for ev in lst:
                if ch not in ev.ch_names:
                    continue
                idx = ev.ch_names.index(ch)
                data = ev.data[idx] * 1e6  # µV
                data_list.append(data)
                if times_ref is None:
                    times_ref = ev.times
            if not data_list:
                continue
            arr = np.vstack(data_list)
            mean = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
            stats[ch][k] = {'times': times_ref, 'mean': mean, 'sem': sem, 'n': arr.shape[0]}

    # Determine common y-limits across all subplots (mean ± sem)
    ymins, ymaxs = [], []
    for ch in channels:
        for k in stats.get(ch, {}):
            d = stats[ch][k]
            ymins.append(np.min(d['mean'] - d['sem']))
            ymaxs.append(np.max(d['mean'] + d['sem']))
    if ymins and ymaxs:
        y_min = min(ymins)
        y_max = max(ymaxs)
        # add small padding
        padding = 0.05 * (y_max - y_min) if (y_max - y_min) > 0 else 0.5
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = -2.0, 2.0

    # Create 2x3 figure: columns are channels, rows are comparisons
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    # Row 0: high vs low; Row 1: mid1 vs mid2
    top_ints = ['high', 'low']
    bot_ints = ['mid1', 'mid2']

    # We'll keep track of plotted labels for a global legend
    plotted_labels = []
    for col, ch in enumerate(channels):
        # Top: high vs low
        ax_top = axes[0, col]
        plotted_top = False
        for k in top_ints:
            info = stats.get(ch, {}).get(k)
            if not info:
                continue
            t = info['times']
            mean = info['mean']
            sem = info['sem']
            line = ax_top.plot(t, mean, label=k, color=colors.get(k))
            ax_top.fill_between(t, mean - sem, mean + sem, color=colors.get(k), alpha=0.25)
            if k not in plotted_labels:
                plotted_labels.append(k)
            plotted_top = True
        if not plotted_top:
            ax_top.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax_top.transAxes)
        ax_top.axvline(0, color='k', linestyle='--', label='_nolegend_')
        if col == 0:
            ax_top.set_ylabel('Amplitude (µV)')
        ax_top.set_title(ch)
        ax_top.set_xlim(args.tmin, args.tmax)
        ax_top.set_ylim(y_min, y_max)
        if col == 2:
            ax_top.legend(loc='upper right')

        # Bottom: mid1 vs mid2
        ax_bot = axes[1, col]
        plotted_bot = False
        for k in bot_ints:
            info = stats.get(ch, {}).get(k)
            if not info:
                continue
            t = info['times']
            mean = info['mean']
            sem = info['sem']
            line = ax_bot.plot(t, mean, label=k, color=colors.get(k))
            ax_bot.fill_between(t, mean - sem, mean + sem, color=colors.get(k), alpha=0.25)
            if k not in plotted_labels:
                plotted_labels.append(k)
            plotted_bot = True
        if not plotted_bot:
            ax_bot.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax_bot.transAxes)
        ax_bot.axvline(0, color='k', linestyle='--', label='_nolegend_')
        ax_bot.set_xlabel('Time (s)')
        if col == 0:
            ax_bot.set_ylabel('Amplitude (µV)')
        ax_bot.set_xlim(args.tmin, args.tmax)
        ax_bot.set_ylim(y_min, y_max)

    # add per-subplot legends at lower right using explicit handles/labels so mid1/mid2 appear
    for col, ch in enumerate(channels):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        # top legend
        top_lines = ax_top.get_lines()
        if top_lines:
            top_handles = []
            top_labels = []
            for ln in top_lines:
                lbl = ln.get_label()
                # skip internal/no-legend labels (matplotlib convention: labels starting with '_')
                if not lbl or lbl.startswith('_'):
                    continue
                top_handles.append(ln)
                top_labels.append(lbl)
            if top_handles:
                ax_top.legend(handles=top_handles, labels=top_labels, loc='lower right', fontsize='small', frameon=False)
        # bottom legend
        bot_lines = ax_bot.get_lines()
        if bot_lines:
            bot_handles = []
            bot_labels = []
            for ln in bot_lines:
                lbl = ln.get_label()
                if not lbl or lbl.startswith('_'):
                    continue
                bot_handles.append(ln)
                bot_labels.append(lbl)
            if bot_handles:
                ax_bot.legend(handles=bot_handles, labels=bot_labels, loc='lower right', fontsize='small', frameon=False)
    plt.tight_layout()
    out_tiff = erp_plots / f"group_evoked_all_channels-{'_'.join(channels)}_intensity_sem.tiff"
    plt.savefig(out_tiff, format='tiff', dpi=300)
    plt.close()
    print(f"Saved group plot: {out_tiff}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group ERP analysis (compute grand averages across subjects)')
    # Base preprocessing options (matching preprocessing.py)
    parser.add_argument('--subjects', nargs='+', default=["01","02","03","04","05","06","07","08"], help='subject list')
    parser.add_argument('--runs', nargs='+', type=int, default=[1,2,3,4,5,6], help='run numbers')
    parser.add_argument('--bids_root', type=str, default='.', help='BIDS root path')

    # ERP-specific options
    parser.add_argument('--tmin', type=float, default=-0.2, help='Epoch start time (s) relative to stimulus onset')
    parser.add_argument('--tmax', type=float, default=0.8, help='Epoch end time (s) relative to stimulus onset')
    parser.add_argument('--channels', type=str, default='O1,Oz,O2', help='Comma-separated channel names for ERP analysis')
    parser.add_argument('--baseline_tmin', type=float, default=-0.2, help='Baseline start time (s), pass None to disable baseline correction')
    parser.add_argument('--baseline_tmax', type=float, default=-0.05, help='Baseline end time (s)')
    parser.add_argument('--detrend', type=int, default=1, help='Detrend order to apply to epochs (0=none, 1=linear)')
    
    args = parser.parse_args()
    main(args)
