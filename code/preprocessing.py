import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
import numpy as np
import matplotlib
# use a non-interactive backend for headless environments (file saving)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# VEP-only preprocessing script

def detect_eog_components(ica, raw_for_ica, eog_candidates):
    """Detect EOG components using threshold and correlation-based methods"""
    eog_inds_set = set()
    eog_source_map = {}
    eog_detect_method = {}
    eog_epochs_list = []
    
    for eog_ch in eog_candidates:
        try:
            # Remove DC / mean from the EOG candidate
            try:
                if eog_ch in raw_for_ica.ch_names:
                    eidx = raw_for_ica.ch_names.index(eog_ch)
                    sig = raw_for_ica.get_data(picks=[eog_ch])[0]
                    meanv = float(np.nanmean(sig))
                    raw_for_ica._data[eidx, :] = raw_for_ica._data[eidx, :] - meanv
                    print(f"    Zero-meaned EOG channel {eog_ch} (mean {meanv:.3e}) for ICA detection")
            except Exception:
                pass
            
            # Create EOG epochs and try threshold-based detection
            eog_epochs = create_eog_epochs(raw_for_ica, ch_name=eog_ch)
            eog_epochs_list.append(eog_epochs)
            inds, scores = ica.find_bads_eog(eog_epochs, ch_name=eog_ch, threshold=3.0)
            
            # Record detection method for threshold-based detection
            if inds:
                for comp in inds:
                    eog_detect_method[comp] = "threshold=3.0"
            
            # If strict threshold found nothing, try laxer threshold
            if not inds:
                try:
                    inds2, scores2 = ica.find_bads_eog(eog_epochs, ch_name=eog_ch, threshold=2.0)
                    if inds2:
                        inds = inds2
                        scores = scores2
                        # Record detection method for laxer threshold
                        for comp in inds:
                            eog_detect_method[comp] = "threshold=2.0"
                except Exception:
                    pass

            # If still nothing, try correlation-based detection
            if not inds:
                inds = _correlation_based_detection(ica, raw_for_ica, eog_ch, eog_source_map, eog_detect_method)

            if inds:
                eog_inds_set.update(inds)
                for comp in inds:
                    eog_source_map.setdefault(comp, []).append(eog_ch)
            print(f"    EOG detection using {eog_ch} found components: {inds}")
        except Exception as e:
            print(f"    EOG detection using {eog_ch} failed: {e}")
    
    return sorted(list(eog_inds_set)), eog_source_map, eog_detect_method, eog_epochs_list

def _correlation_based_detection(ica, raw_for_ica, eog_ch, eog_source_map, eog_detect_method):
    """Perform correlation-based EOG component detection"""
    try:
        src_raw = ica.get_sources(raw_for_ica)
        src_data = src_raw.get_data()
        if not src_data.size:
            return []
        
        # Get EOG signal for correlation
        eog_sig = raw_for_ica.get_data(picks=[eog_ch])[0]
        corrs = []
        for ci in range(src_data.shape[0]):
            try:
                c = np.corrcoef(src_data[ci], eog_sig)[0, 1]
            except Exception:
                c = 0.0
            if np.isnan(c):
                c = 0.0
            corrs.append(c)
        corrs = np.array(corrs)
        
        # Try progressively laxer thresholds
        corr_thresholds = [0.30, 0.20, 0.15]
        found_sel = []
        for corr_thresh in corr_thresholds:
            sel = np.where(np.abs(corrs) >= corr_thresh)[0].tolist()
            if sel:
                found_sel = sel
                for comp in sel:
                    eog_source_map.setdefault(comp, []).append(f"corr({eog_ch})={corrs[comp]:.2f}")
                    eog_detect_method[comp] = f"corr>={corr_thresh}"
                print(f"    Correlation-based detection using {eog_ch} found components: {sel} (corr_thresh={corr_thresh})")
                break
        
        # If still nothing, pick top-k correlated components
        if not found_sel:
            try:
                k = min(3, src_data.shape[0])
                if k > 0:
                    top_idx = np.argsort(-np.abs(corrs))[:k].tolist()
                    found_sel = top_idx
                    for comp in top_idx:
                        eog_source_map.setdefault(comp, []).append(f"corr_top({eog_ch})={corrs[comp]:.2f}")
                        eog_detect_method[comp] = f"top{len(top_idx)}"
                    print(f"    Top-{k} correlation fallback using {eog_ch} selected components: {top_idx}")
            except Exception as e:
                print(f"    Top-k correlation fallback failed for {eog_ch}: {e}")
        
        return found_sel
    except Exception as e:
        print(f"    Correlation fallback failed for {eog_ch}: {e}")
        return []

def save_ica_diagnostics(ica, eog_inds, eog_source_map, eog_detect_method, eog_epochs_list, 
                        out_path_vep, subject, run, raw_for_ica):
    """Save ICA diagnostic outputs (text summary and plots)"""
    try:
        ica_dir = out_path_vep.fpath.parent / 'ica_plots' / f"sub-{subject}_run-{int(run):02d}"
        ica_dir.mkdir(parents=True, exist_ok=True)
        
        # Write text summary
        _save_exclusion_summary(ica_dir, subject, run, eog_inds, eog_source_map, eog_detect_method)
        
        # Save individual component plots
        _save_individual_component_plots(ica_dir, subject, run, eog_inds, ica, raw_for_ica, eog_epochs_list)
        
        # Save combined components figure
        _save_combined_components_plot(ica_dir, subject, run, eog_inds, ica)
        
    except Exception as e:
        print(f"    WARNING: Could not produce ICA diagnostic plots: {e}")

def _save_exclusion_summary(ica_dir, subject, run, eog_inds, eog_source_map, eog_detect_method):
    """Save text summary of excluded components"""
    summary_file = ica_dir / f"sub-{subject}_run-{int(run):02d}_ica_excluded.txt"
    with open(summary_file, 'w', encoding='utf-8') as fh:
        fh.write(f"Excluded ICA components (VEOG ∪ HEOG): {eog_inds}\n")
        for comp in sorted(eog_inds):
            sources = eog_source_map.get(comp, [])
            method = eog_detect_method.get(comp, 'unknown')
            fh.write(f"comp {comp}: method={method}; detected_by={','.join(sources)}\n")
    print(f"    Saved ICA exclusion summary: {summary_file}")

def _save_individual_component_plots(ica_dir, subject, run, eog_inds, ica, raw_for_ica, eog_epochs_list):
    """Save individual topomap and source plots for each excluded component"""
    # Ensure montage for plotting
    try:
        if not raw_for_ica.get_montage():
            mont = mne.channels.make_standard_montage('standard_1020')
            raw_for_ica.set_montage(mont, on_missing='ignore')
        ica.info = raw_for_ica.info
    except Exception:
        pass
    
    for comp in eog_inds:
        # Save topomap
        try:
            fig = ica.plot_components(picks=[comp], show=False)
            png_topo = ica_dir / f"sub-{subject}_run-{int(run):02d}_ica_comp-{comp}_topomap.png"
            
            if isinstance(fig, (list, tuple)):
                for i, f in enumerate(fig):
                    try:
                        f.savefig(str(png_topo.with_name(png_topo.stem + f"_{i}.png")), dpi=150)
                        plt.close(f)
                    except Exception:
                        pass
            else:
                try:
                    fig.savefig(str(png_topo), dpi=150)
                    plt.close(fig)
                except Exception:
                    pass
        except Exception as e:
            print(f"    WARNING: Failed to save topomap for comp {comp}: {e}")
        
        # Save sources plot
        try:
            ep_for_plot = eog_epochs_list[0] if eog_epochs_list else None
            if ep_for_plot is not None:
                _save_sources_plot(ica_dir, subject, run, comp, ica, ep_for_plot)
        except Exception as e:
            print(f"    WARNING: Failed to save sources plot for comp {comp}: {e}")

def _save_sources_plot(ica_dir, subject, run, comp, ica, ep_for_plot):
    """Save sources plot for a specific component"""
    try:
        sources_ep = ica.get_sources(ep_for_plot)
        arr = sources_ep.get_data()
        if arr.size == 0:
            raise RuntimeError('No source data')
        
        n_epochs_src = arr.shape[0]
        n_show = min(20, n_epochs_src)
        if n_show <= 0:
            raise RuntimeError('No epochs available for plotting')
        
        if n_epochs_src <= n_show:
            sel_idx = list(range(n_epochs_src))
        else:
            sel_idx = np.linspace(0, n_epochs_src - 1, n_show, dtype=int).tolist()

        try:
            ep_subset = ep_for_plot[sel_idx]
        except Exception:
            ep_subset = mne.concatenate_epochs([ep_for_plot[i:i+1] for i in sel_idx])

        figs = ica.plot_sources(ep_subset, picks=[comp], show=False)
        if isinstance(figs, (list, tuple)):
            for i, f in enumerate(figs):
                try:
                    fname = ica_dir / f"sub-{subject}_run-{int(run):02d}_ica_comp-{comp}_sources_{i}.png"
                    f.savefig(str(fname), dpi=150)
                    plt.close(f)
                except Exception:
                    pass
        else:
            try:
                fname = ica_dir / f"sub-{subject}_run-{int(run):02d}_ica_comp-{comp}_sources.png"
                figs.savefig(str(fname), dpi=150)
                plt.close(figs)
            except Exception:
                pass
    except Exception as e:
        print(f"    WARNING: Failed to compute/save MNE sources plot for comp {comp}: {e}")

def _save_combined_components_plot(ica_dir, subject, run, eog_inds, ica):
    """Save combined components figure if there are multiple excluded components"""
    if eog_inds:
        try:
            fig_all = ica.plot_components(picks=eog_inds, show=False)
            png_all = ica_dir / f"sub-{subject}_run-{int(run):02d}_ica_excluded_components_all.png"
            if isinstance(fig_all, (list, tuple)):
                for i, f in enumerate(fig_all):
                    try:
                        f.savefig(str(png_all.with_name(png_all.stem + f"_{i}.png")), dpi=150)
                        plt.close(f)
                    except Exception:
                        pass
            else:
                try:
                    fig_all.savefig(str(png_all), dpi=150)
                    plt.close(fig_all)
                except Exception:
                    pass
        except Exception as e:
            print(f"    WARNING: Failed to save combined components figure: {e}")

def perform_ica_correction(raw_vep, raw_for_ica, picks_eeg, subject, run, out_path_vep, eog_candidates):
    """Perform complete ICA-based EOG correction"""
    try:
        # Fit ICA
        ica = ICA(n_components=0.99, method='fastica', random_state=97, max_iter=1000)
        ica.fit(raw_for_ica, picks=picks_eeg)
        
        # Detect EOG components (EOG candidates passed from main)
        eog_inds, eog_source_map, eog_detect_method, eog_epochs_list = detect_eog_components(
            ica, raw_for_ica, eog_candidates)
        
        # Apply exclusions
        ica.exclude = eog_inds
        print(f"    ICA removed components (detected by EOG channels): {eog_inds}")
        
        # Apply ICA to VEP data
        raw_vep_ica = ica.apply(raw_vep.copy())
        
        # Save diagnostics
        save_ica_diagnostics(ica, eog_inds, eog_source_map, eog_detect_method, 
                           eog_epochs_list, out_path_vep, subject, run, raw_for_ica)
        
        # Save ICA object
        try:
            ica_fname = out_path_vep.fpath.with_name(out_path_vep.fpath.stem + '-ica.fif')
            ica.save(str(ica_fname), overwrite=True)
            print(f"    Saved ICA: {ica_fname}")
        except Exception as e:
            print(f"    WARNING: Failed to save ICA: {e}")
        
        return raw_vep_ica
        
    except Exception as e:
        print(f"    WARNING: ICA failed: {e}")
        return raw_vep.copy()

def main(subjects, runs, bids_root, right_ear_label, new_sfreq=1000, exclude_channels=None):
    bids_root = Path(bids_root)
    deriv_root = bids_root / "derivatives"
    vep_root = deriv_root / "preproc_vep"
    vep_root.mkdir(parents=True, exist_ok=True)

    REF_DESC = "Linked earlobes (average of left and right earlobes)"
    
    for subject in subjects:
        for run in runs:
            bids_path = BIDSPath(subject=subject, task="visstim", run=f"{int(run):02d}",
                                datatype="eeg", root=bids_root)

            vhdr_path = bids_path.copy().update(suffix="eeg", extension=".vhdr").fpath
            if not vhdr_path.exists():
                print(f"WARNING: File not found: {vhdr_path}. Skipping this run.")
                continue

            try:
                print(f"\n▶ Loading {vhdr_path.name}")
                raw = read_raw_bids(bids_path=bids_path, verbose=False)
                raw.load_data()
            except Exception as e:
                print(f"WARNING: Error loading {vhdr_path.name}: {e}. Skipping this run.")
                continue

            # Mark excluded channels as bad
            if exclude_channels:
                resolved = []
                for ch in exclude_channels:
                    if ch in raw.ch_names:
                        resolved.append(ch)
                    else:
                        try:
                            idx = int(ch)
                            if 0 <= idx < len(raw.ch_names):
                                resolved.append(raw.ch_names[idx])
                            elif 1 <= idx <= len(raw.ch_names):
                                resolved.append(raw.ch_names[idx - 1])
                            else:
                                print(f"WARNING: exclude channel index {ch} out of range for {vhdr_path.name}")
                        except Exception:
                            print(f"WARNING: exclude channel {ch} not found in raw channels and not an index")
                if resolved:
                    existing_bads = list(raw.info.get('bads', []))
                    for r in resolved:
                        if r not in existing_bads:
                            existing_bads.append(r)
                    raw.info['bads'] = existing_bads
                    print(f"  → Marked channels as bad (excluded from ICA/topoplot): {existing_bads}")

            # Step 1: Re-reference to linked earlobes
            if right_ear_label in raw.ch_names:
                right_ear_signal = raw.get_data(picks=[right_ear_label])
                raw._data -= 0.5 * right_ear_signal
                print(f"  → Re-referenced to {REF_DESC}")
            else:
                print(f"WARNING: Right earlobe channel {right_ear_label} not found. Skipping re-reference.")

            # Step 2: VEP preprocessing
            raw_vep = raw.copy()
            raw_vep.filter(0.01, 30, picks="eeg", verbose=False)
            raw_vep.resample(new_sfreq, npad="auto")

            # Prepare output path
            out_path_vep = bids_path.copy().update(root=vep_root, suffix="eeg", extension=".fif")
            out_path_vep.fpath.parent.mkdir(parents=True, exist_ok=True)

            # Step 3: ICA-based EOG correction
            eog_chs = [raw_vep.ch_names[i] for i in mne.pick_types(raw_vep.info, eeg=False, eog=True)]
            
            if eog_chs:
                print(f"  → Found EOG channels: {eog_chs}")
                print(f"  → Performing ICA for blink correction...")
                
                # Prepare data for ICA fitting
                raw_for_ica = raw_vep.copy()
                raw_for_ica.filter(1.0, None, picks='eeg', verbose='ERROR')
                raw_for_ica.resample(200, npad='auto')
                
                # Select EEG channels for ICA
                picks_eeg = mne.pick_types(raw_for_ica.info, eeg=True, eog=False, exclude='bads')
                
                # Exclude right ear and photosensor channels
                if right_ear_label in raw_for_ica.ch_names:
                    idx = raw_for_ica.ch_names.index(right_ear_label)
                    if idx in picks_eeg:
                        picks_eeg = [p for p in picks_eeg if p != idx]
                
                photosensor_names = [n for n in raw_for_ica.ch_names if 'photosensor' in n.lower()]
                for pn in photosensor_names:
                    i = raw_for_ica.ch_names.index(pn)
                    if i in picks_eeg:
                        picks_eeg = [p for p in picks_eeg if p != i]
                
                print(f"    ICA using {len(picks_eeg)} EEG channels")
                
                # Perform ICA correction (EOG channels passed as argument)
                if len(picks_eeg) >= 3:
                    raw_vep_ica = perform_ica_correction(raw_vep, raw_for_ica, picks_eeg, subject, run, out_path_vep, eog_chs)
                else:
                    print(f"    Too few EEG channels ({len(picks_eeg)}) for ICA")
                    raw_vep_ica = raw_vep.copy()
            else:
                print("  → No EOG channels found in BIDS metadata. Skipping ICA.")
                raw_vep_ica = raw_vep.copy()

            # Save cleaned VEP data
            try:
                raw_vep_ica.save(out_path_vep.fpath, overwrite=True)
            except Exception as e:
                print(f"WARNING: Failed to save VEP output {out_path_vep.fpath}: {e}")

            # Write sidecar metadata
            try:
                (out_path_vep.fpath.with_suffix('.json')).write_text(
                    f"""{{
        "Description": "Preprocessed for standard visual ERP (VEP) analysis with ICA-based EOG correction.",
        "Reference": "{REF_DESC}",
        "HighpassFilter": 0.01,
        "LowpassFilter": 30,
        "ResampledTo": {new_sfreq},
        "ArtifactCorrection": "ICA on VEOG/HEOG (blink removal)",
        "Software": "MNE-Python"
        }}""",
                    encoding='utf-8'
                )
            except Exception as e:
                print(f"WARNING: Failed to write VEP sidecar {out_path_vep.fpath.with_suffix('.json')}: {e}")

    print("\n✅ VEP-only preprocessing complete and saved under derivatives/preproc_vep/.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VEP-only preprocessing (bandpass 0.01-30 Hz, ICA)')
    parser.add_argument('--subjects', nargs='+', default=["01","02","03","04","05","06","07","08"], help='subject list')
    parser.add_argument('--runs', nargs='+', type=int, default=[1,2,3,4,5,6], help='run numbers')
    parser.add_argument('--bids_root', type=str, default='.', help='BIDS root path')
    parser.add_argument('--right_ear', type=str, default='X2', help='Right earlobe channel name used for re-reference')
    parser.add_argument('--sfreq', type=int, default=1000, help='Target sampling frequency')
    parser.add_argument('--exclude-channels', nargs='*', default=['I1', 'I2'], help='channel names or indices to mark as bad and exclude from ICA/topoplot')
    args = parser.parse_args()
    main(args.subjects, args.runs, args.bids_root, args.right_ear, new_sfreq=args.sfreq, exclude_channels=args.exclude_channels)
