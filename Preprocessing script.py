"""
EEG Preprocessing Pipeline — Subject 1, Session 1, All Blocks
==============================================================
Downloads raw EDF files for all 19 blocks from HuggingFace,
runs the full preprocessing pipeline (filter → ICA → epoch → clean),
and saves per-block numpy arrays plus one combined output.

Requirements:
    pip install mne autoreject huggingface_hub pandas numpy

Usage:
    python preprocess_all_blocks.py
"""

import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from autoreject import get_rejection_threshold
from huggingface_hub import hf_hub_download

mne.set_log_level('error')

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ID        = "Alljoined/Alljoined-1.6M"
SUBJECT        = 1
SESSION        = 1
N_BLOCKS       = 19
OUTPUT_DIR     = Path("preprocessed_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# The exact filename pattern on HuggingFace — adjust if the repo uses a different naming convention
# Based on your block 1 file: "Subject 1, Session 1, Block 1 Recording_FLEX2_213075_2025.01.25T15.18.28.08.00.md.edf"
# We assume the dataset stores files under a subfolder like raw/Sub_1/ or similar.
# UPDATE THIS PATTERN to match the actual HuggingFace repo file structure.
HF_FILE_PATTERN = "raw/Sub_{sub}/Subject {sub}, Session {ses}, Block {blk} Recording*.edf"

# Channel setup
TARGET_CHANNELS = [
    'Cz', 'FCz', 'Afz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1',
    'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2',
    'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2',
    'F6', 'Fp2'
]

# Filter parameters
BANDPASS_LOW  = 0.1
BANDPASS_HIGH = 40.0
ICA_HP        = 1.0    # High-pass used only during ICA fitting

# Epoch parameters
TMIN     = -0.200
TMAX     =  1.000
BASELINE = (None, 0)

# ICA parameters
ICA_N_COMPONENTS = 0.99
RANDOM_STATE     = 42
ICA_Z_THRESH     = 1.96
EOG_CHANNELS     = ['Fp1', 'F6']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def download_block(subject: int, session: int, block: int) -> Path:
    """
    Downloads the raw EDF file for a given block from HuggingFace.
    Returns the local path to the downloaded file.

    Path structure on HuggingFace:
        raw_eeg/sub-01/session_01/block_01/<filename>.edf
    """
    # Zero-pad the numbers to match HuggingFace folder naming
    sub_str     = f"sub-{subject:02d}"
    ses_str     = f"session_{session:02d}"
    blk_str     = f"block_{block:02d}"
    subfolder   = f"raw_eeg/{sub_str}/{ses_str}/{blk_str}"

    # The EDF filename inside each block folder
    # Based on block 1: "Subject 1, Session 1, Block 1 Recording_FLEX2_213075_2025.01.25T15.18.28.08.00.md.edf"
    # The timestamp and device ID suffix varies per block, so we use huggingface_hub
    # to list the folder contents and find the EDF automatically.
    from huggingface_hub import list_repo_files

    # Find the EDF file in this block's folder
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    edf_file = None
    for f in all_files:
        if f.startswith(subfolder) and f.endswith('.edf'):
            edf_file = f
            break

    if edf_file is None:
        raise FileNotFoundError(f"No EDF file found in {subfolder} on HuggingFace.")

    # Extract just the filename (last part after the final slash)
    filename      = edf_file.split('/')[-1]
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        subfolder=subfolder,
        local_dir=OUTPUT_DIR / "raw_cache",
    )
    return Path(local_path)


def load_and_prepare_raw(edf_path: Path) -> mne.io.Raw:
    """
    Loads raw EDF, picks the 32 target channels, fixes the AFz name,
    sets the montage, and applies the main bandpass filter.
    Returns the filtered Raw object.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)

    # Keep only the 32 core EEG channels
    raw.pick_channels(TARGET_CHANNELS)

    # Fix Emotiv's non-standard central electrode name
    raw.rename_channels({'Afz': 'AFz'})

    # Set the standard 10-20 montage
    raw.set_montage('standard_1020')

    # Apply main bandpass filter
    raw_filt = raw.copy().filter(BANDPASS_LOW, BANDPASS_HIGH)

    return raw_filt


def fit_ica(raw_filt: mne.io.Raw) -> mne.preprocessing.ICA:
    """
    Fits ICA on a 1Hz-high-passed copy of the filtered raw data.
    Automatically identifies and marks ocular artifact components.
    Returns the fitted ICA object.
    """
    # ICA works better with stronger high-pass to remove slow drift
    raw_ica = raw_filt.copy().filter(ICA_HP, BANDPASS_HIGH)

    # Epoch into fixed 1-second windows for autoreject threshold estimation
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
    epochs_ica = mne.Epochs(
        raw_ica, events_ica,
        tmin=0.0, tmax=tstep,
        baseline=None, preload=True
    )

    # Automatically determine rejection threshold
    reject = get_rejection_threshold(epochs_ica)

    # Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=RANDOM_STATE,
    )
    ica.fit(epochs_ica, reject=reject, tstep=tstep)

    # Auto-detect eye blink and horizontal movement components
    eog_indices, _ = ica.find_bads_eog(
        raw_ica,
        ch_name=EOG_CHANNELS,
        threshold=ICA_Z_THRESH,
    )
    ica.exclude = eog_indices

    print(f"  ICA fitted. Components excluded (ocular): {eog_indices}")
    return ica


def build_event_mapping(metadata: pd.DataFrame, events: np.ndarray) -> dict:
    """
    Builds a local event_id dictionary filtered to only the image IDs
    that actually appear as hardware triggers in this specific block.
    """
    event_codes = metadata['fname'].str.replace('.jpg', '', regex=False).astype(int)
    event_names = (
        metadata['category_name'].fillna('unknown').astype(str)
        + '/' + event_codes.astype(str)
    )
    full_mapping = {str(k): int(v) for k, v in zip(event_names, event_codes)}

    # Filter to only events present in this block's EEG triggers
    present_events = set(events[:, 2])
    local_mapping = {
        name: code for name, code in full_mapping.items()
        if code in present_events
    }

    return local_mapping


def process_block(
    block: int,
    metadata: pd.DataFrame,
    subject: int = SUBJECT,
    session: int = SESSION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline for a single block:
        1. Download EDF
        2. Load and filter
        3. Fit ICA and detect ocular components
        4. Build event mapping
        5. Epoch
        6. Apply ICA to epochs
        7. Extract and save X, y arrays

    Returns (X, y) as numpy arrays.
    """
    print(f"\n{'='*60}")
    print(f"  Processing Block {block} / {N_BLOCKS}")
    print(f"{'='*60}")

    # --- Step 1: Download ---
    print("  [1/6] Downloading EDF from HuggingFace...")
    edf_path = download_block(subject, session, block)
    print(f"        Downloaded to: {edf_path}")

    # --- Step 2: Load and filter ---
    print("  [2/6] Loading raw data and applying bandpass filter...")
    raw_filt = load_and_prepare_raw(edf_path)

    # --- Step 3: Fit ICA ---
    print("  [3/6] Fitting ICA (this may take a minute)...")
    ica = fit_ica(raw_filt)

    # --- Step 4: Event mapping ---
    print("  [4/6] Building stimulus event mapping...")
    events, _ = mne.events_from_annotations(raw_filt)
    local_event_mapping = build_event_mapping(metadata, events)
    print(f"        Unique images in this block: {len(local_event_mapping)}")

    # --- Step 5: Epoch ---
    print("  [5/6] Epoching...")
    epochs = mne.Epochs(
        raw_filt,
        events,
        local_event_mapping,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
    )
    print(f"        Epochs retained: {len(epochs)}")

    # --- Step 6: Apply ICA ---
    print("  [6/6] Applying ICA to epochs...")
    epochs_clean = ica.apply(epochs.copy())

    # --- Step 7: Extract arrays ---
    X = epochs_clean.get_data(copy=False)   # shape: (trials, channels, timepoints)
    y = epochs_clean.events[:, 2]           # shape: (trials,)

    print(f"        X shape: {X.shape} | y shape: {y.shape}")

    # Save per-block outputs
    block_x_path = OUTPUT_DIR / f"Sub{subject}_Ses{session}_Block{block:02d}_X.npy"
    block_y_path = OUTPUT_DIR / f"Sub{subject}_Ses{session}_Block{block:02d}_y.npy"
    np.save(block_x_path, X)
    np.save(block_y_path, y)
    print(f"        Saved: {block_x_path.name} and {block_y_path.name}")

    return X, y


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\nEEG PREPROCESSING PIPELINE")
    print(f"Subject {SUBJECT} | Session {SESSION} | Blocks 1–{N_BLOCKS}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    # Load the metadata parquet once — used for event mapping in every block
    print("Loading experiment metadata...")
    metadata = pd.read_parquet("experiment_metadata_categories.parquet")
    print(f"Metadata loaded. Total images in master list: {len(metadata)}\n")

    all_X = []
    all_y = []

    for block in range(1, N_BLOCKS + 1):
        try:
            X, y = process_block(block, metadata)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"\n  [WARNING] Block {block} failed: {e}")
            print("  Skipping this block and continuing...\n")
            continue

    # Combine all blocks into a single output
    if all_X:
        print(f"\n{'='*60}")
        print("  Combining all blocks into single output...")
        combined_X = np.concatenate(all_X, axis=0)
        combined_y = np.concatenate(all_y, axis=0)

        combined_x_path = OUTPUT_DIR / f"Sub{SUBJECT}_Ses{SESSION}_ALL_X.npy"
        combined_y_path = OUTPUT_DIR / f"Sub{SUBJECT}_Ses{SESSION}_ALL_y.npy"
        np.save(combined_x_path, combined_X)
        np.save(combined_y_path, combined_y)

        print(f"  Combined X shape: {combined_X.shape}")
        print(f"  Combined y shape: {combined_y.shape}")
        print(f"  Saved: {combined_x_path.name}")
        print(f"  Saved: {combined_y_path.name}")
        print(f"\nDone. All outputs in: {OUTPUT_DIR.resolve()}")
    else:
        print("\n[ERROR] No blocks were successfully processed.")


if __name__ == "__main__":
    main()
