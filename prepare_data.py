#!/usr/bin/env python3
"""
Script to prepare StyleTTS2 data files from mounted voice-checkpoint data.
Converts metadata.tsv to train_list.txt and val_list.txt in the required format.
"""

import os
import random
from pathlib import Path

# Configuration
MOUNTED_DATA_DIR = Path("/home/Ilya/voice-checkpoint")
METADATA_FILE = MOUNTED_DATA_DIR / "metadata.tsv"
WAVS_DIR = MOUNTED_DATA_DIR / "wavs"
OUTPUT_DIR = Path("/home/Ilya/StyleTTS2/Data")
TRAIN_LIST = OUTPUT_DIR / "train_list.txt"
VAL_LIST = OUTPUT_DIR / "val_list.txt"
TRAIN_SPLIT = 0.8  # 80% train, 20% validation
SPEAKER_ID = "0"  # Single speaker dataset

def load_metadata(metadata_path):
    """Load metadata.tsv file and return list of (filename, transcription) tuples."""
    entries = []
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Split by tab
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"Warning: Skipping line {line_num} - invalid format: {line}")
                continue
            
            filename = parts[0].strip()
            transcription = '\t'.join(parts[1:]).strip()  # Handle multiple tabs
            
            if not filename or not transcription:
                print(f"Warning: Skipping line {line_num} - empty filename or transcription")
                continue
            
            entries.append((filename, transcription))
    
    return entries

def verify_wav_files(entries, wavs_dir):
    """Verify that all wav files exist and return valid entries."""
    valid_entries = []
    missing_files = []
    
    for filename, transcription in entries:
        wav_path = wavs_dir / filename
        if wav_path.exists():
            valid_entries.append((filename, transcription))
        else:
            missing_files.append(filename)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} WAV files not found:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return valid_entries

def format_entry(filename, transcription, speaker_id):
    """Format entry in StyleTTS2 format: filename.wav|transcription|speaker_id"""
    return f"{filename}|{transcription}|{speaker_id}"

def split_train_val(entries, train_split):
    """Split entries into train and validation sets."""
    random.seed(42)  # For reproducibility
    shuffled = entries.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_split)
    train_entries = shuffled[:split_idx]
    val_entries = shuffled[split_idx:]
    
    return train_entries, val_entries

def write_data_list(filepath, entries, speaker_id):
    """Write entries to data list file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for filename, transcription in entries:
            entry = format_entry(filename, transcription, speaker_id)
            f.write(entry + '\n')

def main():
    print("Loading metadata...")
    entries = load_metadata(METADATA_FILE)
    print(f"Loaded {len(entries)} entries from metadata.tsv")
    
    print(f"\nVerifying WAV files in {WAVS_DIR}...")
    valid_entries = verify_wav_files(entries, WAVS_DIR)
    print(f"Found {len(valid_entries)} valid entries (with existing WAV files)")
    
    if len(valid_entries) == 0:
        print("Error: No valid entries found. Please check your data paths.")
        return
    
    print(f"\nSplitting into train ({TRAIN_SPLIT*100:.0f}%) and validation ({(1-TRAIN_SPLIT)*100:.0f}%)...")
    train_entries, val_entries = split_train_val(valid_entries, TRAIN_SPLIT)
    print(f"Train: {len(train_entries)} entries")
    print(f"Validation: {len(val_entries)} entries")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting train_list.txt...")
    write_data_list(TRAIN_LIST, train_entries, SPEAKER_ID)
    
    print(f"Writing val_list.txt...")
    write_data_list(VAL_LIST, val_entries, SPEAKER_ID)
    
    print(f"\n✓ Success! Data files created:")
    print(f"  - {TRAIN_LIST}")
    print(f"  - {VAL_LIST}")
    print(f"\nNext step: Update root_path in Configs/config_ft.yml to:")
    print(f"  root_path: \"{WAVS_DIR}\"")

if __name__ == "__main__":
    main()

