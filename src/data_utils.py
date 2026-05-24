"""Data loading and preprocessing utilities."""

import gc
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def get_memory_mb(df):
    """Return DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1024**2


def load_raw_data(data_dir, chunksize=500_000):
    """
    Load all TSV/TXT files from data_dir with column pruning and dtype optimization.
    
    Args:
        data_dir: Path to directory containing data files
        chunksize: Rows per chunk to balance memory vs I/O
    
    Returns:
        DataFrame with columns: square_id, time_interval, internet
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = sorted(data_dir.rglob("*.txt")) + sorted(data_dir.rglob("*.tsv"))
    
    if not files:
        raise FileNotFoundError(f"No .txt or .tsv files in {data_dir}")
    
    print(f"Found {len(files)} files")
    
    COL_NAMES = [
        'square_id', 'time_interval', 'country_code',
        'sms_in', 'sms_out', 'call_in', 'call_out', 'internet'
    ]
    KEEP_COLS = [0, 1, 7]  # square_id, time_interval, internet
    
    chunks = []
    
    for fpath in files:
        print(f"Reading {fpath.name}...")
        
        for chunk in pd.read_csv(
            fpath,
            sep="\t",
            header=None,
            names=COL_NAMES,
            usecols=KEEP_COLS,
            dtype={
                'square_id': 'int16',
                'time_interval': 'int64',
                'internet': 'float32'
            },
            chunksize=chunksize
        ):
            chunk.dropna(subset=["internet"], inplace=True)
            chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    return df


def optimize_dtypes(df):
    """Convert raw data to optimized dtypes."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time_interval'], unit='ms')
    df['internet'] = df['internet'].fillna(0).astype('float32')
    df['square_id'] = df['square_id'].astype('int16')
    df.drop(columns=['time_interval'], inplace=True)
    return df


def prepare_data(df):
    """Sort and prepare data for analysis."""
    df = df.sort_values(['square_id', 'datetime']).reset_index(drop=True)
    return df


def area_series(df, sq_id, start=None, end=None):
    """Extract time series for a single area."""
    s = df[df['square_id'] == sq_id].set_index('datetime')['internet'].sort_index()
    if start:
        s = s[s.index >= start]
    if end:
        s = s[s.index <= end]
    return s


def train_test_split(df, areas, test_start, test_end):
    """Split data for each area into train/test."""
    splits = {}
    
    for label, sq_id in areas.items():
        s = area_series(df, sq_id)
        train = s[s.index < test_start]
        test = s[(s.index >= test_start) & (s.index <= test_end)]
        
        splits[label] = {
            'train': train,
            'test': test,
            'square_id': sq_id
        }
    
    return splits


def print_memory_report(df_before, df_after):
    """Print memory optimization statistics."""
    mem_before = get_memory_mb(df_before)
    mem_after = get_memory_mb(df_after)
    reduction = 100 * (1 - mem_after / mem_before)
    
    print("\n" + "="*50)
    print("Memory Optimization Report")
    print("="*50)
    print(f"Before optimization: {mem_before:.2f} MB")
    print(f"After optimization:  {mem_after:.2f} MB")
    print(f"Memory saved:        {mem_before - mem_after:.2f} MB ({reduction:.1f}%)")
    print("="*50 + "\n")
    
    return mem_before, mem_after, reduction
