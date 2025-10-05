#!/usr/bin/env python3
"""
Rapid CNN data expansion for pixel differences
Prioritizes targets most likely to have good data
"""

import sys
sys.path.append('../src')

from pixel_diff import download_target_pixel_file, compute_pixel_differences
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time

def process_single_target(row):
    """Process a single target for pixel differences"""
    try:
        kepid = int(row['kepid'])
        filename = f'../data/processed/pixel_diffs/pixdiff_{kepid}.npy'
        
        # Skip if already exists
        if os.path.exists(filename):
            return f"Skip {kepid} (exists)"
            
        period = float(row['koi_period'])
        t0 = float(row['koi_time0bk'])
        duration = float(row['koi_duration'])
        
        # Download and process
        tpf = download_target_pixel_file(str(kepid), mission='Kepler')
        if tpf is None:
            return f"Skip {kepid} (no TPF)"
            
        pixel_diffs = compute_pixel_differences(tpf, period, t0, duration)
        if pixel_diffs.size == 0:
            return f"Skip {kepid} (no diffs)"
            
        # Save
        np.save(filename, pixel_diffs)
        return f"✅ {kepid} -> {pixel_diffs.shape}"
        
    except Exception as e:
        return f"❌ {kepid}: {str(e)[:50]}"

def main():
    # Load and filter targets
    df = pd.read_csv('../data/raw/lighkurve_KOI_dataset_enriched.csv')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Prioritize confirmed planets and shorter periods (more likely to have good data)
    df_priority = df[
        (df['koi_disposition'] == 'CONFIRMED') & 
        (df['koi_period'] < 50) &
        (df['koi_period'] > 0.5) &
        (df['koi_duration'] > 0.5) &
        (df['koi_duration'] < 12)
    ].head(200)  # Process top 200 priority targets
    
    print(f"Processing {len(df_priority)} priority targets for pixel differences...")
    
    # Ensure directory exists
    os.makedirs('../data/processed/pixel_diffs', exist_ok=True)
    
    # Process with limited concurrency (too much can overwhelm MAST)
    success_count = 0
    with ThreadPoolExecutor(max_workers=2) as executor:  # Lower for TPF (larger files)
        results = list(executor.map(process_single_target, [row for _, row in df_priority.iterrows()]))
        
    for result in results:
        print(result)
        if result.startswith('✅'):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(df_priority)} successful downloads")

if __name__ == "__main__":
    main()