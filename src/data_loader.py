import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_data(data_dir='data/raw/'):
    """
    Load and merge tabular data from multiple sources.
    
    Returns:
        dict: Contains merged dataframes and metadata
    """
    try:
        # Load data sources
        logger.info("Loading tabular data...")
        
        # Priority 1: Check for real KOI data with actual features
        koi_path = f"{data_dir}/lighkurve_KOI_dataset.csv"
        if os.path.exists(koi_path):
            # Load real KOI data with actual stellar parameters
            koi_df = pd.read_csv(koi_path)
            logger.info(f"Loaded {len(koi_df)} real KOI records with {koi_df.shape[1]} features")
        else:
            # Priority 2: Check for minimal KOI labels file
            koi_labels_path = f"{data_dir}/q1_q17_dr25_sup_koi_2024.07.03_19.12.12.csv"
            global_path = f"{data_dir}/all_global.csv"
            local_path = f"{data_dir}/all_local.csv"
            
            # Strategy: Use synthetic global/local features if they exist
            if os.path.exists(global_path) and os.path.exists(local_path):
                # Load synthetic data format (global + local features)
                global_df = pd.read_csv(global_path)
                local_df = pd.read_csv(local_path) 
                
                # Merge features
                merged_df = global_df.merge(local_df, on='kepid', how='inner')
                
                # Load labels if available
                if os.path.exists(koi_labels_path):
                    koi_labels = pd.read_csv(koi_labels_path)
                    # Merge labels
                    merged_df = merged_df.merge(koi_labels[['kepid', 'koi_disposition']], on='kepid', how='left')
                    
                # Convert to KOI format for processing
                koi_df = merged_df
                logger.info(f"Loaded {len(koi_df)} synthetic records")
                
            elif os.path.exists(koi_labels_path):
                # Fall back to minimal labels only
                koi_df = pd.read_csv(koi_labels_path)
                logger.info(f"Loaded {len(koi_df)} minimal KOI records")
            else:
                logger.warning("No data files found")
                koi_df = pd.DataFrame()
        
        # For now, create empty dataframes for other sources
        toi_df = pd.DataFrame()
        tres_df = pd.DataFrame() 
        lk_df = pd.DataFrame()
        
        # Process and standardize column names
        processed_data = process_tabular_features(koi_df, toi_df, tres_df, lk_df)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def process_tabular_features(koi_df, toi_df, tres_df, lk_df):
    """
    Process and standardize tabular features across datasets.
    
    Returns:
        dict: Processed dataframes with standardized features
    """
    features = []
    
    # Define common features to extract
    common_features = [
        'period', 'period_err1', 'period_err2',
        'epoch', 'epoch_err1', 'epoch_err2', 
        'depth', 'depth_err1', 'depth_err2',
        'duration', 'duration_err1', 'duration_err2',
        'impact', 'impact_err1', 'impact_err2',
        'star_temp', 'star_temp_err1', 'star_temp_err2',
        'star_radius', 'star_radius_err1', 'star_radius_err2',
        'star_mass', 'star_mass_err1', 'star_mass_err2',
        'star_logg', 'star_logg_err1', 'star_logg_err2',
        'star_metallicity', 'star_metallicity_err1', 'star_metallicity_err2'
    ]
    
    # Process KOI data
    if not koi_df.empty:
        koi_processed = extract_koi_features(koi_df)
        features.append(('koi', koi_processed))
    
    # Process TOI data  
    if not toi_df.empty:
        toi_processed = extract_toi_features(toi_df)
        features.append(('toi', toi_processed))
    
    # Process TrES data
    if not tres_df.empty:
        tres_processed = extract_tres_features(tres_df)
        features.append(('tres', tres_processed))
    
    # Process Lightkurve data
    if not lk_df.empty:
        lk_processed = extract_lightkurve_features(lk_df)
        features.append(('lightkurve', lk_processed))
    
    return {'features': features, 'common_columns': common_features}

def extract_koi_features(df):
    """Extract and standardize features from KOI dataset."""
    logger.info("Processing KOI features...")
    
    # Create target variable from disposition
    if 'koi_disposition' in df.columns:
        df['is_planet'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    else:
        # Handle synthetic data without disposition
        df['is_planet'] = 0  # Default to non-planet
    
    # For synthetic data, create dummy features if real KOI columns don't exist
    feature_mapping = {
        'koi_period': 'period',
        'koi_period_err1': 'period_err1', 
        'koi_period_err2': 'period_err2',
        'koi_time0bk': 'epoch',
        'koi_time0bk_err1': 'epoch_err1',
        'koi_time0bk_err2': 'epoch_err2',
        'koi_depth': 'depth',
        'koi_depth_err1': 'depth_err1',
        'koi_depth_err2': 'depth_err2',
        'koi_duration': 'duration', 
        'koi_duration_err1': 'duration_err1',
        'koi_duration_err2': 'duration_err2',
        'koi_impact': 'impact',
        'koi_impact_err1': 'impact_err1',
        'koi_impact_err2': 'impact_err2',
        'koi_slogg': 'star_logg',
        'koi_slogg_err1': 'star_logg_err1',
        'koi_slogg_err2': 'star_logg_err2',
        'koi_smet': 'star_metallicity',
        'koi_smet_err1': 'star_metallicity_err1',
        'koi_smet_err2': 'star_metallicity_err2',
    }
    
    # Check if we have real KOI columns or synthetic data
    has_real_koi_columns = any(col in df.columns for col in feature_mapping.keys())
    
    if has_real_koi_columns:
        # Extract real KOI features using the mapping
        extracted = pd.DataFrame()
        for old_name, new_name in feature_mapping.items():
            if old_name in df.columns:
                extracted[new_name] = df[old_name]
                
        # Also include any other numeric columns not in the mapping
        other_numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in other_numeric_cols:
            if col not in ['kepid'] and col not in feature_mapping.keys():
                extracted[col] = df[col]
                
    else:
        # Handle synthetic data - use existing columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['kepid']]
        
        # Copy all feature columns at once, preserving the DataFrame structure
        extracted = df[feature_cols].copy()
    
    # Add identifiers
    extracted['target_id'] = df.get('kepid', df.index)
    extracted['koi_id'] = df.get('kepoi_name', '')
    extracted['is_planet'] = df['is_planet']
    extracted['source'] = 'koi'
    
    logger.info(f"Extracted {len(extracted)} KOI features with {extracted.shape[1]-4} feature columns")
    return extracted

def extract_toi_features(df):
    """Extract and standardize features from TOI dataset."""
    logger.info("Processing TOI features...")
    
    # Create target variable
    df['is_planet'] = df.get('TFOPWG Disposition', '').str.contains('CP|PC', na=False).astype(int)
    
    feature_mapping = {
        'Period (days)': 'period',
        'Period (days) err': 'period_err1',
        'Epoch (BJD)': 'epoch', 
        'Epoch (BJD) err': 'epoch_err1',
        'Depth (ppm)': 'depth',
        'Depth (ppm) err': 'depth_err1',
        'Duration (hours)': 'duration',
        'Duration (hours) err': 'duration_err1',
        'Stellar Eff Temp (K)': 'star_temp',
        'Stellar Eff Temp (K) err': 'star_temp_err1',
        'Stellar Radius (R_Sun)': 'star_radius', 
        'Stellar Radius (R_Sun) err': 'star_radius_err1',
        'Stellar Mass (M_Sun)': 'star_mass',
        'Stellar Mass (M_Sun) err': 'star_mass_err1',
        'Stellar log(g) (cgs)': 'star_logg',
        'Stellar log(g) (cgs) err': 'star_logg_err1',
    }
    
    extracted = pd.DataFrame()
    for old_name, new_name in feature_mapping.items():
        if old_name in df.columns:
            extracted[new_name] = df[old_name]
    
    extracted['target_id'] = df.get('TIC ID', df.index)
    extracted['toi_id'] = df.get('TOI', '')
    extracted['is_planet'] = df['is_planet']
    extracted['source'] = 'toi'
    
    logger.info(f"Extracted {len(extracted)} TOI features")
    return extracted

def extract_tres_features(df):
    """Extract and standardize features from TrES dataset.""" 
    logger.info("Processing TrES features...")
    
    # TrES are confirmed planets
    df['is_planet'] = 1
    
    # Map available features
    feature_mapping = {
        'pl_orbper': 'period',
        'pl_orbpererr1': 'period_err1',
        'pl_orbpererr2': 'period_err2', 
        'pl_tranmid': 'epoch',
        'pl_tranmiderr1': 'epoch_err1',
        'pl_tranmiderr2': 'epoch_err2',
        'pl_trandep': 'depth',
        'pl_trandepERR1': 'depth_err1',
        'pl_trandepERR2': 'depth_err2',
        'pl_trandur': 'duration',
        'pl_trandurerr1': 'duration_err1', 
        'pl_trandurerr2': 'duration_err2',
        'st_teff': 'star_temp',
        'st_tefferr1': 'star_temp_err1',
        'st_tefferr2': 'star_temp_err2',
        'st_rad': 'star_radius',
        'st_raderr1': 'star_radius_err1',
        'st_raderr2': 'star_radius_err2',
        'st_mass': 'star_mass', 
        'st_masserr1': 'star_mass_err1',
        'st_masserr2': 'star_mass_err2',
        'st_logg': 'star_logg',
        'st_loggerr1': 'star_logg_err1',
        'st_loggerr2': 'star_logg_err2',
        'st_met': 'star_metallicity',
        'st_meterr1': 'star_metallicity_err1',
        'st_meterr2': 'star_metallicity_err2',
    }
    
    extracted = pd.DataFrame()
    for old_name, new_name in feature_mapping.items():
        if old_name in df.columns:
            extracted[new_name] = df[old_name]
    
    extracted['target_id'] = df.get('hostname', df.index) 
    extracted['tres_id'] = df.get('pl_name', '')
    extracted['is_planet'] = df['is_planet']
    extracted['source'] = 'tres'
    
    logger.info(f"Extracted {len(extracted)} TrES features") 
    return extracted

def extract_lightkurve_features(df):
    """Extract features from Lightkurve processed dataset."""
    logger.info("Processing Lightkurve features...")
    
    # Assume processed features already exist
    extracted = df.copy()
    
    if 'is_planet' not in extracted.columns:
        extracted['is_planet'] = 0  # Default to non-planet
    
    extracted['source'] = 'lightkurve'
    
    logger.info(f"Extracted {len(extracted)} Lightkurve features")
    return extracted

def create_train_val_test_splits(data_dict, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train/validation/test splits avoiding star leakage using GroupKFold.
    
    Args:
        data_dict: Output from load_and_prepare_data()
        test_size: Fraction for test set
        val_size: Fraction of remaining for validation
        random_state: Random seed
    
    Returns:
        dict: Train/val/test splits with indices
    """
    logger.info("Creating train/validation/test splits...")
    
    # Combine all features
    all_features = []
    all_targets = []
    all_groups = []
    source_info = []
    
    for source, df in data_dict['features']:
        if len(df) > 0:
            # Get numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['is_planet', 'target_id']]
            
            features = df[feature_cols].fillna(0).values
            targets = df['is_planet'].values
            groups = df['target_id'].values  # Group by stellar system
            
            all_features.append(features)
            all_targets.extend(targets)
            all_groups.extend(groups)
            source_info.extend([source] * len(df))
    
    if not all_features:
        raise ValueError("No features found in data")
    
    # Combine features
    X = np.vstack(all_features)
    y = np.array(all_targets)
    groups = np.array(all_groups)
    
    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}, Positive class: {y.sum()}")
    
    # Create splits avoiding group leakage
    unique_groups = np.unique(groups)
    
    # First split: separate test set
    group_train, group_test = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state, 
        stratify=None  # Can't stratify groups directly
    )
    
    # Second split: separate validation from train
    group_train, group_val = train_test_split(
        group_train, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Get indices for each split
    train_idx = np.where(np.isin(groups, group_train))[0]
    val_idx = np.where(np.isin(groups, group_val))[0] 
    test_idx = np.where(np.isin(groups, group_test))[0]
    
    # Create standardized features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    
    splits = {
        'train': {
            'X': X_train,
            'y': y[train_idx],
            'groups': groups[train_idx],
            'indices': train_idx
        },
        'val': {
            'X': X_val, 
            'y': y[val_idx],
            'groups': groups[val_idx],
            'indices': val_idx
        },
        'test': {
            'X': X_test,
            'y': y[test_idx], 
            'groups': groups[test_idx],
            'indices': test_idx
        },
        'scaler': scaler,
        'feature_names': feature_cols
    }
    
    logger.info(f"Train: {len(train_idx)} ({y[train_idx].sum()} positive)")
    logger.info(f"Val: {len(val_idx)} ({y[val_idx].sum()} positive)")
    logger.info(f"Test: {len(test_idx)} ({y[test_idx].sum()} positive)")
    
    return splits

if __name__ == "__main__":
    # Test the data loading pipeline
    logging.basicConfig(level=logging.INFO)
    
    try:
        data = load_and_prepare_data()
        splits = create_train_val_test_splits(data)
        print("Data loading pipeline test successful!")
        
    except Exception as e:
        print(f"Data loading pipeline test failed: {e}")