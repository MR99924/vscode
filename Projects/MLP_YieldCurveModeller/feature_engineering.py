"""
Feature engineering utilities for yield curve modeling.
This module contains functions for creating and transforming features.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

def forward_fill_to_current_date(df, freq='MS', max_fill_limit=None):
    """
    Forward-fill a DataFrame to the current date with enhanced controls.
    
    Parameters:
        df (DataFrame): DataFrame to forward-fill
        freq (str): Frequency for date_range ('MS' for monthly, 'D' for daily)
        max_fill_limit (int, optional): Maximum number of periods to forward fill. None means no limit.
        
    Returns:
        DataFrame: Forward-filled DataFrame
    """
    if df is None or df.empty:
        logger.warning("Cannot forward fill empty DataFrame")
        return df
    
    original_end = df.index.max()
    today = pd.Timestamp.today()
    
    if original_end < today:
        # Calculate number of periods to forward fill
        if freq == 'MS':
            periods_to_fill = (today.year - original_end.year) * 12 + (today.month - original_end.month)
        elif freq == 'D':
            periods_to_fill = (today - original_end).days
        else:
            # For other frequencies, let pandas handle it
            periods_to_fill = None
        
        # Apply max fill limit if specified
        if max_fill_limit is not None and periods_to_fill is not None:
            if periods_to_fill > max_fill_limit:
                logger.warning(f"Forward fill would exceed max limit ({periods_to_fill} > {max_fill_limit}). Limiting to {max_fill_limit} periods.")
                end_date = pd.date_range(start=original_end, periods=max_fill_limit+1, freq=freq)[-1]
            else:
                end_date = today
        else:
            end_date = today
            
        # Create a date range from the earliest date to end_date
        extended_dates = pd.date_range(start=df.index.min(), end=end_date, freq=freq)
        
        # Check if original index frequency matches requested frequency
        if df.index.inferred_freq != freq:
            logger.warning(f"Original data frequency ({df.index.inferred_freq}) doesn't match requested frequency ({freq})")
        
        # Reindex and forward fill
        df_extended = df.reindex(extended_dates).ffill()
        
        # Log the extension
        logger.info(f"Forward-filled data from {original_end.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Check for excessive filling
        fill_pct = (len(df_extended) - len(df)) / len(df) * 100
        if fill_pct > 20:  # If more than 20% of data is filled
            logger.warning(f"Large portion of data ({fill_pct:.1f}%) is forward-filled rather than observed")
        
        return df_extended
    
    return df

def combine_features(base_features, engineered_features, handle_missing='impute', max_missing_pct=0.9):
    """
    Combine base and engineered features with options for handling missing values.
    
    Parameters:
        base_features (DataFrame): Original features
        engineered_features (DataFrame): Engineered features
        handle_missing (str): Strategy for missing values ('drop', 'impute', or 'forward_fill')
        max_missing_pct (float): Maximum allowable percentage of missing values per column
        
    Returns:
        DataFrame: Combined features with missing values handled
    """
    if base_features is None or base_features.empty:
        logger.warning("Base features DataFrame is empty")
        return engineered_features
    
    if engineered_features is None or engineered_features.empty:
        logger.warning("Engineered features DataFrame is empty")
        return base_features
    
    # Combine features
    combined = pd.concat([base_features, engineered_features], axis=1)
    
    # Remove duplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # Handle columns with too many missing values
    missing_pct = combined.isna().mean()
    cols_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
    
    if cols_to_drop:
        logger.warning(f"Dropping {len(cols_to_drop)} columns with more than {max_missing_pct*100}% missing values")
        combined = combined.drop(columns=cols_to_drop)
    
    # Handle remaining missing values
    if handle_missing == 'drop':
        combined = combined.dropna()
        logger.info(f"Dropped rows with missing values. Remaining rows: {len(combined)}")
    elif handle_missing == 'impute':
        # For each column, impute missing values with median
        for col in combined.columns:
            if combined[col].isna().any():
                median = combined[col].median()
                combined[col] = combined[col].fillna(median)
                logger.info(f"Imputed {combined[col].isna().sum()} missing values in {col} with median ({median})")
    elif handle_missing == 'forward_fill':
        # Forward fill then backfill to handle missing values at the beginning
        combined = combined.ffill().bfill()
        logger.info("Applied forward fill and backfill to handle missing values")
    
    return combined


def select_features(X, y, method='correlation', threshold=0.05, top_n=None):
    """
    Select most relevant features based on specified method.
    
    Parameters:
        X (DataFrame): Features
        y (Series): Target variable
        method (str): Feature selection method ('correlation', 'mutual_info', or 'variance')
        threshold (float): Minimum threshold for feature importance
        top_n (int): If provided, select top N features regardless of threshold
        
    Returns:
        DataFrame: Selected features
    """
    if X is None or X.empty or y is None or len(y) == 0:
        logger.warning("Cannot perform feature selection on empty data")
        return X
    
    # Ensure X and y have the same indices
    common_idx = X.index.intersection(y.index)
    if len(common_idx) < len(X) or len(common_idx) < len(y):
        logger.warning(f"X and y have different indices. Using {len(common_idx)} common points")
        X = X.loc[common_idx]
        y = y.loc[common_idx]
    
    selected_cols = []
    
    if method == 'correlation':
        # Calculate correlations with target
        correlations = {}
        for col in X.columns:
            correlations[col] = abs(X[col].corr(y))
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Apply threshold or top_n
        if top_n is not None:
            selected_cols = [col for col, corr in sorted_correlations[:top_n]]
        else:
            selected_cols = [col for col, corr in sorted_correlations if corr >= threshold]
        
        logger.info(f"Selected {len(selected_cols)} features using correlation method")
        
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        
        # Sort by mutual information
        sorted_scores = mi_scores.sort_values(ascending=False)
        
        # Apply threshold or top_n
        if top_n is not None:
            selected_cols = sorted_scores.index[:top_n].tolist()
        else:
            selected_cols = sorted_scores[sorted_scores >= threshold].index.tolist()
        
        logger.info(f"Selected {len(selected_cols)} features using mutual information method")
        
    elif method == 'variance':
        from sklearn.feature_selection import VarianceThreshold
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get selected features
        selected_cols = X.columns[selector.get_support()].tolist()
        
        # Apply top_n if specified
        if top_n is not None and len(selected_cols) > top_n:
            # Calculate variances for each feature
            variances = X[selected_cols].var()
            # Sort by variance
            sorted_vars = variances.sort_values(ascending=False)
            # Select top_n features
            selected_cols = sorted_vars.index[:top_n].tolist()
        
        logger.info(f"Selected {len(selected_cols)} features using variance threshold method")
    
    else:
        logger.warning(f"Unknown feature selection method: {method}. Using all features.")
        return X
    
    # Ensure we have at least some features
    if not selected_cols:
        logger.warning("No features met the selection criteria. Using all features.")
        return X
    
    return X[selected_cols]

def create_yield_curve_features(data, country_codes):
    """
    Create yield curve specific features like slopes, term premiums, and curvature.
    
    Parameters:
        data (DataFrame): Input data containing yield information
        country_codes (list): List of country codes to process
        
    Returns:
        DataFrame: DataFrame with yield curve features
    """
    if data is None or data.empty:
        logger.warning("Cannot create yield curve features from empty data")
        return pd.DataFrame()
    
    result = pd.DataFrame(index=data.index)
    features_created = 0
    
    tenors = ['yld_2yr', 'yld_5yr', 'yld_10yr', 'yld_30yr']
    
    for country in country_codes:
        # Check which tenors are available for this country
        available_tenors = [tenor for tenor in tenors 
                          if f"{tenor}_{country}" in data.columns]
        
        if len(available_tenors) < 2:
            logger.warning(f"Not enough tenors available for {country} to create yield curve features")
            continue
        
        # Create 2s10s slope (a common yield curve measure)
        if 'yld_2yr' in available_tenors and 'yld_10yr' in available_tenors:
            result[f"slope_2s10s_{country}"] = (
                data[f"yld_10yr_{country}"] - data[f"yld_2yr_{country}"]
            )
            features_created += 1
        
        # Create 5s30s slope
        if 'yld_5yr' in available_tenors and 'yld_30yr' in available_tenors:
            result[f"slope_5s30s_{country}"] = (
                data[f"yld_30yr_{country}"] - data[f"yld_5yr_{country}"]
            )
            features_created += 1
        
        # Create curvature (if 3 tenors are available)
        if ('yld_2yr' in available_tenors and 
            'yld_10yr' in available_tenors and 
            'yld_30yr' in available_tenors):
            
            result[f"curvature_{country}"] = (
                data[f"yld_2yr_{country}"] - 
                2 * data[f"yld_10yr_{country}"] + 
                data[f"yld_30yr_{country}"]
            )
            features_created += 1
        
        # Create term premium (yield - policy rate)
        if 'pol_rat' in data.columns:
            for tenor in available_tenors:
                result[f"term_premium_{tenor}_{country}"] = (
                    data[f"{tenor}_{country}"] - data[f"pol_rat_{country}"]
                )
                features_created += 1
    
    logger.info(f"Created {features_created} yield curve features for {len(country_codes)} countries")
    return result

def normalize_features(X, method='standard', target_col=None):
    """
    Normalize features using various methods.
    
    Parameters:
        X (DataFrame): Features to normalize
        method (str): Normalization method ('standard', 'minmax', 'robust')
        target_col (str): Optional target column to exclude from normalization
        
    Returns:
        DataFrame: Normalized features
    """
    if X is None or X.empty:
        logger.warning("Cannot normalize empty DataFrame")
        return X
    
    # Make a copy to avoid modifying original
    X_norm = X.copy()
    
    # Exclude target column if specified
    cols_to_normalize = X_norm.columns
    if target_col is not None and target_col in cols_to_normalize:
        cols_to_normalize = cols_to_normalize.drop(target_col)
    
    if method == 'standard':
        # Standardize to mean=0, std=1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_norm[cols_to_normalize] = scaler.fit_transform(X_norm[cols_to_normalize])
        logger.info(f"Applied standard scaling to {len(cols_to_normalize)} features")
        
    elif method == 'minmax':
        # Scale to range [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_norm[cols_to_normalize] = scaler.fit_transform(X_norm[cols_to_normalize])
        logger.info(f"Applied min-max scaling to {len(cols_to_normalize)} features")
        
    elif method == 'robust':
        # Scale using median and interquartile range
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_norm[cols_to_normalize] = scaler.fit_transform(X_norm[cols_to_normalize])
        logger.info(f"Applied robust scaling to {len(cols_to_normalize)} features")
        
    else:
        logger.warning(f"Unknown normalization method: {method}. No scaling applied.")
    
    return X_norm