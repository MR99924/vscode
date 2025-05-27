"""
Model persistence utilities for yield curve modeling.
This module contains functions for saving, loading, and versioning models.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
# Configure logging
logger = logging.getLogger(__name__)



def update_model_registry(metadata, model_path, scaler_path, meta_path, output_dir):
    """
    Update the model registry with a new model entry.
    
    Parameters:
        metadata (dict): Model metadata
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        meta_path (str): Path to the saved metadata
        output_dir (str): Base directory for models
    """
    registry_path = os.path.join(output_dir, "model_registry.json")
    
    # Create registry entry
    entry = {
        'model_id': metadata['model_id'],
        'model_type': metadata['model_type'],
        'country': metadata['country'],
        'tenor': metadata['tenor'],
        'timestamp': metadata['timestamp'],
        'model_path': model_path,
        'scaler_path': scaler_path,
        'meta_path': meta_path,
        'metrics': metadata.get('metrics', {})
    }
    
    # Load existing registry or create new one
    registry = []
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except:
            logger.warning("Could not load existing model registry, creating new one")
    
    # Add new entry and save
    registry.append(entry)
    
    try:
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
        logger.info(f"Updated model registry with {metadata['model_id']}")
    except Exception as e:
        logger.error(f"Error updating model registry: {str(e)}")


def apply_model(model, X, scaler=None):
    """
    Apply a loaded model to new data.
    
    Parameters:
        model: The loaded model object
        X (DataFrame): Input features
        scaler: The scaler to preprocess features
        
    Returns:
        array: Model predictions
    """
    if model is None:
        logger.error("No model provided")
        return None
    
    try:
        # Preprocess features if scaler is provided
        X_processed = X
        if scaler is not None:
            X_processed = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_processed)
        return predictions
    
    except Exception as e:
        logger.error(f"Error applying model: {str(e)}")
        return None

def get_model_summary(base_dir='models'):
    """
    Get a summary of all models in the registry.
    
    Parameters:
        base_dir (str): Base directory for models
        
    Returns:
        DataFrame: Summary of all models
    """
    registry_path = os.path.join(base_dir, "model_registry.json")
    
    if not os.path.exists(registry_path):
        logger.error(f"Model registry not found at {registry_path}")
        return None
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Error loading model registry: {str(e)}")
        return None
    
    # Extract relevant information
    summary_data = []
    for entry in registry:
        model_id = entry.get('model_id', '')
        model_type = entry.get('model_type', '')
        country = entry.get('country', '')
        tenor = entry.get('tenor', '')
        timestamp = entry.get('timestamp', '')
        
        # Get metrics if available
        metrics = entry.get('metrics', {})
        rmse = metrics.get('rmse', None)
        r2 = metrics.get('r2', None)
        
        summary_data.append({
            'model_id': model_id,
            'model_type': model_type,
            'country': country,
            'tenor': tenor,
            'timestamp': timestamp,
            'rmse': rmse,
            'r2': r2
        })
    
    # Convert to DataFrame
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Convert timestamp to datetime for sorting
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
        
        # Sort by country, tenor, and timestamp (descending)
        df = df.sort_values(['country', 'tenor', 'timestamp'], 
                           ascending=[True, True, False])
        
        return df
    
    return pd.DataFrame()

def cleanup_old_models(country=None, tenor=None, keep_best=1, keep_latest=2, base_dir='models'):
    """
    Clean up old model versions while keeping the best/latest ones.
    
    Parameters:
        country (str): Country code to filter models
        tenor (str): Tenor to filter models
        keep_best (int): Number of best models to keep (by RMSE)
        keep_latest (int): Number of latest models to keep
        base_dir (str): Base directory for models
        
    Returns:
        int: Number of models removed
    """
    registry_path = os.path.join(base_dir, "model_registry.json")
    
    if not os.path.exists(registry_path):
        logger.error(f"Model registry not found at {registry_path}")
        return 0
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Error loading model registry: {str(e)}")
        return 0
    
    # Filter models by country and tenor
    filtered_registry = registry
    if country:
        filtered_registry = [entry for entry in filtered_registry 
                            if entry.get('country') == country]
    
    if tenor:
        filtered_registry = [entry for entry in filtered_registry 
                            if entry.get('tenor') == tenor]
    
    # Group models by country and tenor
    groups = {}
    for entry in filtered_registry:
        key = f"{entry.get('country')}_{entry.get('tenor')}_{entry.get('model_type')}"
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)
    
    models_to_keep = []
    for key, entries in groups.items():
        # Sort by timestamp (descending)
        by_date = sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Keep the latest N models
        models_to_keep.extend(by_date[:keep_latest])
        
        # Sort by RMSE (ascending)
        by_performance = sorted(entries, 
                              key=lambda x: x.get('metrics', {}).get('rmse', float('inf')))
        
        # Keep the best performing N models
        models_to_keep.extend(by_performance[:keep_best])
    
    # Remove duplicates
    models_to_keep = list({entry['model_id']: entry for entry in models_to_keep}.values())
    
    # Identify models to delete
    models_to_delete = [entry for entry in filtered_registry 
                       if entry not in models_to_keep]
    
    # Delete models
    deleted_count = 0
    for entry in models_to_delete:
        try:
            model_path = entry.get('model_path')
            scaler_path = entry.get('scaler_path')
            meta_path = entry.get('meta_path')
            
            # Delete files if they exist
            for path in [model_path, scaler_path, meta_path]:
                if path and os.path.exists(path):
                    os.remove(path)
                    deleted_count += 1
        except Exception as e:
            logger.warning(f"Error deleting model files for {entry.get('model_id')}: {str(e)}")
    
    # Update registry
    if deleted_count > 0:
        new_registry = [entry for entry in registry if entry not in models_to_delete]
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(new_registry, f, indent=4)
            
            logger.info(f"Removed {len(models_to_delete)} old models from registry")
        except Exception as e:
            logger.error(f"Error updating registry after cleanup: {str(e)}")
    
    return len(models_to_delete)