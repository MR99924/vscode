"""
Model persistence utilities for yield curve modeling.
This module contains functions for saving, loading, and versioning models.
Serves as the single source of truth for all model saving/loading operations.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, Any, Optional, Union, List, Tuple

# Initialize logging
logger = logging.getLogger(__name__)


def save_model_with_metadata(model, scaler, metadata: Dict[str, Any], 
                           country: str, tenor: str, model_type: str,
                           output_dir: str = "models") -> Dict[str, str]:
    """
    Save a trained model with its scaler and metadata.
    
    Parameters:
        model: The trained model object
        scaler: The fitted scaler object
        metadata: Dictionary containing model metadata
        country: Country code
        tenor: Tenor name
        model_type: Type of model
        output_dir: Base directory for saving models
        
    Returns:
        dict: Dictionary containing paths to saved files
    """
    try:
        # Create output directory structure
        country_dir = os.path.join(output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        
        # Generate model ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{country}_{tenor}_{model_type}_{timestamp}"
        
        # Define file paths
        model_path = os.path.join(country_dir, f"{model_id}_model.pkl")
        scaler_path = os.path.join(country_dir, f"{model_id}_scaler.pkl")
        meta_path = os.path.join(country_dir, f"{model_id}_metadata.json")
        
        # Save model
        joblib.dump(model, model_path)
        logger.debug(f"Saved model to {model_path}")
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        logger.debug(f"Saved scaler to {scaler_path}")
        
        # Prepare metadata
        complete_metadata = {
            'model_id': model_id,
            'model_type': model_type,
            'country': country,
            'tenor': tenor,
            'timestamp': timestamp,
            'creation_time': datetime.now().isoformat(),
            **metadata
        }
        
        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump(complete_metadata, f, indent=4, default=str)
        logger.debug(f"Saved metadata to {meta_path}")
        
        # Update registry
        update_model_registry(complete_metadata, model_path, scaler_path, meta_path, output_dir)
        
        logger.info(f"Successfully saved model package for {country} - {tenor} - {model_type}")
        
        return {
            'model_id': model_id,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'meta_path': meta_path
        }
        
    except Exception as e:
        logger.error(f"Error saving model package: {e}")
        logger.exception("Full traceback:")
        raise


def load_model_with_metadata(model_path: str, scaler_path: str = None, 
                           meta_path: str = None) -> Dict[str, Any]:
    """
    Load a trained model with its scaler and metadata.
    
    Parameters:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler (optional)
        meta_path: Path to the saved metadata (optional)
        
    Returns:
        dict: Dictionary containing loaded model, scaler, and metadata
    """
    try:
        result = {}
        
        # Load model
        if os.path.exists(model_path):
            result['model'] = joblib.load(model_path)
            logger.debug(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load scaler if path provided
        if scaler_path and os.path.exists(scaler_path):
            result['scaler'] = joblib.load(scaler_path)
            logger.debug(f"Loaded scaler from {scaler_path}")
        elif scaler_path:
            logger.warning(f"Scaler file not found: {scaler_path}")
            result['scaler'] = None
        
        # Load metadata if path provided
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                result['metadata'] = json.load(f)
            logger.debug(f"Loaded metadata from {meta_path}")
        elif meta_path:
            logger.warning(f"Metadata file not found: {meta_path}")
            result['metadata'] = {}
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading model package: {e}")
        logger.exception("Full traceback:")
        raise


def save_model_package(model_package: Dict[str, Any], country: str, tenor: str, 
                      output_dir: str = "models") -> str:
    """
    Save a complete model package with all components.
    
    Parameters:
        model_package: Dictionary containing model, scaler, and metadata
        country: Country code
        tenor: Tenor name
        output_dir: Base directory for saving models
        
    Returns:
        str: Path to the saved model package
    """
    try:
        # Create output directory
        country_dir = os.path.join(output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        
        # Generate filename
        model_type = model_package.get('model_type', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{country}_{tenor}_{model_type}_{timestamp}_package.pkl"
        package_path = os.path.join(country_dir, filename)
        
        # Add metadata to package
        enhanced_package = {
            **model_package,
            'save_timestamp': timestamp,
            'save_time': datetime.now().isoformat(),
            'country': country,
            'tenor': tenor
        }
        
        # Save complete package
        joblib.dump(enhanced_package, package_path)
        logger.info(f"Saved model package to {package_path}")
        
        return package_path
        
    except Exception as e:
        logger.error(f"Error saving model package: {e}")
        logger.exception("Full traceback:")
        raise


def load_model_package(package_path: str) -> Dict[str, Any]:
    """
    Load a complete model package.
    
    Parameters:
        package_path: Path to the saved model package
        
    Returns:
        dict: Complete model package
    """
    try:
        if not os.path.exists(package_path):
            raise FileNotFoundError(f"Model package not found: {package_path}")
        
        package = joblib.load(package_path)
        logger.info(f"Loaded model package from {package_path}")
        
        return package
        
    except Exception as e:
        logger.error(f"Error loading model package: {e}")
        logger.exception("Full traceback:")
        raise


def save_best_model(model_result: Dict[str, Any], country: str, tenor: str,
                   model_type: str, output_dir: str = "models") -> str:
    """
    Save the best model for a country-tenor combination.
    
    Parameters:
        model_result: Dictionary containing model results
        country: Country code
        tenor: Tenor name
        model_type: Type of model
        output_dir: Base directory for saving models
        
    Returns:
        str: Path to the saved best model
    """
    try:
        # Create output directory
        country_dir = os.path.join(output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        
        # Create best model path
        best_model_path = os.path.join(country_dir, f"{country}_{tenor}_best_model.pkl")
        
        # Extract components from model result
        model = model_result.get('model')
        scaler = model_result.get('scaler')
        metrics = model_result.get('metrics', {})
        feature_names = model_result.get('feature_names', [])
        
        if model is None:
            raise ValueError("No model found in model_result")
        
        # Create best model package
        best_model_package = {
            'model': model,
            'scaler': scaler,
            'model_type': model_type,
            'feature_names': feature_names,
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance': metrics,
            'country': country,
            'tenor': tenor,
            'is_best_model': True
        }
        
        # Save best model package
        joblib.dump(best_model_package, best_model_path)
        logger.info(f"Saved best model for {country} - {tenor}: {model_type} to {best_model_path}")
        
        return best_model_path
        
    except Exception as e:
        logger.error(f"Error saving best model for {country} - {tenor}: {e}")
        logger.exception("Full traceback:")
        raise


def load_best_model(country: str, tenor: str, output_dir: str = "models") -> Dict[str, Any]:
    """
    Load the best model for a country-tenor combination.
    
    Parameters:
        country: Country code
        tenor: Tenor name
        output_dir: Base directory for models
        
    Returns:
        dict: Best model package
    """
    try:
        best_model_path = os.path.join(output_dir, country, f"{country}_{tenor}_best_model.pkl")
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found: {best_model_path}")
        
        best_model_package = joblib.load(best_model_path)
        logger.info(f"Loaded best model for {country} - {tenor}")
        
        return best_model_package
        
    except Exception as e:
        logger.error(f"Error loading best model for {country} - {tenor}: {e}")
        logger.exception("Full traceback:")
        raise


def save_consolidated_fitted_values(all_fitted_values: Dict[str, pd.DataFrame], 
                                  results_summary: Dict[str, Any] = None, 
                                  config = None) -> None:
    """
    Save only the best model fitted values for each country-tenor combination.
    Also creates a summary file of best model locations with absolute paths.
    
    Parameters:
        all_fitted_values: Dictionary containing all fitted values from model training
        results_summary: Summary of model results containing best model selections
        config: Configuration module containing paths and settings
    """
    if not all_fitted_values:
        logger.warning("No fitted values available to consolidate")
        return

    try:
        # Create a registry of best models with absolute paths
        best_models_registry = {}

        # If results_summary is provided, filter to include only the best models
        if results_summary is not None and 'best_models' in results_summary:
            best_fitted_values = {}

            # Iterate through best models by country and tenor
            for country, tenor_models in results_summary['best_models'].items():
                for tenor, model_type in tenor_models.items():
                    # Create a key that matches the format in all_fitted_values
                    key = f"{country}_{tenor}_{model_type}"

                    # Use absolute path to avoid path resolution issues
                    model_dir = getattr(config, 'MODEL_DIR', '.') if config else '.'
                    country_dir = os.path.join(model_dir, country)
                    os.makedirs(country_dir, exist_ok=True)
                    model_path = os.path.abspath(os.path.join(country_dir, f"{country}_{tenor}_best_model.pkl"))

                    # Record in registry
                    best_models_registry[f"{country}_{tenor}"] = {
                        'model_type': model_type,
                        'model_path': model_path
                    }

                    # If this key exists in all_fitted_values, add it to best_fitted_values
                    if key in all_fitted_values:
                        best_fitted_values[key] = all_fitted_values[key]
                        logger.info(f"Adding best model {key} to consolidated results")
                    else:
                        logger.warning(f"Best model {key} not found in fitted values")

            # Save the registry to a JSON file
            output_dir = getattr(config, 'OUTPUT_DIR', '.') if config else '.'
            os.makedirs(output_dir, exist_ok=True)
            registry_path = os.path.join(output_dir, "best_models_registry.json")
            with open(registry_path, 'w') as f:
                json.dump(best_models_registry, f, indent=4)
            logger.info(f"Saved best models registry to {registry_path}")

            # If we found any best models, use only those
            if best_fitted_values:
                logger.info(f"Saving {len(best_fitted_values)} best models (filtering from {len(all_fitted_values)} total models)")
                consolidated_df = pd.concat(best_fitted_values.values(), ignore_index=True)
                
                # Save to output directory
                best_fitted_path = os.path.join(output_dir, "best_fitted_values.csv")
                consolidated_df.to_csv(best_fitted_path, index=False)
                logger.info(f"Saved best fitted values to '{best_fitted_path}'")

                # Also save the original complete set if needed
                all_fitted_path = os.path.join(output_dir, "all_fitted_values.csv")
                consolidated_all_df = pd.concat(all_fitted_values.values(), ignore_index=True)
                consolidated_all_df.to_csv(all_fitted_path, index=False)
                logger.info(f"Also saved complete fitted values to '{all_fitted_path}'")

                return

        # Fallback to saving all values if no filtering was done
        logger.info(f"Saving all {len(all_fitted_values)} fitted models (no filtering)")
        output_dir = getattr(config, 'OUTPUT_DIR', '.') if config else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        consolidated_df = pd.concat(all_fitted_values.values(), ignore_index=True)
        all_fitted_path = os.path.join(output_dir, "all_fitted_values.csv")
        consolidated_df.to_csv(all_fitted_path, index=False)
        logger.info(f"Saved consolidated fitted values to '{all_fitted_path}'")

    except Exception as e:
        logger.error(f"Error saving consolidated fitted values: {e}")
        logger.exception("Full traceback:")


def update_model_registry(metadata: Dict[str, Any], model_path: str, scaler_path: str, 
                         meta_path: str, output_dir: str) -> None:
    """
    Update the model registry with a new model entry.
    
    Parameters:
        metadata: Model metadata
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        meta_path: Path to the saved metadata
        output_dir: Base directory for models
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
        except Exception as e:
            logger.warning(f"Could not load existing model registry, creating new one: {e}")
    
    # Add new entry and save
    registry.append(entry)
    
    try:
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
        logger.info(f"Updated model registry with {metadata['model_id']}")
    except Exception as e:
        logger.error(f"Error updating model registry: {e}")


def apply_model(model, X: pd.DataFrame, scaler=None) -> Optional[np.ndarray]:
    """
    Apply a loaded model to new data.
    
    Parameters:
        model: The loaded model object
        X: Input features
        scaler: The scaler to preprocess features
        
    Returns:
        array: Model predictions or None if error
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
        logger.error(f"Error applying model: {e}")
        logger.exception("Full traceback:")
        return None


def get_model_summary(base_dir: str = 'models') -> Optional[pd.DataFrame]:
    """
    Get a summary of all models in the registry.
    
    Parameters:
        base_dir: Base directory for models
        
    Returns:
        DataFrame: Summary of all models or None if error
    """
    registry_path = os.path.join(base_dir, "model_registry.json")
    
    if not os.path.exists(registry_path):
        logger.error(f"Model registry not found at {registry_path}")
        return None
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Error loading model registry: {e}")
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


def cleanup_old_models(country: str = None, tenor: str = None, keep_best: int = 1, 
                      keep_latest: int = 2, base_dir: str = 'models') -> int:
    """
    Clean up old model versions while keeping the best/latest ones.
    
    Parameters:
        country: Country code to filter models
        tenor: Tenor to filter models
        keep_best: Number of best models to keep (by RMSE)
        keep_latest: Number of latest models to keep
        base_dir: Base directory for models
        
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
        logger.error(f"Error loading model registry: {e}")
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
            logger.warning(f"Error deleting model files for {entry.get('model_id')}: {e}")
    
    # Update registry
    if deleted_count > 0:
        new_registry = [entry for entry in registry if entry not in models_to_delete]
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(new_registry, f, indent=4)
            
            logger.info(f"Removed {len(models_to_delete)} old models from registry")
        except Exception as e:
            logger.error(f"Error updating registry after cleanup: {e}")
    
    return len(models_to_delete)


def batch_save_models(model_results: Dict[str, Dict[str, Any]], output_dir: str = "models") -> Dict[str, str]:
    """
    Save multiple models in batch with consistent naming and metadata.
    
    Parameters:
        model_results: Dictionary of model results keyed by model identifier
        output_dir: Base directory for saving models
        
    Returns:
        dict: Dictionary of saved model paths keyed by model identifier
    """
    saved_paths = {}
    
    for model_key, model_result in model_results.items():
        try:
            # Parse model key to extract components
            parts = model_key.split('_')
            if len(parts) >= 3:
                country = parts[0]
                tenor = parts[1]
                model_type = '_'.join(parts[2:])
            else:
                logger.warning(f"Could not parse model key: {model_key}")
                continue
            
            # Extract metadata
            metadata = {
                'metrics': model_result.get('metrics', {}),
                'feature_names': model_result.get('feature_names', []),
                'training_info': model_result.get('training_info', {})
            }
            
            # Save model
            paths = save_model_with_metadata(
                model=model_result.get('model'),
                scaler=model_result.get('scaler'),
                metadata=metadata,
                country=country,
                tenor=tenor,
                model_type=model_type,
                output_dir=output_dir
            )
            
            saved_paths[model_key] = paths['model_path']
            
        except Exception as e:
            logger.error(f"Error saving model {model_key}: {e}")
            logger.exception("Full traceback:")
    
    logger.info(f"Batch saved {len(saved_paths)} models")
    return saved_paths


def validate_model_files(base_dir: str = 'models') -> Dict[str, List[str]]:
    """
    Validate that all model files referenced in the registry actually exist.
    
    Parameters:
        base_dir: Base directory for models
        
    Returns:
        dict: Dictionary with 'valid' and 'missing' file lists
    """
    registry_path = os.path.join(base_dir, "model_registry.json")
    
    if not os.path.exists(registry_path):
        logger.error(f"Model registry not found at {registry_path}")
        return {'valid': [], 'missing': []}
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Error loading model registry: {e}")
        return {'valid': [], 'missing': []}
    
    valid_files = []
    missing_files = []
    
    for entry in registry:
        model_id = entry.get('model_id', 'unknown')
        files_to_check = [
            entry.get('model_path'),
            entry.get('scaler_path'),
            entry.get('meta_path')
        ]
        
        for file_path in files_to_check:
            if file_path:
                if os.path.exists(file_path):
                    valid_files.append(file_path)
                else:
                    missing_files.append(f"{model_id}: {file_path}")
    
    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing model files")
        for missing in missing_files:
            logger.warning(f"Missing: {missing}")
    
    logger.info(f"Validated {len(valid_files)} existing model files")
    
    return {'valid': valid_files, 'missing': missing_files}