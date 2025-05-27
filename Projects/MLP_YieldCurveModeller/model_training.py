"""
Advanced model training functions for yield curve modeling.
This module contains functions for training models with cross-validation, 
hyperparameter optimization, and ensemble methods.
"""
import warnings
try:
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
except ImportError:
    warnings.filterwarnings("ignore", message=".*R\\^2 score.*")

import numpy as np
import pandas as pd
import os
import logging
import json
import time
from datetime import datetime  
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import joblib
from model_classes import add_missing_mlp_attributes

# Import custom model classes
from model_classes import TimeSeriesEstimator, EnsembleModel

# Configure logging
logger = logging.getLogger(__name__)

def safe_r2_score(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    """
    Safe implementation of R² score that handles edge cases.
    """
    import numpy as np
    from sklearn.metrics import r2_score
    
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check for too few samples
    if len(y_true) < 2:
        return 0.0
    
    # Check for constant values
    if np.all(y_true == y_true[0]):
        return 0.0
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return r2_score(y_true, y_pred, 
                          sample_weight=sample_weight, 
                          multioutput=multioutput)
    except Exception:
        return 0.0

    # Data validation and cleaning step
def clean_data_for_training(X, y):
    """Clean data by handling NaN, inf values and extreme outliers."""
        
    # Create copies to avoid modifying originals
    X_clean = X.copy()
    y_clean = y.copy()
        
    # Check if there are any non-finite values in X
    if not np.all(np.isfinite(X_clean)):
        # Replace inf/NaN with appropriate values
        X_clean = np.nan_to_num(X_clean, nan=0, posinf=1e10, neginf=-1e10)
            
        # For remaining values, cap extreme outliers
        for col in range(X_clean.shape[1]):
            if np.any(X_clean[:, col]):  # Skip columns with all zeros
                valid_values = X_clean[:, col][np.isfinite(X_clean[:, col])]
                if len(valid_values) > 0:
                    q1 = np.percentile(valid_values, 1)
                    q3 = np.percentile(valid_values, 99)
                    X_clean[:, col] = np.clip(X_clean[:, col], q1, q3)
        
    # Handle NaN values in y if any
    if np.any(np.isnan(y_clean)):
        y_clean = np.nan_to_num(y_clean, nan=np.nanmedian(y_clean))
        
    return X_clean, y_clean

def train_model_with_cross_validation(X, y, model_type, n_splits=5, params=None):
    """
    Simplified training function that uses a single train-test split instead of complex cross-validation.
    This avoids all the indexing issues while still providing model evaluation.
    
    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        model_type (str): Type of model to train
        n_splits (int): Not used in this implementation
        params (dict): Model parameters
        
    Returns:
        tuple: (model_package, metrics, feature_importance)
    """
    # Input validation
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        logger.error("Cannot train model with empty data")
        return None, None, None
    
    logger.info(f"Using simplified train-test split approach instead of cross-validation")
    
    # Clean data
    X_cleaned, y_cleaned = clean_data_for_training(X, y)
    
    # Double-check we still have data after cleaning
    if len(X_cleaned) == 0 or len(y_cleaned) == 0:
        logger.error("No valid data after cleaning")
        return None, None, None
    
    # Initialize model based on type with default params if none provided
    if params is None:
        params = {}
    
    # Create base model with appropriate parameters
    try:
        if model_type == 'mlp':
            base_model = MLPRegressor(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (64, 32)),
                activation=params.get('activation', 'relu'),
                solver=params.get('solver', 'adam'),
                alpha=params.get('alpha', 0.0001),
                batch_size=params.get('batch_size', 'auto'),
                max_iter=params.get('max_iter', 1000),
                early_stopping=params.get('early_stopping', True),
                validation_fraction=params.get('validation_fraction', 0.1),
                n_iter_no_change=params.get('n_iter_no_change', 10),
                random_state=42
            )
        elif model_type == 'gbm':
            base_model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                subsample=params.get('subsample', 1.0),
                random_state=42
            )
        elif model_type == 'xgboost':
            base_model = xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                min_child_weight=params.get('min_child_weight', 1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                random_state=42
            )
        elif model_type == 'elasticnet':
            base_model = ElasticNet(
                alpha=params.get('alpha', 1.0),
                l1_ratio=params.get('l1_ratio', 0.5),
                max_iter=params.get('max_iter', 1000),
                random_state=42
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None, None, None
    except Exception as e:
        logger.error(f"Error initializing {model_type} model: {str(e)}")
        return None, None, None
        
    # Simple train-test split (80-20)
    try:
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_cleaned, y_cleaned, test_size=0.5, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        start_time = time.time()
        model = base_model
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Generate predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = safe_r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = safe_r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        logger.info(f"Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        # Store metrics in a format compatible with the original function
        cv_results = {
            'train_mse': [train_mse],
            'train_rmse': [train_rmse],
            'train_r2': [train_r2],
            'train_mae': [train_mae],
            'test_mse': [test_mse],
            'test_rmse': [test_rmse],
            'test_r2': [test_r2],
            'test_mae': [test_mae],
            'avg_train_mse': train_mse,
            'avg_train_rmse': train_rmse,
            'avg_train_r2': train_r2,
            'avg_test_mse': test_mse,
            'avg_test_rmse': test_rmse,
            'avg_test_r2': test_r2
        }
        
        # Extract feature importance
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
                feature_importance = dict(zip(X.columns, importance))
            elif model_type == 'mlp' and hasattr(model, 'coefs_'):
                # Neural network - use first layer weights
                weights = np.abs(model.coefs_[0])
                importance = np.mean(weights, axis=1)
                importance = importance / np.sum(importance) if np.sum(importance) > 0 else np.ones(len(importance)) / len(importance)
                feature_importance = dict(zip(X.columns, importance))
            elif hasattr(model, 'coef_'):
                # Linear models
                importance = np.abs(model.coef_)
                feature_importance = dict(zip(X.columns, importance))
        except Exception as e:
            logger.warning(f"Error extracting feature importance: {str(e)}")
            
        # Create model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'model_type': model_type,
            'params': params,
            'feature_names': list(X.columns),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_data_size': len(X)
        }
        
        return model_package, cv_results, feature_importance
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def optimize_hyperparameters(X, y, model_type, param_grid=None, n_splits=3, 
                            n_iter=None, cv_method='grid', scoring='neg_root_mean_squared_error'):
    """
    Perform hyperparameter optimization for a model using time series cross-validation.
    
    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        model_type (str): Type of model to train
        param_grid (dict): Dictionary of parameters to search
        n_splits (int): Number of splits for cross-validation
        n_iter (int): Number of iterations for random search (if cv_method='random')
        cv_method (str): Cross-validation method ('grid' or 'random')
        scoring (str): Scoring metric to optimize
        
    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    if X is None or X.empty or y is None or len(y) == 0:
        logger.error("Cannot optimize hyperparameters with empty data")
        return None, None, None
    
    # Set default param grid if not provided
    if param_grid is None:
        if model_type == 'mlp':
            param_grid = {
                'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64, 32)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        elif model_type == 'gbm':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        elif model_type == 'elasticnet':
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        elif model_type == 'lstm':
            param_grid = {
                'units': [32, 64, 128],
                'dropout': [0.1, 0.2, 0.3],
                'batch_size': [16, 32, 64]
            }
        elif model_type == 'arima':
            param_grid = {
                'p': [1, 2, 3],
                'd': [0, 1],
                'q': [0, 1, 2]
            }
    
    # Initialize base model
    if model_type == 'mlp':
        base_model = MLPRegressor(early_stopping=True, random_state=42)
    elif model_type == 'gbm':
        base_model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'xgboost':
        base_model = xgb.XGBRegressor(random_state=42)
    elif model_type == 'elasticnet':
        base_model = ElasticNet(random_state=42)
    elif model_type == 'arima':
        base_model = TimeSeriesEstimator()
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, None, None
    
    # Create a pipeline with scaling
    if model_type != 'lstm' and model_type != 'arima':
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
    else:
        # LSTM and ARIMA handle scaling differently
        pipeline = base_model
        
        # For LSTM and ARIMA, we need to scale manually before CV
        if model_type == 'lstm':
            scaler = StandardScaler()
            X = pd.DataFrame(
                scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
    
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Perform hyperparameter search
    start_time = time.time()
    logger.info(f"Starting hyperparameter optimization with {cv_method} search")
    
    if cv_method == 'grid':
        if model_type != 'lstm' and model_type != 'arima':
            search = GridSearchCV(
                pipeline,
                {'model__' + k: v for k, v in param_grid.items()},
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
    elif cv_method == 'random':
        if n_iter is None:
            n_iter = 10
        
        if model_type != 'lstm' and model_type != 'arima':
            search = RandomizedSearchCV(
                pipeline,
                {'model__' + k: v for k, v in param_grid.items()},
                n_iter=n_iter,
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=n_iter,
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
    else:
        logger.error(f"Unknown CV method: {cv_method}")
        return None, None, None
    
    # Fit the search
    search.fit(X, y)
    
    # Get results
    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best score: {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")
    
    if model_type == 'mlp':
        final_model = add_missing_mlp_attributes(final_model)

    # Convert parameter names back for model
    if model_type != 'lstm' and model_type != 'arima':
        best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    else:
        best_params = search.best_params_
    
        
    # Train final model with best parameters
    final_model, cv_results, _ = train_model_with_cross_validation(
        X, y, model_type=model_type, params=best_params
    )

    # If it's an MLP model, apply the attribute fix
    if model_type == 'mlp' and final_model is not None and 'model' in final_model:
        final_model['model'] = add_missing_mlp_attributes(final_model['model'])

    return final_model, best_params, search.cv_results_

def train_benchmark_models(X, y, models_to_train=None, country=None, tenor=None):
    """
    Train multiple model types for benchmarking.
    
    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        models_to_train (list): List of model types to train
        country (str): Country name for logging
        tenor (str): Tenor name for logging
        
    Returns:
        dict: Dictionary of trained model packages and results
    """
    if X is None or X.empty or y is None or len(y) == 0:
        logger.error("Cannot train benchmark models with empty data")
        return None
    
    # Default models to train
    if models_to_train is None:
        models_to_train = ['mlp', 'gbm', 'xgboost', 'elasticnet']
    
    results = {}
    
    for model_type in models_to_train:
        logger.info(f"Training {model_type} model for {country} - {tenor}")
        
        try:
            # Train model with default parameters
            model_package, cv_results, feature_importance = train_model_with_cross_validation(
                X, y, model_type=model_type
            )
            
            # Store results
            results[model_type] = {
                'model_package': model_package,
                'cv_results': cv_results,
                'feature_importance': feature_importance
            }
            
            # Log performance
            logger.info(f"{model_type} model - Avg RMSE: {cv_results['avg_test_rmse']:.4f}, "
                       f"Avg R²: {cv_results['avg_test_r2']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
    
    # Compare models
    if len(results) > 0:
        logger.info("Model comparison:")
        model_comparison = pd.DataFrame({
            model_type: {
                'RMSE': results[model_type]['cv_results']['avg_test_rmse'],
                'R²': results[model_type]['cv_results']['avg_test_r2'],
                'MAE': results[model_type]['cv_results']['avg_test_mae']
            }
            for model_type in results
        }).T
        
        logger.info("\n" + model_comparison.to_string())
        
        # Identify best model based on RMSE
        best_model = model_comparison['RMSE'].idxmin()
        logger.info(f"Best model based on RMSE: {best_model}")
        
        # Add comparison to results
        results['comparison'] = model_comparison
        results['best_model'] = best_model
    
    return results

def build_ensemble_model(trained_models, X, y, weights=None, weighted_by_performance=True):
    """
    Build an ensemble model from multiple trained models.
    
    Parameters:
        trained_models (dict): Dictionary of trained model packages
        X (DataFrame): Feature matrix for validation
        y (Series): Target variable for validation
        weights (dict): Manual weights for each model
        weighted_by_performance (bool): Whether to weight models by performance
        
    Returns:
        tuple: (ensemble_model, performance_metrics)
    """
    if trained_models is None or len(trained_models) == 0:
        logger.error("Cannot build ensemble with no trained models")
        return None, None
    
    # Extract models from packages
    models = {}
    for model_type, package in trained_models.items():
        if model_type != 'comparison' and model_type != 'best_model':
            models[model_type] = package['model_package']['model']
    
    # Create ensemble model instance
    ensemble = EnsembleModel()
    
    # Add each model to the ensemble
    for model_type, model in models.items():
        ensemble.add_model(model)
    
    # If weighting by performance, calculate weights based on validation RMSE
    if weighted_by_performance:
        # Scale X for each model
        scalers = {model_type: package['model_package']['scaler'] 
                 for model_type, package in trained_models.items()
                 if model_type != 'comparison' and model_type != 'best_model'}
        
        scaled_data = {}
        for model_type, scaler in scalers.items():
            scaled_data[model_type] = scaler.transform(X)
        
        # Get predictions for each model
        predictions = {}
        rmse_values = {}
        
        for model_type, model in models.items():
            X_scaled = scaled_data.get(model_type, X)  # Use scaled data if available
            preds = model.predict(X_scaled)
            predictions[model_type] = preds
            rmse = np.sqrt(mean_squared_error(y, preds))
            rmse_values[model_type] = rmse
        
        # Calculate weights as inverse of RMSE (better models get higher weights)
        inverse_rmse = {model_type: 1/rmse for model_type, rmse in rmse_values.items()}
        total = sum(inverse_rmse.values())
        calculated_weights = {model_type: inv_rmse/total for model_type, inv_rmse in inverse_rmse.items()}
        
        # Set weights
        ensemble.weights = [calculated_weights[model_type] for model_type in models.keys()]
        
        logger.info("Ensemble model weights based on performance:")
        for model_type, weight in calculated_weights.items():
            logger.info(f"  {model_type}: {weight:.4f}")
    
    # If manual weights provided, use those
    elif weights is not None:
        ensemble.weights = [weights.get(model_type, 1.0) for model_type in models.keys()]
        
        logger.info("Ensemble model weights (manual):")
        for model_type, weight in zip(models.keys(), ensemble.weights):
            logger.info(f"  {model_type}: {weight:.4f}")
    
    # Generate predictions with ensemble
    predictions = ensemble.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }
    
    logger.info(f"Ensemble model - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    
    return ensemble, metrics

def fix_mlp_missing_attributes(model):
    """Fix missing attributes in MLPRegressor"""
    if hasattr(model, '__class__') and model.__class__.__name__ == 'MLPRegressor':
        if not hasattr(model, '_best_coefs') and hasattr(model, 'coefs_'):
            model._best_coefs = model.coefs_
        if not hasattr(model, '_best_intercepts') and hasattr(model, 'intercepts_'):
            model._best_intercepts = model.intercepts_
    return model