import numpy as np
import datetime
import time
import os
import json
import pandas as pd
import logging
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split

# Local imports - assuming these are in the same package/directory
import config
from feature_engineering import forward_fill_to_current_date
from model_evaluation import evaluate_model_performance, perform_residual_analysis
from model_training import optimize_hyperparameters, train_model_with_cross_validation, build_ensemble_model, train_benchmark_models
from prepare_data import prepare_data
from model_tester import extract_feature_names

# Get logger instance - will be configured by mlp_yield_modeller.py
logger = logging.getLogger(__name__)

# Global variables - consider dependency injection for larger systems
model_predictions = {}
all_fitted_values = {}

def get_bloomberg_date(bbg_client, tickers, date_from, date_to, field="PX_LAST", periodicity="DAILY"):
    """ Fetch data from Bloomberg with pandas 2.0+ compatibility. """
    try:
        print(bbg_client)

        if bbg_client is None:
            logger.error("Bloomberg client is None")
            return pd.DataFrame(index=pd.date_range(date_from, date_to))

        # Temporarily suppress the append warning/error
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Try the Bloomberg request
                       
            df = bbg_client.historicalRequest(
                tickers,
                field,
                date_from,
                date_to,
                periodicitySelection=periodicity
            )

            print(df)

        # Process the returned data
        if hasattr(df, 'pivot_table'):
            df = pd.pivot_table(df, values='bbergvalue', index=['bbergdate'], columns=['bbergsymbol'], aggfunc='first', # Changed from np.max
            )

        # Rest of your existing code...
        for ticker in tickers:
            if ticker not in df.columns:
                logger.warning(f"Ticker {ticker} not found in Bloomberg data")

        existing_tickers = [t for t in tickers if t in df.columns]
        if existing_tickers:
            df = df[existing_tickers]
        else:
            logger.warning("None of the requested tickers were found")
            df = pd.DataFrame(index=pd.date_range(date_from, date_to))

        return df

    except Exception as e:
        logger.error(f"Error fetching Bloomberg data: {e}")
        # Return empty DataFrame with proper date index as fallback
        return pd.DataFrame(index=pd.date_range(date_from, date_to))
    

def show_popup_alert(title, message):
    """
    Show a popup alert dialog with the given title and message.
    Falls back to console output if GUI is not available.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showwarning(title, message)
        root.destroy()
        return True
    except Exception:
        logger.warning(f"{title}: {message}")
        return False

def validate_model_training_data(x, y, country, tenor_name):
    """
    Validate input data for model training.
    
    Parameters:
        x: DataFrame - Feature data
        y: Series - Target data
        country: str - Country name
        tenor_name: str - Tenor name
        
    Returns:
        dict: Validation results with status and error message if any
    """
    validation_result = {'valid': True, 'error': None}
    
    if x is None or x.empty or y is None or len(y) == 0:
        validation_result['valid'] = False
        validation_result['error'] = 'No overlapping data between features and target'
        logger.error(f"Insufficient data to train model for {country} - {tenor_name}")
        return validation_result
        
    if len(x) < config.MIN_DATA_POINTS_FOR_MODEL:
        validation_result['valid'] = False
        validation_result['error'] = f'Only {len(x)} data points available (minimum {config.MIN_DATA_POINTS_FOR_MODEL} required)'
        logger.error(f"Not enough data points ({len(x)}) to train a reliable model")
        return validation_result
    
    return validation_result

def prepare_model_training_data(country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, 
                              iip_gdp, act_track, risk_rating, historical_forecasts, unemployment_rate, 
                              predicted_yields=None, use_advanced_training=True):
    """
    Prepare data for model training with enhanced feature engineering.
    
    Returns:
        tuple: (x, y, feature_details) or (None, None, None) if preparation fails
    """
    try:
        x, y, feature_details = prepare_data(
            country=country,
            tenor_name=tenor_name,
            country_code_mapping=country_code_mapping,
            tenor=tenor_data,
            pol_rat=pol_rat,
            cpi_inf=cpi_inf,
            act_track=act_track,
            risk_rating=risk_rating,
            historical_forecasts=historical_forecasts,
            unemployment_rate=unemployment_rate,
            iip_gdp=iip_gdp,
            predicted_yields=predicted_yields,
            forward_fill=True,
            enhance_features=use_advanced_training,
            handle_missing='ffill'
        )
        
        if 'feature_columns' not in feature_details:
            feature_details['feature_columns'] = feature_details
            
        return x, y, feature_details
        
    except Exception as e:
        logger.error(f"Error preparing data for {country} - {tenor_name}: {str(e)}")
        return None, None, None

def train_model_simple_split(X, y, model_type, params=None, test_size=0.3):
    """
    Train a model using a simple train/test split instead of cross-validation.
    Use this for very small datasets.
    """
    logger.info(f"Training model with simple split: {model_type}, test_size={test_size}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model based on type
    model = create_model_by_type(model_type, params)
    
    # Train model
    try:
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = safe_r2_score(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = safe_r2_score(y_test, y_test_pred)
        
        cv_results = {
            'train_mse': [train_mse],
            'train_rmse': [train_rmse],
            'train_r2': [train_r2],
            'test_mse': [test_mse],
            'test_rmse': [test_rmse],
            'test_r2': [test_r2],
            'avg_train_mse': train_mse,
            'avg_train_rmse': train_rmse,
            'avg_train_r2': train_r2,
            'avg_test_mse': test_mse,
            'avg_test_rmse': test_rmse,
            'avg_test_r2': test_r2
        }
        
        # Feature importance
        feature_importance = calculate_feature_importance(model, model_type, X.columns)
        
        return {
            'model': model,
            'scaler': scaler
        }, cv_results, feature_importance
        
    except Exception as e:
        logger.error(f"Error in simple split training: {str(e)}")
        return None, {'error': str(e)}, {}

def create_model_by_type(model_type, params=None):
    """
    Create a model instance based on the specified type and parameters.
    
    Parameters:
        model_type: str - Type of model to create
        params: dict - Model parameters
        
    Returns:
        Model instance
    """
    if params is None:
        params = {}
        
    if model_type == 'mlp':
        mlp_params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'max_iter': 2000,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 20,
            'tol': 1e-4,
            'random_state': 42
        }
        mlp_params.update(params)
        return MLPRegressor(**mlp_params)
    elif model_type == 'gbm':
        gbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        gbm_params.update(params)
        return GradientBoostingRegressor(**gbm_params)
    elif model_type == 'elasticnet':
        from sklearn.linear_model import ElasticNet
        elasticnet_params = {
            'alpha': 0.5,
            'l1_ratio': 0.5,
            'max_iter': 500,
            'random_state': 42
        }
        elasticnet_params.update(params)
        return ElasticNet(**elasticnet_params)
    else:
        # Default to ridge regression as most stable option
        from sklearn.linear_model import Ridge
        ridge_params = {'alpha': 1.0, 'random_state': 42}
        ridge_params.update(params)
        return Ridge(**ridge_params)

def calculate_feature_importance(model, model_type, feature_names):
    """
    Calculate feature importance based on model type.
    
    Parameters:
        model: Trained model
        model_type: str - Type of model
        feature_names: list - List of feature names
        
    Returns:
        dict: Feature importance scores
    """
    feature_importance = {}
    
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            for i, col in enumerate(feature_names):
                feature_importance[col] = importances[i]
        elif model_type == 'mlp' and hasattr(model, 'coefs_'):
            # For MLPRegressor, use weights as approximate importance
            weights = np.abs(model.coefs_[0])
            importances = np.sum(weights, axis=1) / np.sum(weights)
            for i, col in enumerate(feature_names):
                feature_importance[col] = importances[i]
        else:
            # Default to equal importance if no method available
            importance_value = 1.0 / len(feature_names)
            for col in feature_names:
                feature_importance[col] = importance_value
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {e}")
        # Default to equal importance
        importance_value = 1.0 / len(feature_names)
        for col in feature_names:
            feature_importance[col] = importance_value
    
    # Sort by importance
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

def handle_small_dataset_training(x, y, model_type, params=None):
    """
    Handle training for very small datasets with special considerations.
    
    Parameters:
        x: DataFrame - Feature data
        y: Series - Target data
        model_type: str - Model type
        params: dict - Model parameters
        
    Returns:
        dict: Training results
    """
    logger.warning(f"Very small dataset ({len(x)} points). Using simplified training approach.")
    
    try:
        # Use a simple 70/30 train/test split instead of cross-validation
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use a simpler model configuration for small datasets
        if model_type == 'mlp':
            model = MLPRegressor(
                hidden_layer_sizes=(16, 8),  # Smaller network
                activation='relu',
                solver='adam',
                alpha=0.01,  # Stronger regularization
                max_iter=2000,
                tol=1e-4,
                random_state=42
            )
        else:
            model = create_model_by_type(model_type, params)
        
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Create full predictions
        X_all_scaled = scaler.transform(x)
        all_predictions = model.predict(X_all_scaled)
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': pd.Series(all_predictions, index=x.index),
            'metrics': {
                'train': {'rmse': train_rmse, 'r2': train_r2},
                'test': {'rmse': test_rmse, 'r2': test_r2}
            },
            'status': 'Success - Simple Training (Small Dataset)'
        }
        
    except Exception as e:
        logger.error(f"Error in simple training for small dataset: {str(e)}")
        return {
            'status': 'Failed - Small Dataset Training Error',
            'error': str(e)
        }

def generate_model_predictions(model, scaler, x, y, country, tenor_name, model_type):
    """
    Generate predictions and store results for fitted values consolidation.
    
    Parameters:
        model: Trained model
        scaler: Data scaler
        x: DataFrame - Feature data
        y: Series - Target data
        country: str - Country name
        tenor_name: str - Tenor name
        model_type: str - Model type
        
    Returns:
        pandas.Series: Model predictions
    """
    try:
        if scaler is not None:
            x_scaled = scaler.transform(x)
            all_predictions = model.predict(x_scaled)
        else:
            all_predictions = model.predict(x)
        
        # Store predictions
        predictions = pd.Series(all_predictions, index=x.index) if isinstance(all_predictions, np.ndarray) else all_predictions
        
        # Store data for consolidated fitted values
        fitted_df = pd.DataFrame({
            'date': x.index,
            'actual': y,
            'predicted': predictions,
            'residual': y - predictions,
            'country': country,
            'tenor': tenor_name,
            'model_type': model_type
        })
        
        # Store in global all_fitted_values
        key = f"{country}_{tenor_name}_{model_type}"
        all_fitted_values[key] = fitted_df
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return None

def safe_r2_score(y_true, y_pred, fallback_value=0.0):
    """
    Safe implementation of R² score that handles edge cases.
    """
    if len(y_true) < 2:
        logger.warning("R² score calculation attempted with less than two samples")
        return fallback_value
    
    # Check if all values are identical
    if np.all(y_true == y_true[0]):
        logger.warning("R² score undefined - all true values are identical")
        return fallback_value
    
    try:
        return r2_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Error calculating R² score: {e}")
        return fallback_value
    

def train_evaluate_model(country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, iip_gdp,
                        act_track, risk_rating, historical_forecasts, unemployment_rate, model_type, 
                        predicted_yields=None, use_advanced_training=True, compare_models=False, 
                        optimize_params=True, output_dir=None, use_yield_curve_models=True):
    """
    Train and evaluate a model for a specific country and yield tenor.
    Enhanced to support multiple model types, Nelson-Siegel, and advanced ensemble techniques.
    
    Parameters:
        bbg_client: Bloomberg client instance
        mb_client: Macrobond client instance
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        country_code_mapping: dict - Mapping from country names to country codes
        tenor_data: DataFrame - Yield data for the specified tenor
        pol_rat: DataFrame - Policy rate data
        cpi_inf: DataFrame - Inflation data
        act_track: DataFrame - Economic activity tracker data
        risk_rating: DataFrame - Risk rating data
        historical_forecasts: dict - Historical forecasts from forecast_generator
        unemployment_rate: DataFrame - Unemployment rate data
        predicted_yields: dict - Dictionary containing predicted yields for shorter tenors
        model_type: str - Type of model to train
        use_advanced_training: bool - Whether to use advanced training methods
        compare_models: bool - Whether to compare against other model types
        optimize_params: bool - Whether to perform hyperparameter optimization
        output_dir: str - Directory to save outputs
        use_yield_curve_models: bool - Whether to use yield curve specific models
        
    Returns:
        dict: Model evaluation results and diagnostics
    """
    logger.info(f"\n=== TRAINING {model_type.upper()} MODEL FOR {country} - {tenor_name} ===")
    
    # Initialize results dictionary
    results = initialize_results_dict(country, tenor_name, model_type)
    
    # Set up output directory
    if output_dir is None:
        try:
            output_dir = os.path.join(config.RESULTS_DIR, country, tenor_name)
        except Exception as e:
            logger.warning(f"Error setting output directory: {str(e)}. Using current directory.")
            output_dir = os.path.join('.', country, tenor_name)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Error creating output directory: {str(e)}. Output may not be saved correctly.")
    
    # Get country code
    if country not in country_code_mapping:
        logger.error(f"Country {country} not found in country_code_mapping")
        results['status'] = 'Failed - Invalid Country'
        results['error'] = f"Country {country} not found in mapping"
        return results
    
    country_code = country_code_mapping[country]
    
    try:
        # Prepare data with enhanced feature engineering
        x, y, feature_details = prepare_model_training_data(
            country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, 
            iip_gdp, act_track, risk_rating, historical_forecasts, unemployment_rate, 
            predicted_yields, use_advanced_training
        )
        
        if x is None or y is None:
            results['status'] = 'Failed - Data Preparation'
            results['error'] = 'Data preparation failed'
            return results
        
        # Store data summary in results
        results['data_summary'] = feature_details
        
        # Validate data for training
        validation = validate_model_training_data(x, y, country, tenor_name)
        if not validation['valid']:
            results['status'] = 'Failed - Insufficient Data'
            results['error'] = validation['error']
            return results
        
        # Handle very small datasets
        if len(x) < 30:
            small_dataset_results = handle_small_dataset_training(x, y, model_type)
            results.update(small_dataset_results)
            if results['status'].startswith('Success'):
                # Generate and store predictions
                predictions = generate_model_predictions(
                    results['model'], results['scaler'], x, y, country, tenor_name, model_type
                )
                results['predictions'] = predictions
            return results
        
        logger.info(f"Training {model_type} model with {len(x)} data points and {x.shape[1]} features")
        results['status'] = 'Data Prepared'
        
        # Execute training based on approach
        if use_advanced_training:
            results = execute_advanced_training(
                results, x, y, model_type, optimize_params, compare_models, 
                use_yield_curve_models, predicted_yields, country, tenor_name
            )
        else:
            results = execute_traditional_training(results, x, y, model_type)
        
        # Generate predictions if model training was successful
        if results['status'].startswith('Success') and 'model' in results:
            predictions = generate_model_predictions(
                results['model'], results['scaler'], x, y, country, tenor_name, model_type
            )
            results['predictions'] = predictions
        
        # Save results to files
        save_model_results(results, output_dir, country, tenor_name, model_type, x, y)
        
        logger.info(f"Training complete for {country} - {tenor_name} using {model_type} model")
        
    except Exception as e:
        logger.error(f"Error in train_evaluate_model: {str(e)}")
        logger.error(traceback.format_exc())
        results['status'] = 'Failed - Exception'
        results['error'] = str(e)
    
    return results

def initialize_results_dict(country, tenor_name, model_type):
    """Initialize the results dictionary with default values."""
    return {
        'country': country,
        'tenor': tenor_name,
        'model_type': model_type,
        'status': 'Not Started',
        'metrics': {
            'train': {},
            'test': {},
            'validation': {}
        },
        'feature_importance': None,
        'data_summary': None,
        'error': None,
        'predictions': None,
        'model': None,
        'scaler': None,
        'economic_validity': {}
    }

def execute_advanced_training(results, x, y, model_type, optimize_params, compare_models, 
                            use_yield_curve_models, predicted_yields, country, tenor_name):
    """
    Execute advanced training approach with cross-validation and optimization.
    """
    logger.info(f"Using advanced training approach with cross-validation for {model_type}")
    
    # Handle Nelson-Siegel model if requested
    if use_yield_curve_models and model_type == 'nelson_siegel':
        return handle_nelson_siegel_training(results, x, y, predicted_yields, country, tenor_name)
    
    # Define base model parameters
    model_params = get_base_model_params(model_type)
    
    # Optimize hyperparameters if requested
    if optimize_params:
        model_params = optimize_model_hyperparameters(x, y, model_type, model_params)
    
    # Train with cross-validation
    try:
        model_package, cv_results, feature_imp = train_model_with_cross_validation(
            x, y, model_type=model_type, n_splits=3, params=model_params
        )
        
        if model_package is None:
            logger.error(f"Cross-validation training failed for {country} - {tenor_name} - {model_type}")
            results['status'] = f'Failed - {model_type} Training Error'
            results['error'] = 'Cross-validation training failed to produce a valid model'
            return results
        
        # Store model and metrics
        results['model'] = model_package['model']
        results['scaler'] = model_package['scaler']
        results['metrics'] = cv_results
        results['feature_importance'] = feature_imp
        results['status'] = f'Success - Advanced {model_type.upper()} Training'
        
    except Exception as e:
        logger.error(f"Error in cross-validation training: {str(e)}")
        results['status'] = f'Failed - {model_type} Training Exception'
        results['error'] = str(e)
        return results
    
    # Handle model comparison if requested
    if compare_models:
        results = handle_model_comparison(results, x, y, model_type, country, tenor_name)
    
    return results

def execute_traditional_training(results, x, y, model_type):
    """
    Execute traditional training approach with chronological train/test split.
    """
    logger.info("Using traditional training approach with time-based split")
    
    # Use chronological train/test split
    train_size = int(len(x) * getattr(config, 'DEFAULT_TRAIN_TEST_SPLIT', 0.8))
    
    x_train = x.iloc[:train_size]
    y_train = y.iloc[:train_size]
    x_test = x.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    logger.info(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")
    
    # Scale the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test) if not x_test.empty else None
    
    # Create and train model
    try:
        model = create_model_by_type(model_type)
        model.fit(x_train_scaled, y_train)
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        results['status'] = 'Failed - Model Training'
        results['error'] = str(e)
        return results
    
    # Calculate training metrics
    try:
        y_train_pred = model.predict(x_train_scaled)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = safe_r2_score(y_train, y_train_pred)
        
        logger.info(f"Training metrics: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
        
        # Store training metrics
        results['metrics']['train'] = {
            'mse': train_mse,
            'rmse': train_rmse,
            'r2': train_r2
        }
        
    except Exception as e:
        logger.error(f"Error calculating training metrics: {str(e)}")
        results['metrics']['train'] = {'error': str(e)}
    
    # Calculate test metrics if test data is available
    if not x_test.empty and x_test_scaled is not None:
        try:
            y_test_pred = model.predict(x_test_scaled)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = safe_r2_score(y_test, y_test_pred)
            
            logger.info(f"Test metrics: RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
            
            results['metrics']['test'] = {
                'mse': test_mse,
                'rmse': test_rmse,
                'r2': test_r2
            }
            
        except Exception as e:
            logger.error(f"Error calculating test metrics: {str(e)}")
            results['metrics']['test'] = {'error': str(e)}
    
    # Store model and scaler
    results['model'] = model
    results['scaler'] = scaler
    results['status'] = f'Success - Traditional {model_type.upper()} Training'
    
    return results

def get_base_model_params(model_type):
    """
    Get base model parameters for different model types.
    """
    if model_type == 'mlp' or model_type == 'enhanced_mlp':
        return {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42
        }
    elif model_type == 'gbm':
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
    elif model_type == 'elasticnet':
        return {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'max_iter': 1000,
            'random_state': 42
        }
    elif model_type == 'gp':
        return {
            'length_scale': 1.0,
            'alpha': 0.1,
            'n_restarts_optimizer': 5,
            'normalize_y': True
        }
    else:
        return {}

def optimize_model_hyperparameters(x, y, model_type, base_params):
    """
    Optimize hyperparameters for the specified model type.
    """
    logger.info(f"Performing hyperparameter optimization for {model_type}")
    
    # Define parameter grids for different model types
    param_grids = {
        'mlp': {
            'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        },
        'gbm': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 2, 5]
        },
        'elasticnet': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        'gp': {
            'length_scale': [0.1, 1.0, 10.0],
            'alpha': [0.01, 0.1, 1.0]
        }
    }
    
    param_grid = param_grids.get(model_type, {})
    
    try:
        # Optimize parameters
        best_model, best_params, cv_results = optimize_hyperparameters(
            x, y, model_type=model_type, param_grid=param_grid, cv_method='grid'
        )
        
        # Update parameters with optimized values
        if best_params:
            base_params.update(best_params)
            logger.info(f"Optimized parameters: {best_params}")
        
        return base_params
        
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        logger.warning("Continuing with default parameters")
        return base_params

def handle_nelson_siegel_training(results, x, y, predicted_yields, country, tenor_name):
    """
    Handle Nelson-Siegel model training for yield curve modeling.
    """
    # Placeholder for Nelson-Siegel implementation
    # This would require additional implementation based on your specific needs
    logger.warning("Nelson-Siegel model not fully implemented")
    results['status'] = 'Failed - Nelson-Siegel Not Implemented'
    results['error'] = 'Nelson-Siegel model requires additional implementation'
    return results

def handle_model_comparison(results, x, y, model_type, country, tenor_name):
    """
    Handle comparison with other model types and ensemble creation.
    """
    logger.info("Training benchmark models for comparison")
    
    try:
        # Define models to compare with
        models_to_train = ['mlp', 'gbm', 'xgboost', 'elasticnet']
        
        # Add GP model if data size is reasonable (GP can be slow with large datasets)
        if len(x) < 1000:
            models_to_train.concat('gp')
        
        # Make sure we're not duplicating the current model
        if model_type in models_to_train:
            models_to_train.remove(model_type)
        
        benchmark_results = train_benchmark_models(
            x, y, models_to_train=models_to_train,
            country=country, tenor=tenor_name
        )
        
        if benchmark_results:
            # Store benchmark comparison
            results['benchmark_comparison'] = benchmark_results.get('comparison')
            results['best_model_type'] = benchmark_results.get('best_model')
            
            # Create ensemble if enough models succeeded
            model_success_count = sum(1 for key in benchmark_results 
                                   if key not in ['comparison', 'best_model'] 
                                   and benchmark_results[key].get('model_package') is not None)
            
            if model_success_count >= 2:
                results = create_ensemble_model(results, benchmark_results, x, y, model_type)
    
    except Exception as e:
        logger.error(f"Error in benchmark model training: {str(e)}")
    
    return results

def create_ensemble_model(results, benchmark_results, x, y, model_type):
    """
    Create an ensemble model from successful benchmark models.
    """
    try:
        logger.info("Building ensemble model from successful benchmark models")
        
        # Get individual models
        models = {}
        for model_type_key, result in benchmark_results.items():
            if model_type_key not in ['comparison', 'best_model']:
                model_package = result.get('model_package')
                if model_package and 'model' in model_package:
                    models[model_type_key] = model_package['model']
        
        # Add current model
        if results['model'] is not None:
            models[model_type] = results['model']
        
        if len(models) >= 2:
            # Build ensemble model
            ensemble_model, ensemble_metrics = build_ensemble_model(
                models, x, y, weighted_by_performance=True
            )
            
            # Store ensemble results
            results['ensemble_model'] = ensemble_model
            results['ensemble_metrics'] = ensemble_metrics
            
            logger.info(f"Ensemble model performance: RMSE={ensemble_metrics.get('rmse', 'N/A')}, R²={ensemble_metrics.get('r2', 'N/A')}")
            
            # Check if ensemble performs better than single model
            if ('rmse' in ensemble_metrics and 'avg_test_rmse' in results['metrics'] and 
                ensemble_metrics['rmse'] < results['metrics']['avg_test_rmse']):
                logger.info("Ensemble model outperforms single model - using ensemble as primary model")
                results['model'] = ensemble_model
                results['metrics']['ensemble'] = ensemble_metrics
                results['status'] = 'Success - Ensemble Model'
        else:
            logger.warning(f"Not enough successful models to build ensemble (need at least 2, got {len(models)})")
    
    except Exception as e:
        logger.error(f"Error building ensemble model: {str(e)}")
    
    return results

def save_model_results(results, output_dir, country, tenor_name, model_type, x, y):
    """
    Save model predictions and summary to files.
    """
    try:
        # Save model predictions to CSV
        if 'predictions' in results and results['predictions'] is not None:
            predictions_df = pd.DataFrame({
                'date': x.index,
                'actual': y,
                'predicted': results['predictions'],
                'residual': y - results['predictions'],
                'model_type': model_type
            })
            
            predictions_file = os.path.join(output_dir, f"{country}_{tenor_name}_{model_type}_predictions.csv")
            predictions_df.to_csv(predictions_file)
            logger.info(f"Saved predictions to {predictions_file}")
    
    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {str(e)}")
    
    try:
        # Create a summary file with key metrics
        summary = create_model_summary(results, country, tenor_name, model_type, x)
        
        summary_file = os.path.join(output_dir, f"{country}_{tenor_name}_{model_type}_summary.json")
        
        # Handle non-serializable objects
        def json_serializer(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, default=json_serializer)
        
    except Exception as e:
        logger.error(f"Error creating summary file: {str(e)}")

def create_model_summary(results, country, tenor_name, model_type, x):
    """
    Create a summary dictionary with key model information.
    """
    summary = {
        'country': country,
        'tenor': tenor_name,
        'model_type': model_type,
        'data_points': len(x),
        'features': len(x.columns),
        'date_range': f"{x.index.min().strftime('%Y-%m-%d')} to {x.index.max().strftime('%Y-%m-%d')}"
    }
    
    # Add metrics
    if 'train' in results['metrics']:
        summary['train_rmse'] = results['metrics']['train'].get('rmse')
        summary['train_r2'] = results['metrics']['train'].get('r2')
    
    if 'test' in results['metrics']:
        summary['test_rmse'] = results['metrics']['test'].get('rmse')
        summary['test_r2'] = results['metrics']['test'].get('r2')
    
    if 'ensemble' in results['metrics']:
        summary['ensemble_rmse'] = results['metrics']['ensemble'].get('rmse')
        summary['ensemble_r2'] = results['metrics']['ensemble'].get('r2')
        summary['used_ensemble'] = True
    
    return summary


def visualize_model_predictions(model, x_train, y_train, x_test, y_test, scaler, country, tenor_name, model_type):
    """
    Create visualizations of model performance including:
    1. Training data fit
    2. Test data predictions
    3. Combined visualization
    
    Parameters:
        model: Trained model
        x_train, y_train: Training data
        x_test, y_test: Test data
        scaler: The StandardScaler used to scale the input data
        country: Country name
        tenor_name: Yield tenor name
        model_type: Type of model used
    """
    logger.info(f"Creating visualizations for {model_type.upper()} model for {country} - {tenor_name}")
    
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # 1. Training data fit
        if not x_train.empty and len(y_train) > 0:
            try:
                x_train_scaled = scaler.transform(x_train)
                y_train_pred = model.predict(x_train_scaled)
                
                axes[0].plot(x_train.index, y_train, 'b-', label='Actual', linewidth=2)
                axes[0].plot(x_train.index, y_train_pred, 'r--', label='Model fit', linewidth=2)
                axes[0].set_title(f"{country} - {tenor_name}: {model_type.upper()} Training Data Fit", fontsize=14)
                axes[0].set_xlabel('Date', fontsize=12)
                axes[0].set_ylabel('Yield (%)', fontsize=12)
                axes[0].legend(fontsize=12)
                axes[0].grid(True)
                
                # Calculate and display RMSE
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_r2 = safe_r2_score(y_train, y_train_pred)
                
                axes[0].annotate(f"RMSE: {train_rmse:.4f}\nR²: {train_r2:.4f}", 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                            fontsize=12)
                
                # Format x-axis to show years
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axes[0].xaxis.set_major_locator(mdates.YearLocator(5))
                axes[0].tick_params(axis='x', rotation=45)
                
            except Exception as e:
                logger.error(f"Error generating training visualization: {e}")
                axes[0].text(0.5, 0.5, f"Training visualization error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0].transAxes, fontsize=12)
        else:
            axes[0].text(0.5, 0.5, "No training data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[0].transAxes, fontsize=14)
        
        # 2. Test data predictions
        if not x_test.empty and len(y_test) > 0:
            try:
                x_test_scaled = scaler.transform(x_test)
                y_test_pred = model.predict(x_test_scaled)
                
                axes[1].plot(x_test.index, y_test, 'b-', label='Actual', linewidth=2)
                axes[1].plot(x_test.index, y_test_pred, 'r--', label='Predicted', linewidth=2)
                axes[1].set_title(f"{country} - {tenor_name}: {model_type.upper()} Test Data Predictions", fontsize=14)
                axes[1].set_xlabel('Date', fontsize=12)
                axes[1].set_ylabel('Yield (%)', fontsize=12)
                axes[1].legend(fontsize=12)
                axes[1].grid(True)
                
                # Calculate and display RMSE
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_r2 = safe_r2_score(y_test, y_test_pred)
                
                axes[1].annotate(f"RMSE: {test_rmse:.4f}\nR²: {test_r2:.4f}", 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                            fontsize=12)
                
                # Format x-axis to show years
                axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
                axes[1].tick_params(axis='x', rotation=45)
                
            except Exception as e:
                logger.error(f"Error generating test visualization: {e}")
                axes[1].text(0.5, 0.5, f"Test visualization error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes, fontsize=12)
        else:
            axes[1].text(0.5, 0.5, "No test data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes, fontsize=14)
        
        # 3. Combined training and test data visualization
        try:
            ax = axes[2]

            # Plot training data
            if not x_train.empty and len(y_train) > 0:
                x_train_scaled = scaler.transform(x_train)
                y_train_pred = model.predict(x_train_scaled)

                ax.plot(y_train.index, y_train, 'b-', label='Actual (Training)', linewidth=1.5)
                ax.plot(y_train.index, y_train_pred, 'r--', label='Predicted (Training)', linewidth=1.5)

            # Plot test data
            if not x_test.empty and len(y_test) > 0:
                x_test_scaled = scaler.transform(x_test)
                y_test_pred = model.predict(x_test_scaled)

                ax.plot(y_test.index, y_test, 'g-', label='Actual (Test)', linewidth=1.5)
                ax.plot(y_test.index, y_test_pred, 'm--', label='Predicted (Test)', linewidth=1.5)

            ax.set_title(f"{country} - {tenor_name}: Combined {model_type.upper()} Model Performance", fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Yield (%)', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)

            # Format x-axis to show years
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(3))
            ax.tick_params(axis='x', rotation=45)

        except Exception as e:
            logger.error(f"Error generating combined visualization: {e}")
            axes[2].text(0.5, 0.5, f"Combined visualization error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes, fontsize=12)

        plt.tight_layout()

        # Save to file
        try:
            filename = f"{country}_{tenor_name}_{model_type}_model_analysis.png"
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            plt.savefig(filename)
            logger.info(f"Model visualization saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving visualization file for {country}_{tenor_name}: {e}")
            plt.savefig(f"model_analysis_{country}_{tenor_name.replace('yld_', '')}_{model_type}.png")

        plt.close(fig)

    except Exception as e:
        logger.error(f"Error visualizing model for {country} - {tenor_name}: {e}")
        logger.error(traceback.format_exc())

def generate_feature_importance_report(mlp_model, x_data, y_data, scaler, country, tenor_name):
    """
    Analyze and report the importance of features in the trained MLP model.
    Uses multiple methods including permutation importance, correlation analysis,
    and partial dependence plots.
    
    Parameters:
        mlp_model: Trained MLPRegressor model
        x_data: DataFrame - Input features
        y_data: Series - Target values
        scaler: The StandardScaler used to scale the input data
        country: str - Country name
        tenor_name: str - Yield tenor name
        
    Returns:
        dict: Feature importance analysis results
    """
    logger.info(f"Generating feature importance analysis for {country} - {tenor_name}")
    
    if mlp_model is None or x_data.empty or len(y_data) == 0:
        logger.error("Cannot generate feature importance report without a trained model and data")
        return None
    
    # Store results in a dictionary
    importance_results = {
        'country': country,
        'tenor': tenor_name,
        'features': [],
        'correlations': {},
        'permutation_importance': {},
        'summary': {}
    }
    
    # 1. Calculate simple correlations
    logger.info("Calculating feature correlations...")
    correlations = {}
    for column in x_data.columns:
        correlation = x_data[column].corr(y_data)
        correlations[column] = correlation
    
    # Sort by absolute correlation values
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    importance_results['correlations'] = {k: round(v, 4) for k, v in sorted_correlations}
    
    # 2. Calculate permutation importance
    try:
        logger.info("Calculating permutation importance...")
        from sklearn.inspection import permutation_importance
        
        # Scale the data
        x_scaled = scaler.transform(x_data)
        
        # Calculate permutation importance
        r = permutation_importance(mlp_model, x_scaled, y_data, 
                                n_repeats=10, 
                                random_state=42)
        
        # Store permutation importance scores
        perm_importance = {}
        for i in range(len(x_data.columns)):
            feature_name = x_data.columns[i]
            importance_score = r.importances_mean[i]
            importance_std = r.importances_std[i]
            perm_importance[feature_name] = {
                'score': importance_score,
                'std': importance_std
            }
        
        # Sort by importance score
        sorted_importance = sorted(perm_importance.items(), 
                                 key=lambda x: abs(x[1]['score']), 
                                 reverse=True)
        
        importance_results['permutation_importance'] = {
            k: {'score': round(v['score'], 4), 'std': round(v['std'], 4)} 
            for k, v in sorted_importance
        }
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {e}")
        importance_results['permutation_importance'] = {'error': str(e)}
    
    # 3. Analyze feature weights for MLPRegressor
    try:
        logger.info("Analyzing neural network weights...")
        
        # Get weights from first layer
        first_layer_weights = mlp_model.coefs_[0]  # Shape: [n_features, n_neurons]
        
        # Calculate average absolute weight for each feature
        avg_weights = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Normalize weights to sum to 1
        if np.sum(avg_weights) > 0:
            normalized_weights = avg_weights / np.sum(avg_weights)
        else:
            normalized_weights = np.ones_like(avg_weights) / len(avg_weights)
        
        weight_importance = {}
        for i, feature in enumerate(x_data.columns):
            weight_importance[feature] = normalized_weights[i]
        
        # Sort by weight importance
        sorted_weights = sorted(weight_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        importance_results['weight_importance'] = {k: round(float(v), 4) for k, v in sorted_weights}
    except Exception as e:
        logger.error(f"Error analyzing neural network weights: {e}")
        importance_results['weight_importance'] = {'error': str(e)}
    
    # 4. Create consensus ranking
    importance_results['consensus_ranking'] = create_consensus_ranking(
        sorted_correlations, importance_results
    )
    
    # 5. Create visualizations
    create_importance_visualizations(importance_results, country, tenor_name, x_data, y_data)
    
    # 6. Export to CSV
    try:
        importance_df = pd.DataFrame(importance_results['consensus_ranking'])
        importance_df.to_csv(f"{country}_{tenor_name}_feature_importance.csv", index=False)
        logger.info(f"Feature importance data saved to {country}_{tenor_name}_feature_importance.csv")
    except Exception as e:
        logger.error(f"Error exporting feature importance to CSV: {e}")
    
    return importance_results

def create_consensus_ranking(sorted_correlations, importance_results):
    """
    Create a consensus feature ranking from different importance methods.
    """
    # Get all features
    features = [item[0] for item in sorted_correlations]
    
    # Initialize feature scores
    feature_scores = {feature: {'score': 0, 'methods': {}} for feature in features}
    
    # Add correlation scores
    for feature, corr in sorted_correlations:
        feature_scores[feature]['methods']['correlation'] = abs(corr)
    
    # Add permutation importance scores
    if 'error' not in importance_results['permutation_importance']:
        for feature, data in importance_results['permutation_importance'].items():
            if feature in feature_scores:
                feature_scores[feature]['methods']['permutation'] = abs(data['score'])
    
    # Add weight importance scores
    if 'error' not in importance_results.get('weight_importance', {'error': True}):
        for feature, weight in importance_results['weight_importance'].items():
            if feature in feature_scores:
                feature_scores[feature]['methods']['weight'] = abs(weight)
    
    # Calculate consensus scores (average of available methods)
    for feature in feature_scores:
        method_scores = list(feature_scores[feature]['methods'].values())
        if method_scores:
            feature_scores[feature]['score'] = sum(method_scores) / len(method_scores)
        else:
            feature_scores[feature]['score'] = 0
    
    # Sort features by consensus score
    consensus_ranking = sorted(feature_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Create consensus ranking list
    ranking_list = []
    for feature, data in consensus_ranking:
        methods = data['methods']
        ranking_list.append({
            'feature': feature,
            'consensus_score': round(data['score'], 4),
            'correlation': round(methods.get('correlation', 0), 4),
            'permutation': round(methods.get('permutation', 0), 4),
            'weight': round(methods.get('weight', 0), 4)
        })
    
    return ranking_list

def create_importance_visualizations(importance_results, country, tenor_name, x_data, y_data):
    """
    Create visualizations for feature importance analysis.
    """
    try:
        # Bar chart of feature importance
        plt.figure(figsize=(10, 6))
        
        features = [item['feature'] for item in importance_results['consensus_ranking']]
        scores = [item['consensus_score'] for item in importance_results['consensus_ranking']]
        
        plt.bar(features, scores, color='skyblue')
        plt.title(f"{country} - {tenor_name}: Feature Importance", fontsize=14)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{country}_{tenor_name}_feature_importance.png")
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix between all features and target
        all_data = x_data.copy()
        all_data[tenor_name] = y_data
        
        corr_matrix = all_data.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"{country} - {tenor_name}: Feature Correlation Heatmap", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{country}_{tenor_name}_correlation_heatmap.png")
        plt.close()
        
        logger.info(f"Feature importance visualizations saved for {country} - {tenor_name}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def fetch_all_data_sources(bbg_client, mb_client):
    """
    Fetch all required data sources from Bloomberg and Macrobond.
    
    Parameters:
        bbg_client: Bloomberg client instance
        mb_client: Macrobond client instance
    
    Returns:
        dict: Dictionary containing all data sources
    """
    logger.info("Fetching all data sources")
    
    data_sources = {}
    
    try:
        # Define date range for data fetching
        dt_from = config.DEFAULT_DATE_FROM
        dt_to = config.DEFAULT_DATE_TO
        
        # 1. Fetch yield data for different tenors
        yield_data = fetch_yield_data(bbg_client, dt_from, dt_to)
        data_sources['yield_data'] = yield_data
        
        # 2. Fetch policy rate data
        pol_rat = fetch_policy_rate_data(bbg_client, dt_from, dt_to)
        data_sources['policy_rates'] = pol_rat
        
        # 3. Fetch economic activity tracker data
        act_track = fetch_activity_data(bbg_client, dt_from, dt_to)
        data_sources['activity'] = act_track
        
        # 4. Fetch inflation data
        cpi_inf = fetch_inflation_data(mb_client)
        data_sources['inflation'] = cpi_inf
        
        # 5. Fetch unemployment data
        unemployment_rate = fetch_unemployment_data(mb_client)
        data_sources['unemployment_rate'] = unemployment_rate
        
        # 6. Fetch IIP/GDP data
        iip_gdp = fetch_iip_gdp_data(mb_client)
        data_sources['iip_gdp'] = iip_gdp
        
        # 7. Fetch credit rating data
        risk_rating = fetch_risk_rating_data(mb_client)
        data_sources['risk_rating'] = risk_rating
        
        logger.info("Successfully fetched all data sources")
        
    except Exception as e:
        logger.error(f"Error in fetch_all_data_sources: {e}")
        raise
    
    return data_sources

def fetch_yield_data(bbg_client, dt_from, dt_to):
    """Fetch yield data for different tenors."""
    yield_data = {}
    
    # 2-year bond yields
    yld_2yr = get_bloomberg_date(
        bbg_client,
        list(config.bond_yield_tickers['2yr'].keys()), 
        dt_from, dt_to, 
        periodicity=config.BLOOMBERG_DAILY_PERIODICITY
    )
    yld_2yr = yld_2yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_2yr'])
    yld_2yr_ann = yld_2yr.resample('ME').mean()
    yield_data['yld_2yr'] = yld_2yr
    yield_data['yld_2yr_ann'] = yld_2yr_ann
    
    # 5-year bond yields
    yld_5yr = get_bloomberg_date(
        bbg_client,
        list(config.bond_yield_tickers['5yr'].keys()), 
        dt_from, dt_to, 
        periodicity=config.BLOOMBERG_DAILY_PERIODICITY
    )
    yld_5yr = yld_5yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_5yr'])
    yld_5yr_ann = yld_5yr.resample('ME').mean()
    yield_data['yld_5yr'] = yld_5yr
    yield_data['yld_5yr_ann'] = yld_5yr_ann
    
    # 10-year bond yields
    yld_10yr = get_bloomberg_date(
        bbg_client,
        list(config.bond_yield_tickers['10yr'].keys()), 
        dt_from, dt_to, 
        periodicity=config.BLOOMBERG_DAILY_PERIODICITY
    )
    yld_10yr = yld_10yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_10yr'])
    yld_10yr_ann = yld_10yr.resample('ME').mean()
    yield_data['yld_10yr'] = yld_10yr
    yield_data['yld_10yr_ann'] = yld_10yr_ann
    
    # 30-year bond yields
    yld_30yr = get_bloomberg_date(
        bbg_client,
        list(config.bond_yield_tickers['30yr'].keys()), 
        dt_from, dt_to, 
        periodicity=config.BLOOMBERG_DAILY_PERIODICITY
    )
    yld_30yr = yld_30yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_30yr'])
    yld_30yr_ann = yld_30yr.resample('ME').mean()
    yield_data['yld_30yr'] = yld_30yr
    yield_data['yld_30yr_ann'] = yld_30yr_ann
    
    return yield_data

def fetch_policy_rate_data(bbg_client, dt_from, dt_to):
    """Fetch policy rate data."""
    pol_rat = get_bloomberg_date(
        bbg_client,
        list(config.pol_rat_tickers.keys()), 
        dt_from, dt_to, 
        periodicity=config.BLOOMBERG_DAILY_PERIODICITY
    )
    pol_rat = pol_rat.rename(columns=config.COLUMN_MAPPINGS['policy_rates'])
    pol_rat = pol_rat.resample('ME').mean()
    return pol_rat

def fetch_activity_data(bbg_client, dt_from, dt_to):
    """Fetch economic activity tracker data."""
    act_track = get_bloomberg_date(
        bbg_client,
        list(config.act_track_tickers.keys()), 
        dt_from, dt_to, 
        periodicity=config.BLOOMBERG_MONTHLY_PERIODICITY
    )
    act_track = act_track.rename(columns=config.COLUMN_MAPPINGS['activity'])
    if not act_track.empty:
        act_track.index = act_track.index.to_period("M").to_timestamp("M")
        act_track = act_track.resample('ME').first().ffill()
        act_track.index = pd.DatetimeIndex(act_track.index.strftime('%Y-%m-%d'))
    return act_track

def fetch_inflation_data(mb_client):
    """Fetch inflation data from Macrobond."""
    try:
        if mb_client is None:
            logger.error("Macrobond client is None")
            return pd.DataFrame()
        
        cpi_inf = mb_client.FetchSeries(list(config.cpi_inf_tickers.keys()))
        cpi_inf = cpi_inf.rename(columns=config.COLUMN_MAPPINGS['cpi_inf'])
        cpi_inf = cpi_inf.resample('ME').mean()
        cpi_inf = cpi_inf.pct_change(periods=12) * 100  # Convert to year-over-year percentage change
        return cpi_inf
    except Exception as e:
        logger.error(f"Error fetching inflation data: {e}")
        return pd.DataFrame()

def fetch_unemployment_data(mb_client):
    """Fetch unemployment data from Macrobond."""
    try:
        if mb_client is None:
            logger.error("Macrobond client is None")
            return pd.DataFrame()
        
        unemployment_rate = mb_client.FetchSeries(list(config.unemployment_tickers.keys()))
        unemployment_rate = unemployment_rate.rename(columns=config.COLUMN_MAPPINGS['unemployment_rate'])
        unemployment_rate = unemployment_rate.resample('ME').mean()
        return unemployment_rate
    except Exception as e:
        logger.error(f"Error fetching unemployment data: {e}")
        return pd.DataFrame()

def fetch_iip_gdp_data(mb_client):
    """Fetch IIP/GDP data from Macrobond."""
    try:
        if mb_client is None:
            logger.error("Macrobond client is None")
            return pd.DataFrame()
        
        logger.info(f"config.iip_gdp_tickers exists: {hasattr(config, 'iip_gdp_tickers')}")
        if hasattr(config, 'iip_gdp_tickers'):
            logger.info(f"config.iip_gdp_tickers type: {type(config.iip_gdp_tickers)}")
            logger.info(f"config.iip_gdp_tickers value: {config.iip_gdp_tickers}")
        
        iip_gdp = mb_client.FetchSeries(list(config.iip_gdp_tickers.keys()))
        iip_gdp = iip_gdp.rename(columns=config.COLUMN_MAPPINGS['iip_gdp'])
        iip_gdp = iip_gdp.resample('ME').mean()
        return iip_gdp
    except Exception as e:
        logger.error(f"Error fetching IIP/GDP data: {e}")
        return pd.DataFrame()

def fetch_risk_rating_data(mb_client):
    """Fetch and consolidate credit rating data from multiple agencies."""
    try:
        if mb_client is None:
            logger.error("Macrobond client is None")
            return pd.DataFrame()
        
        # Fetch Moody's, Fitch, and S&P ratings
        m_rating = mb_client.FetchSeries(list(config.moodys_rating_tickers.keys()))
        m_rating.index.name = "Date"
        m_rating = m_rating.rename(columns=config.COLUMN_MAPPINGS['moody_ratings'])
        m_rating = m_rating.resample('ME').mean()
        
        f_rating = mb_client.FetchSeries(list(config.fitch_rating_tickers.keys()))
        f_rating.index.name = "Date"
        f_rating = f_rating.rename(columns=config.COLUMN_MAPPINGS['fitch_ratings'])
        f_rating = f_rating.resample('ME').mean()
        
        s_rating = mb_client.FetchSeries(list(config.sp_rating_tickers.keys()))
        s_rating.index.name = "Date"
        s_rating = s_rating.rename(columns=config.COLUMN_MAPPINGS['sp_ratings'])
        s_rating = s_rating.resample('ME').mean()
        
        # Calculate consolidated risk rating
        if not m_rating.empty and not s_rating.empty and not f_rating.empty:
            country_rating = pd.concat([m_rating, s_rating, f_rating], axis=1)
            
            countries = list(config.country_list_mapping.values())
            averages = {}
            
            for country_code in countries:
                country_columns = [col for col in country_rating.columns if col.endswith(f'_{country_code}')]
                if country_columns:  # Only process countries with data
                    averages[country_code] = country_rating[country_columns].mean(axis=1)
            
            risk_rating = pd.DataFrame(averages)
            risk_rating = risk_rating.rename(columns=config.COLUMN_MAPPINGS['consolidated_ratings'])
            return risk_rating
        else:
            logger.warning("Cannot calculate risk rating: one or more rating agencies' data is missing")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching credit rating data: {e}")
        return pd.DataFrame()

def forward_fill_to_current_date(df, freq='ME'):
    """
    Forward-fill a DataFrame to the current date
    
    Parameters:
        df: DataFrame - DataFrame to forward-fill
        freq: str - Frequency for date_range ('ME' for monthly, 'D' for daily)
        
    Returns:
        DataFrame: Forward-filled DataFrame
    """
    if df is None or df.empty:
        return df
        
    today = pd.Timestamp.today()
    if df.index.max() < today:
        # Create a date range from the earliest date to today
        extended_dates = pd.date_range(start=df.index.min(), end=today, freq=freq)
        # Reindex and forward fill
        df_extended = df.reindex(extended_dates).ffill()
        return df_extended
    
    return df

def select_best_model(model_results_dict, country, tenor_name, output_dir=None):
    """
    Select the best model based on a comprehensive set of evaluation criteria.
    Also saves the best model to disk.
    
    Parameters:
        model_results_dict: dict - Dictionary of model results by model type
        country: str - Country name
        tenor_name: str - Yield tenor name
        output_dir: str - Directory to save the best model (defaults to config.MODEL_DIR)
            
    Returns:
        str: Best model type or None if no successful models
    """
    if not model_results_dict:
        logger.warning("No model results provided for selection")
        return None
    
    # If output_dir is not specified, use the MODEL_DIR from config
    if output_dir is None:
        output_dir = getattr(config, 'MODEL_DIR', '.')
        
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
                
    # Collect successful models
    successful_models = {
        model_type: result for model_type, result in model_results_dict.items() 
        if result.get('status', '').startswith('Success')
    }
    
    if not successful_models:
        logger.warning("No successful models to select from")
        return None
    
    # If only one successful model, return it
    if len(successful_models) == 1:
        return list(successful_models.keys())[0]
            
    # Calculate scores for each model based on multiple criteria
    model_scores = calculate_model_scores(successful_models)
    
    # Select model with highest score
    if model_scores:
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        best_model_type = sorted_models[0][0]
        logger.info(f"Selected {best_model_type} as best model with score {model_scores[best_model_type]['total_score']:.4f}")
        
        # Log score differences for transparency
        if len(sorted_models) > 1:
            logger.info("Model score comparison:")
            for model_type, score_info in sorted_models:
                logger.info(f"  {model_type}: {score_info['total_score']:.4f}")
        
        # Save the best model
        save_best_model(model_results_dict[best_model_type], best_model_type, country, tenor_name, output_dir)
        
        return best_model_type
    else:
        logger.warning("Could not calculate scores for any models")
        return None

def calculate_model_scores(successful_models):
    """
    Calculate scores for each model based on multiple criteria.
    """
    # Define weights for different metrics (configurable)
    weights = getattr(config, 'MODEL_SELECTION_WEIGHTS', {
        'test_rmse': 1.0,        # Primary metric (lower is better)
        'test_r2': 0.8,          # Secondary metric (higher is better)
        'train_rmse': 0.5,       # Training error (lower is better)
        'overfitting': 0.7,      # Difference between train and test error (lower is better)
        'simplicity': 0.3,       # Model complexity penalty (higher is better)
        'economic_validity': 0.6, # How well model aligns with economic theory
    })
    
    # Model complexity rankings (lower is simpler/better)
    complexity_ranking = {
        'elasticnet': 1,    # Linear models are simplest
        'ridge': 1,
        'lasso': 1,
        'nelson_siegel': 1, # Domain-specific models
        'gbm': 1,           # Tree-based models
        'randomforest': 1,
        'xgboost': 1,
        'mlp': 1,           # Neural networks
        'gp': 1,            # Gaussian Process
        'ensemble': 1       # Ensemble methods most complex
    }
    
    model_scores = {}
    
    # Calculate scores for each model
    for model_type, result in successful_models.items():
        score_components = calculate_score_components(result, model_type, complexity_ranking)
        
        # Calculate weighted score
        total_score = sum(score_components[metric] * weights[metric] for metric in score_components)
        
        # Store for comparison
        model_scores[model_type] = {
            'total_score': total_score,
            'components': score_components
        }
        
        logger.info(f"Model {model_type} score: {total_score:.4f} (components: {score_components})")
    
    return model_scores

def calculate_score_components(result, model_type, complexity_ranking):
    """
    Calculate individual score components for a model result.
    """
    metrics = result.get('metrics', {})
    
    # Initialize score components
    score_components = {
        'test_rmse': 0,
        'test_r2': 0,
        'train_rmse': 0,
        'overfitting': 0,
        'simplicity': 0,
        'economic_validity': 0
    }
    
    # 1. Test RMSE (lower is better)
    test_metrics = metrics.get('test', {})
    if test_metrics and 'rmse' in test_metrics:
        test_rmse = test_metrics['rmse']
        score_components['test_rmse'] = -test_rmse  # Negative because lower is better
    
    # 2. Test R² (higher is better)
    if test_metrics and 'r2' in test_metrics:
        test_r2 = test_metrics['r2']
        score_components['test_r2'] = test_r2
    
    # 3. Train RMSE (lower is better)
    train_metrics = metrics.get('train', {})
    if train_metrics and 'rmse' in train_metrics:
        train_rmse = train_metrics['rmse']
        score_components['train_rmse'] = -train_rmse  # Negative because lower is better
    
    # 4. Overfitting (difference between train and test performance)
    if 'rmse' in train_metrics and 'rmse' in test_metrics:
        # Calculate overfitting as the ratio of test to train RMSE (closer to 1.0 is better)
        overfitting_ratio = test_metrics['rmse'] / train_metrics['rmse'] if train_metrics['rmse'] > 0 else float('inf')
        # Penalize more for overfitting (ratio > 1.3) or underfitting (ratio < 0.9)
        overfitting_penalty = -abs(overfitting_ratio - 1.0) * 2.0
        score_components['overfitting'] = overfitting_penalty
    
    # 5. Model Simplicity (simpler models are preferred when performance is similar)
    model_complexity = complexity_ranking.get(model_type, 3.0)  # Default to middle complexity
    # Normalize to 0-1 range (1 is best/simplest)
    simplicity_score = (6.0 - model_complexity) / 5.0  
    score_components['simplicity'] = simplicity_score
    
    # 6. Economic Validity (models that align with economic theory are preferred)
    economic_validity = result.get('economic_validity', {})
    valid_relationships = 0
    total_relationships = 0
    
    # Check inflation impact (should be positive for yields)
    if 'inflation_impact' in economic_validity:
        valid_relationships += 1 if economic_validity['inflation_impact'] else 0
        total_relationships += 1
        
    # Check policy rate impact (should be positive for yields)
    if 'policy_rate_impact' in economic_validity:
        valid_relationships += 1 if economic_validity['policy_rate_impact'] else 0
        total_relationships += 1
    
    # Calculate economic validity score
    if total_relationships > 0:
        score_components['economic_validity'] = valid_relationships / total_relationships
    else:
        # Default to neutral if no relationships checked
        score_components['economic_validity'] = 0.5
    
    return score_components

def save_best_model(best_model_data, best_model_type, country, tenor_name, output_dir):
    """
    Save the best model to disk with all necessary components.
    """
    if 'model' in best_model_data and 'scaler' in best_model_data:
        model_filename = f"{country}_{tenor_name}_best_model.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        # Get feature names if available
        feature_names = extract_feature_names(best_model_data, country, tenor_name)
        
        # Create a package with all necessary components
        model_package = {
            'model': best_model_data['model'],
            'scaler': best_model_data['scaler'],
            'model_type': best_model_type,
            'performance': best_model_data.get('metrics', {}),
            'feature_names': feature_names,
            'creation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save using joblib
        try:
            joblib.dump(model_package, model_path)
            logger.info(f"Saved best model ({best_model_type}) for {country} - {tenor_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving best model: {str(e)}")
    else:
        logger.warning(f"Cannot save best model ({best_model_type}) - missing required components")