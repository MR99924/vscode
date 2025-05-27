import numpy as np
import datetime
import time
import os
import sys
import config 
import traceback
import json
import pandas as pd
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import bloomberg
import logging
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from macrobond import Macrobond
from sklearn.preprocessing import LabelEncoder
from feature_engineering import forward_fill_to_current_date
from model_evaluation import evaluate_model_performance, perform_residual_analysis
from model_training import optimize_hyperparameters, train_model_with_cross_validation, build_ensemble_model, train_benchmark_models
from prepare_data import prepare_data
from model_tester import extract_feature_names
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np


# Initialize API connectors
mb = Macrobond()
label_encoder = LabelEncoder()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_predictions = {}

def get_bloomberg_date(tickers, date_from, date_to, field="PX_LAST", periodicity="DAILY"):
    """
    Fetch data from Bloomberg for the given tickers and date range.
    
    Parameters:
        tickers: list - List of Bloomberg tickers
        date_from: datetime.date - Start date for data
        date_to: datetime.date - End date for data
        field: str - Bloomberg field (default: "PX_LAST")
        periodicity: str - Data frequency (default: "DAILY")
        
    Returns:
        DataFrame: Data for requested tickers and date range
    """
    try:
        bbg = bloomberg.Bloomberg()
        df = bbg.historicalRequest(tickers,
                                field,
                                date_from,
                                date_to,
                                periodicitySelection=periodicity,
                                nonTradingDayFillOption="ALL_CALENDAR_DAYS",
                                nonTradingDayFillMethod="PREVIOUS_VALUE",
                                )
        
        # Pivot the data to get a clean dataframe with dates as index
        df = pd.pivot_table(df,
                        values='bbergvalue',
                        index=['bbergdate'],
                        columns=['bbergsymbol'],
                        aggfunc=np.max,
                        )
        
        # Ensure all requested tickers are in the final DataFrame
        for ticker in tickers:
            if ticker not in df.columns:
                logger.warning(f"Ticker {ticker} not found in Bloomberg data")
        
        # Keep only the requested tickers that were found
        existing_tickers = [t for t in tickers if t in df.columns]
        if existing_tickers:
            df = df[existing_tickers]
        else:
            logger.warning("None of the requested tickers were found")
            df = pd.DataFrame(index=pd.date_range(date_from, date_to))
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching Bloomberg data: {e}")
        return pd.DataFrame(index=pd.date_range(date_from, date_to))

def show_popup_alert(title, message):
    """
    Show a popup alert dialog with the given title and message.
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showwarning(title, message)
        root.destroy()
        return True
    except Exception:
        print(f"\n{'=' * 50}\n{title}\n{'-' * 50}\n{message}\n{'=' * 50}\n")
        return False

all_fitted_values = {}

def train_evaluate_model(country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, iip_gdp,
                        act_track, risk_rating, historical_forecasts, unemployment_rate, model_type, predicted_yields=None,
                         use_advanced_training=True, compare_models=False, optimize_params=True, 
                        output_dir=None, use_yield_curve_models=True):
    """
    Train and evaluate a model for a specific country and yield tenor.
    Enhanced to support multiple model types, Nelson-Siegel, and advanced ensemble techniques.
    Also includes improved error handling and compatibility fixes.
    
    Parameters:
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
        model_type: str - Type of model to train ('mlp', 'gbm', 'elasticnet', 'gp', 'ensemble', 'nelson_siegel')
        use_advanced_training: bool - Whether to use advanced training methods (cross-validation, etc.)
        compare_models: bool - Whether to compare against other model types
        optimize_params: bool - Whether to perform hyperparameter optimization
        output_dir: str - Directory to save outputs (defaults to RESULTS_DIR/country/tenor)
        use_yield_curve_models: bool - Whether to use yield curve specific models for appropriate tenors
        
    Returns:
        dict: Model evaluation results and diagnostics
    """

    logger.info(f"\n=== TRAINING {model_type.upper()} MODEL FOR {country} - {tenor_name} ===")
    
    # Create output directory if specified
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
    
    # Initialize results dictionary
    results = {
        'country': country,
        'tenor': tenor_name,
        'model_type': model_type,
        'status': 'Not Started',
        'metrics': {
            'train': {},
            'test': {},
            'validation': {}  # Added for cross-validation metrics
        },
        'feature_importance': None,
        'data_summary': None,
        'error': None,
        'predictions': None,
        'model': None,
        'scaler': None,
        'economic_validity': {}  # Track if predictions align with economic theory
    }
    
    # Get country code
    if country not in country_code_mapping:
        logger.error(f"Country {country} not found in country_code_mapping")
        results['status'] = 'Failed - Invalid Country'
        results['error'] = f"Country {country} not found in mapping"
        return results
    
    country_code = country_code_mapping[country]
    
    try:
        # Prepare data with enhanced feature engineering
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
        
        
        # Store data summary in results
        results['data_summary'] = feature_details
        
        # Check if we have enough data
        if x is None or x.empty or y is None or len(y) == 0:
            logger.error(f"Insufficient data to train model for {country} - {tenor_name}")
            results['status'] = 'Failed - Insufficient Data'
            results['error'] = 'No overlapping data between features and target'
            return results
            
        if len(x) < config.MIN_DATA_POINTS_FOR_MODEL:  # Use value from config
            logger.error(f"Not enough data points ({len(x)}) to train a reliable model")
            results['status'] = 'Failed - Too Few Data Points'
            results['error'] = f'Only {len(x)} data points available (minimum {config.MIN_DATA_POINTS_FOR_MODEL} required)'
            return results
        
        # Add after checking for minimum data points in train_evaluate_model function
        if len(x) < 30:  # Very small dataset
            logger.warning(f"Very small dataset ({len(x)} points). Using simplified training approach instead of cross-validation.")
            try:
                # Use a simple 70/30 train/test split instead of cross-validation
                from sklearn.model_selection import train_test_split
                
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Use a simpler model
                mlp = MLPRegressor(
                    hidden_layer_sizes=(16, 8),  # Smaller network
                    activation='relu',
                    solver='adam',
                    alpha=0.01,  # Stronger regularization
                    max_iter=2000,  # More iterations
                    tol=1e-4,  # More relaxed tolerance
                    random_state=42
                )
                
                mlp.fit(X_train_scaled, y_train)
                
                # Calculate metrics
                y_train_pred = mlp.predict(X_train_scaled)
                y_test_pred = mlp.predict(X_test_scaled)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Create full predictions
                X_all_scaled = scaler.transform(x)
                all_predictions = mlp.predict(X_all_scaled)
                
                # Return results
                results['model'] = mlp
                results['scaler'] = scaler
                results['predictions'] = pd.Series(all_predictions, index=x.index)
                results['metrics'] = {
                    'train': {'rmse': train_rmse, 'r2': train_r2},
                    'test': {'rmse': test_rmse, 'r2': test_r2}
                }
                results['status'] = 'Success - Simple Training (Small Dataset)'
                
                return results
            except Exception as e:
                logger.error(f"Error in simple training for small dataset: {str(e)}")


        
        logger.info(f"Training {model_type} model with {len(x)} data points and {x.shape[1]} features")
        
        if x is None or x.empty or y is None or len(y) == 0:
            logger.error(f"Insufficient data to train model for {country} - {tenor_name}")
            results['status'] = 'Failed - Insufficient Data'
            results['error'] = 'No overlapping data between features and target'
            return results
            
        if len(x) < config.MIN_DATA_POINTS_FOR_MODEL:  # Use value from config
            logger.error(f"Not enough data points ({len(x)}) to train a reliable model")
            results['status'] = 'Failed - Too Few Data Points'
            results['error'] = f'Only {len(x)} data points available (minimum {config.MIN_DATA_POINTS_FOR_MODEL} required)'
            return results

        # Update status
        results['status'] = 'Data Prepared'

        # ADDED: Try ultra-simple training first as fallback
        logger.info(f"Using fallback simple training approach for {country} - {tenor_name}")
        try:
            # Simple train/test split (80/20)
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            
            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create a very simple model based on model_type
            if model_type == 'mlp':
                model = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam', 
                                alpha=0.1, max_iter=500, random_state=42)
            elif model_type == 'elasticnet':
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=500, random_state=42)
            elif model_type == 'gbm':
                model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, 
                                                max_depth=3, random_state=42)
            else:
                # Default to ridge regression as most stable option
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0, random_state=42)
                logger.info(f"Using Ridge regression for {model_type} model type")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Calculate metrics
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_train_pred)
            
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Generate predictions for all data
            X_all_scaled = scaler.transform(x)
            all_predictions = model.predict(X_all_scaled)
            
            # Store results
            results['model'] = model
            results['scaler'] = scaler
            results['predictions'] = pd.Series(all_predictions, index=x.index)
            results['metrics'] = {
                'train': {'rmse': train_rmse, 'r2': train_r2, 'mse': train_mse},
                'test': {'rmse': test_rmse, 'r2': test_r2, 'mse': test_mse}
            }
            results['status'] = f'Success - Simple {model_type.upper()} Training'
            
            # Store in all_fitted_values for later consolidation
            fitted_df = pd.DataFrame({
                'date': x.index,
                'actual': y,
                'predicted': all_predictions,
                'residual': y - all_predictions,
                'country': country,
                'tenor': tenor_name,
                'model_type': model_type
            })
            
            # Use globals() to access the global variable
            if 'all_fitted_values' in globals():
                globals()['all_fitted_values'][f"{country}_{tenor_name}_{model_type}"] = fitted_df
            
            logger.info(f"Fallback training successful for {country} - {tenor_name}")
            logger.info(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            
            return results
            
        except Exception as e:
            logger.warning(f"Fallback training failed: {str(e)}. Trying advanced training...")
    # Continue to advanced training as a second attempt
        
        # Determine whether to use basic or advanced training approach
        if use_advanced_training:
            logger.info(f"Using advanced training approach with cross-validation for {model_type}")
            
            # Check if we should use a yield curve specific model for this tenor
            if use_yield_curve_models and model_type == 'nelson_siegel':
                # Infer tenor value in years from tenor_name
                tenor_years = None
                if tenor_name == 'yld_2yr':
                    tenor_years = 2
                elif tenor_name == 'yld_5yr':
                    tenor_years = 5
                elif tenor_name == 'yld_10yr':
                    tenor_years = 10
                elif tenor_name == 'yld_30yr':
                    tenor_years = 30
                
                if tenor_years is not None:
                    logger.info(f"Using Nelson-Siegel model for {tenor_name} ({tenor_years} years)")
                    
                    # Create tenor array and corresponding yields array for all available tenors
                    tenors = []
                    yields = []
                    
                    if predicted_yields is not None and country in predicted_yields:
                        # Add data from other tenors if available
                        for other_tenor, tenor_data_dict in predicted_yields[country].items():
                            for model_key, yield_data in tenor_data_dict.items():
                                other_tenor_years = None
                                if 'tenor_years' in yield_data:
                                    other_tenor_years = yield_data['tenor_years']
                                else:
                                    # Infer tenor years from tenor name
                                    if other_tenor == 'yld_2yr':
                                        other_tenor_years = 2
                                    elif other_tenor == 'yld_5yr':
                                        other_tenor_years = 5
                                    elif other_tenor == 'yld_10yr':
                                        other_tenor_years = 10
                                    elif other_tenor == 'yld_30yr':
                                        other_tenor_years = 30
                                
                                if other_tenor_years is not None and 'current_value' in yield_data:
                                    # Add if not already in the list
                                    if other_tenor_years not in tenors:
                                        tenors.append(other_tenor_years)
                                        yields.append(yield_data['current_value'])
                                        logger.info(f"Added existing tenor data: {other_tenor} - {other_tenor_years} years")
                    
                    # Add current tenor
                    if tenor_years not in tenors:
                        tenors.append(tenor_years)
                        yields.append(y.mean())
                    
                    try:
                        # Convert to numpy arrays
                        tenors_arr = np.array(tenors)
                        yields_arr = np.array(yields)
                        
                        # Check if we have enough data points for Nelson-Siegel
                        if len(tenors_arr) < 2:
                            logger.warning(f"Insufficient tenor points for Nelson-Siegel model. Need at least 2, got {len(tenors_arr)}")
                            logger.warning("Falling back to standard model approach")
                            
                            # [MODIFICATION: Return structured failure result rather than continuing]
                            results['status'] = 'Failed - Insufficient Tenor Points'
                            results['error'] = f"Nelson-Siegel requires at least 2 tenor points, got {len(tenors_arr)}"
                            return results
                        else:

                            
                            # Create Nelson-Siegel model
                            try:
                                start_time = time.time()
                                ns_model = ModelFactory.create_model('nelson_siegel')
                                
                                # Fit model
                                ns_model.fit(tenors_arr.reshape(-1, 1), yields_arr)
                                training_time = time.time() - start_time
                                
                                # Store model
                                results['model'] = ns_model
                                results['status'] = 'Success - Nelson-Siegel'
                                results['training_time'] = training_time
                                
                                # Generate predictions (simulate for time series)
                                all_predictions = np.full_like(y, ns_model.predict(np.array([tenor_years]).reshape(-1, 1))[0])
                                
                                # Store predictions
                                results['predictions'] = pd.Series(all_predictions, index=x.index)
                                
                                # Calculate metrics
                                mse = mean_squared_error(y, all_predictions)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y, all_predictions)
                                
                                # Store metrics in a more structured way
                                results['metrics'] = {
                                    'all': {
                                        'mse': mse,
                                        'rmse': rmse,
                                        'r2': r2
                                    },
                                    'train': {
                                        'mse': mse,
                                        'rmse': rmse,
                                        'r2': r2
                                    },
                                    'test': {
                                        'mse': mse,
                                        'rmse': rmse,
                                        'r2': r2
                                    }
                                }
                                
                                logger.info(f"Nelson-Siegel model metrics: RMSE={rmse:.4f}, R²={r2:.4f}")
                                
                                return results
                            except Exception as e:
                                logger.error(f"Error fitting Nelson-Siegel model: {str(e)}")
                                results['status'] = 'Failed - Nelson-Siegel Error'
                                results['error'] = str(e)
                                return results
                    except Exception as e:
                        logger.error(f"Error preparing data for Nelson-Siegel model: {str(e)}")
                        # [MODIFICATION: Return structured failure result]
                        results['status'] = 'Failed - Nelson-Siegel Data Preparation'
                        results['error'] = str(e)
                        return results
                    
            # Define base model parameters for the selected model type
            model_params = {}
            
            if model_type == 'mlp':
                if model_type == 'mlp' or (hasattr(model, '__class__') and model.__class__.__name__ == 'MLPRegressor'):

                    model_params = {
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
                elif model_type == 'enhanced_mlp':
                    model_params = {
                        'hidden_layer_sizes': (64, 32),
                        'activation': 'relu',
                        'solver': 'adam',
                        'alpha': 0.0001,
                        'max_iter': 1000,
                        'early_stopping': True,
                        'validation_fraction': 0.1,
                        'n_iter_no_change': 10,
                        'ensemble_size': 5
                    }
                elif model_type == 'gbm':
                    model_params = {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'random_state': 42
                    }
                elif model_type == 'xgboost':
                    model_params = {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'random_state': 42
                    }
                elif model_type == 'elasticnet':
                    model_params = {
                        'alpha': 0.1,
                        'l1_ratio': 0.5,
                        'max_iter': 1000,
                        'random_state': 42
                    }
                elif model_type == 'gp':
                    model_params = {
                        'length_scale': 1.0,
                        'alpha': 0.1,
                        'n_restarts_optimizer': 5,
                        'normalize_y': True
                    }
            
            # Optimize hyperparameters if requested
            if optimize_params:
                logger.info(f"Performing hyperparameter optimization for {model_type}")
                
                # Define parameter grid based on model type
                if model_type == 'mlp' or model_type == 'enhanced_mlp':
                    param_grid = {
                        'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64, 32)],
                        'activation': ['relu', 'tanh'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate_init': [0.001, 0.01]
                    }
                elif model_type == 'gbm':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                elif model_type == 'xgboost':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [1, 2, 5]
                    }
                elif model_type == 'elasticnet':
                    param_grid = {
                        'alpha': [0.001, 0.01, 0.1, 1.0],
                        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                    }
                elif model_type == 'gp':
                    param_grid = {
                        'length_scale': [0.1, 1.0, 10.0],
                        'alpha': [0.01, 0.1, 1.0]
                    }
                else:
                    param_grid = {}
                
                try:
                    # Optimize parameters
                    best_model, best_params, cv_results = optimize_hyperparameters(
                        x, y, model_type=model_type, param_grid=param_grid, cv_method='grid'
                    )
                    
                    # Update parameters with optimized values
                    if best_params:
                        model_params.update(best_params)
                        logger.info(f"Optimized parameters: {best_params}")
                        
                        # Store optimization results
                        results['optimization'] = {
                            'best_params': best_params,
                            'cv_results': cv_results
                        }
                except Exception as e:
                    logger.error(f"Error during hyperparameter optimization: {str(e)}")
                    logger.warning("Continuing with default parameters")
            
            # Train with cross-validation with error handling
            try:
                model_package, cv_results, feature_imp = train_model_with_cross_validation(
                    x, y, model_type=model_type, n_splits=3, params=model_params
                )
                
                # Check if training was successful
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
            
            # Compare with other model types if requested
            if compare_models:
                logger.info("Training benchmark models for comparison")
                try:
                    # Define models to compare with
                    models_to_train = ['mlp', 'gbm', 'xgboost', 'elasticnet']
                    
                    # Add GP model if data size is reasonable (GP can be slow with large datasets)
                    if len(x) < 1000:
                        models_to_train.append('gp')
                    
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
                        
                        # Create ensemble if requested and if enough models succeeded
                        model_success_count = sum(1 for key in benchmark_results if key not in ['comparison', 'best_model'] 
                                               and benchmark_results[key].get('model_package') is not None)
                        
                        if model_success_count >= 2:
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
                                    # First check if we have the necessary imports
                                    try:
                                        from model_training import build_ensemble_model
                                        
                                        ensemble_model, ensemble_metrics = build_ensemble_model(
                                            models, x, y, weighted_by_performance=True
                                        )
                                        
                                        # Store ensemble results
                                        results['ensemble_model'] = ensemble_model
                                        results['ensemble_metrics'] = ensemble_metrics
                                        
                                        logger.info(f"Ensemble model performance: RMSE={ensemble_metrics.get('rmse', 'N/A')}, R²={ensemble_metrics.get('r2', 'N/A')}")
                                        
                                        # Check if ensemble performs better than single model and if so, use it
                                        if ('rmse' in ensemble_metrics and 'avg_test_rmse' in cv_results and 
                                            ensemble_metrics['rmse'] < cv_results['avg_test_rmse']):
                                            logger.info("Ensemble model outperforms single model - using ensemble as primary model")
                                            results['model'] = ensemble_model
                                            results['metrics']['ensemble'] = ensemble_metrics
                                            results['status'] = 'Success - Ensemble Model'
                                    except ImportError:
                                        logger.error("Could not import build_ensemble_model function. Skipping ensemble creation.")
                                    except Exception as e:
                                        logger.error(f"Error building ensemble model: {str(e)}")
                                else:
                                    logger.warning(f"Not enough successful models to build ensemble (need at least 2, got {len(models)})")
                            except Exception as e:
                                logger.error(f"Error in ensemble creation: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in benchmark model training: {str(e)}")
            
            # Generate predictions for all data
            try:
                if results['model'] is not None:
                    if results['scaler'] is not None:
                        scaler = results['scaler']
                        x_scaled = scaler.transform(x)
                        all_predictions = results['model'].predict(x_scaled)
                    else:
                        # Some models might not need scaling
                        all_predictions = results['model'].predict(x)
                    
                    # Store predictions
                    if isinstance(all_predictions, np.ndarray):
                        results['predictions'] = pd.Series(all_predictions, index=x.index)
                    else:
                        # Handle case where predictions might already be a Series
                        results['predictions'] = all_predictions
                    
                    # Store data for consolidated fitted values
                    if 'predictions' in results and results['predictions'] is not None:
                        fitted_df = pd.DataFrame({
                            'date': x.index,
                            'actual': y,
                            'predicted': results['predictions'],
                            'residual': y - results['predictions'],
                            'country': country,
                            'tenor': tenor_name,
                            'model_type': model_type
                        })
                        
                        # Store in global all_fitted_values
                        if 'all_fitted_values' in globals():
                            key = f"{country}_{tenor_name}_{model_type}"
                            globals()['all_fitted_values'][key] = fitted_df
                        else:
                            logger.warning("all_fitted_values not found in global scope. Results won't be consolidated.")
            except Exception as e:
                logger.error(f"Error generating predictions: {str(e)}")
                if results['status'].startswith('Success'):
                    # Only modify status if it was previously successful
                    results['status'] = f'Partial Success - {model_type} (Prediction Error)'
                    results['error'] = f"Model trained successfully but failed to generate predictions: {str(e)}"
        
        else:
            # Traditional approach with chronological train/test split
            logger.info("Using traditional training approach with time-based split")
            
            # Use original approach with chronological train/test split
            train_size = int(len(x) * 0.6)  # Default 80/20 split if not in config
            if hasattr(config, 'DEFAULT_TRAIN_TEST_SPLIT'):
                train_size = int(len(x) * config.DEFAULT_TRAIN_TEST_SPLIT)
            
            x_train = x.iloc[:train_size]
            y_train = y.iloc[:train_size]
            x_test = x.iloc[train_size:]
            y_test = y.iloc[train_size:]
            
            logger.info(f"Training set: {len(x_train)} samples, Test set: {len(x_test)} samples")
            
            # Scale the data
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test) if not x_test.empty else None
            
            # Create the model based on model_type
            try:
                # Import if needed
                from model_classes import ModelFactory
                
                model = ModelFactory.create_model(model_type)
            except ImportError:
                logger.error("Could not import ModelFactory. Falling back to default MLPRegressor.")
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=42
                )
            
            # Train model
            try:
                model.fit(x_train_scaled, y_train)
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                results['status'] = 'Failed - Model Training'
                results['error'] = str(e)
                return results
            
            # Evaluate on training set
            try:
                y_train_pred = model.predict(x_train_scaled)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_rmse = np.sqrt(train_mse)
                train_r2 = safe_r2_score(y_train, y_train_pred)
                
                logger.info(f"Training metrics: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
            except Exception as e:
                logger.error(f"Error calculating training metrics: {str(e)}")
                train_mse = train_rmse = train_r2 = np.nan
            
            # Evaluate on test set if available
            if not x_test.empty and x_test_scaled is not None:
                try:
                    y_test_pred = model.predict(x_test_scaled)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = safe_r2_score(y_test, y_test_pred)
                    
                    logger.info(f"Test metrics: RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
                    
                    # Store metrics
                    results['metrics'] = {
                        'train': {
                            'mse': train_mse,
                            'rmse': train_rmse,
                            'r2': train_r2
                        },
                        'test': {
                            'mse': test_mse,
                            'rmse': test_rmse,
                            'r2': test_r2
                        }
                    }
                except Exception as e:
                    logger.error(f"Error calculating test metrics: {str(e)}")
                    # Only store training metrics if test metrics fail
                    results['metrics'] = {
                        'train': {
                            'mse': train_mse,
                            'rmse': train_rmse,
                            'r2': train_r2
                        }
                    }
            else:
                # Only store training metrics if no test data
                results['metrics'] = {
                    'train': {
                        'mse': train_mse,
                        'rmse': train_rmse,
                        'r2': train_r2
                    }
                }
            
            # Generate predictions for all data
            try:
                x_all_scaled = scaler.transform(x)
                all_predictions = model.predict(x_all_scaled)
                
                # Store predictions
                results['predictions'] = pd.Series(all_predictions, index=x.index)
                
                # Generate feature importance if available for the model type
                try:
                    # Import if needed
                    from model_classes import create_model_diagnostics
                    
                    importance_results = create_model_diagnostics(
                        model=model,
                        X=x,
                        y=y,
                        scaler=scaler,
                        feature_names=list(x.columns),
                        country=country,
                        tenor=tenor_name
                    )
                    
                    if importance_results and 'feature_importance' in importance_results:
                        results['feature_importance'] = importance_results['feature_importance']
                    if importance_results:
                        results['diagnostics'] = importance_results
                except ImportError:
                    logger.warning("Could not import create_model_diagnostics function. Skipping feature importance.")
                except Exception as e:
                    logger.error(f"Error generating feature importance: {str(e)}")
                
                # Store model and scaler
                results['model'] = model
                results['scaler'] = scaler
                results['status'] = f'Success - Traditional {model_type.upper()} Training'
                
                # Store in all_fitted_values for later consolidation
                fitted_df = pd.DataFrame({
                    'date': x.index,
                    'actual': y,
                    'predicted': all_predictions,
                    'residual': y - all_predictions,
                    'country': country,
                    'tenor': tenor_name,
                    'model_type': model_type
                })
                
                # Use globals() to access the global variable
                if 'all_fitted_values' in globals():
                    globals()['all_fitted_values'][f"{country}_{tenor_name}_{model_type}"] = fitted_df
                else:
                    logger.warning("all_fitted_values not found in global scope. Results won't be consolidated.")
            except Exception as e:
                logger.error(f"Error generating predictions or storing results: {str(e)}")
        
        # Save model predictions to CSV
        try:
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
        
        # Create a summary file with key metrics
        try:
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
            
            logger.info(f"Training complete for {country} - {tenor_name} using {model_type} model")
        except Exception as e:
            logger.error(f"Error creating summary file: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in train_evaluate_model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        results['status'] = 'Failed - Exception'
        results['error'] = str(e)
    
    return results


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
    print(f"\n--- VISUALIZING {model_type.upper()} MODEL FOR {country} - {tenor_name} ---")
    
    # Create a figure with 3 subplots
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # 1. Training data fit
        if not x_train.empty and len(y_train) > 0:
            print("Generating training data fit visualization...")
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
                print(f"  Training RMSE: {train_rmse:.4f}")
                print(f"  Training R²: {train_r2:.4f}")
                
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
                print(f"Error generating training visualization: {e}")
                axes[0].text(0.5, 0.5, f"Training visualization error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0].transAxes, fontsize=12)
        else:
            print("No training data available for visualization")
            axes[0].text(0.5, 0.5, "No training data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[0].transAxes, fontsize=14)
        
        # 2. Test data predictions
        if not x_test.empty and len(y_test) > 0:
            print("Generating test data predictions visualization...")
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
                print(f"  Test RMSE: {test_rmse:.4f}")
                print(f"  Test R²: {test_r2:.4f}")
                
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
                print(f"Error generating test visualization: {e}")
                axes[1].text(0.5, 0.5, f"Test visualization error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes, fontsize=12)
        else:
            print("No test data available for visualization")
            axes[1].text(0.5, 0.5, "No test data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes, fontsize=14)
        
        # 3. Combined training and test data visualization (third subplot)
        print("Generating combined visualization...")
        try:
            # Plot all data together in the third subplot
            ax = axes[2]

            # Plot training data
            if not x_train.empty and len(y_train) > 0:
                x_train_scaled = scaler.transform(x_train)
                y_train_pred = model.predict(x_train_scaled)

                # Actual training data
                ax.plot(y_train.index, y_train, 'b-', label='Actual (Training)', linewidth=1.5)
                # Training predictions
                ax.plot(y_train.index, y_train_pred, 'r--', label='Predicted (Training)', linewidth=1.5)

            # Plot test data
            if not x_test.empty and len(y_test) > 0:
                x_test_scaled = scaler.transform(x_test)
                y_test_pred = model.predict(x_test_scaled)

                ax.plot(y_test.index, y_test, 'g-', label='Actual (Test)', linewidth=1.5)
                ax.plot(y_test.index, y_test_pred, 'm--', label='Predicted (Test)', linewidth=1.5)

            # Add title and labels
            ax.set_title(f"{country} - {tenor_name}: Combined {model_type.upper()} Model Performance", fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Yield (%)', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)

            # Format x-axis to show years
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(3))
            ax.tick_params(axis='x', rotation=45)

            # Add combined metrics if available
            combined_metrics = ""
            if 'train_rmse' in locals():
                combined_metrics += f"Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}\n"
            if 'test_rmse' in locals():
                combined_metrics += f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}"

            if combined_metrics:
                ax.annotate(combined_metrics, 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                            fontsize=12)

        except Exception as e:
            print(f"Error generating combined visualization: {e}")
            axes[2].text(0.5, 0.5, f"Combined visualization error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes, fontsize=12)

        plt.tight_layout()

        # Save to file
        try:
            # Create file name with model type
            filename = f"{country}_{tenor_name}_{model_type}_model_analysis.png"
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            plt.savefig(filename)
            print(f"Model visualization saved to {filename}")
        except Exception as e:
            print(f"Error saving visualization file for {country}_{tenor_name}: {e}")
            plt.savefig(f"model_analysis_{country}_{tenor_name.replace('yld_', '')}_{model_type}.png")
            print("Saved with alternate filename")

        plt.close(fig)

    except Exception as e:
        print(f"Error visualizing model for {country} - {tenor_name}: {e}")
        traceback.print_exc()

    print("--- END OF MODEL VISUALIZATION ---\n")

def save_consolidated_fitted_values(results_summary=None):
    """
    Save only the best model fitted values for each country-tenor combination.
    Also creates a summary file of best model locations with absolute paths.

    Parameters:
    results_summary: dict - Summary of model results containing best model selections
    """
    global all_fitted_values

    if not all_fitted_values:
        print("No fitted values available to consolidate")
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
                    country_dir = os.path.join(config.MODEL_DIR, country)
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
                        print(f"Adding best model {key} to consolidated results")
                    else:
                        print(f"Warning: Best model {key} not found in fitted values")

            # Save the registry to a JSON file
            registry_path = os.path.join(config.OUTPUT_DIR, "best_models_registry.json")
            with open(registry_path, 'w') as f:
                json.dump(best_models_registry, f, indent=4)
            print(f"Saved best models registry to {registry_path}")

            # If we found any best models, use only those
            if best_fitted_values:
                print(f"Saving {len(best_fitted_values)} best models (filtering from {len(all_fitted_values)} total models)")
                consolidated_df = pd.concat(best_fitted_values.values(), ignore_index=True)
                consolidated_df.to_csv("best_fitted_values.csv")
                print("Saved best fitted values to 'best_fitted_values.csv'")

                # Also save the original complete set if needed
                consolidated_all_df = pd.concat(all_fitted_values.values(), ignore_index=True)
                consolidated_all_df.to_csv("all_fitted_values.csv")
                print("Also saved complete fitted values to 'all_fitted_values.csv'")

                return

        # Fallback to saving all values if no filtering was done
        print(f"Saving all {len(all_fitted_values)} fitted models (no filtering)")
        consolidated_df = pd.concat(all_fitted_values.values(), ignore_index=True)
        consolidated_df.to_csv("all_fitted_values.csv")
        print("Saved consolidated fitted values to 'all_fitted_values.csv'")

    except Exception as e:
        print(f"Error saving consolidated fitted values: {e}")
        import traceback
        traceback.print_exc()


def fix_train_model_with_cross_validation(X, y, model_type, n_splits=3, params=None):
    """
    Modified version of train_model_with_cross_validation function to fix issues with small datasets
    and MLPRegressor convergence.
    """
    logger.info(f"Training model with cross-validation: {model_type}, {n_splits} folds")
    
    # Import necessary libraries

    
    # Create safer cross-validation strategy
    # Ensure we have enough samples for the requested folds
    if len(X) < n_splits * 2:  # Need at least 2 samples per fold
        original_n_splits = n_splits
        n_splits = max(2, len(X) // 2)  # At least 2 samples per fold
        logger.warning(f"Dataset too small for {original_n_splits}-fold CV. Adjusted to {n_splits} folds.")
        
        # If dataset is extremely small, don't do cross-validation
        if n_splits < 2:
            logger.warning("Dataset too small for cross-validation. Using a simple 70/30 train/test split.")
            return train_model_simple_split(X, y, model_type=model_type, params=params)
    
    # Initialize results containers
    cv_results = {
        'train_mse': [], 'train_rmse': [], 'train_r2': [],
        'test_mse': [], 'test_rmse': [], 'test_r2': []
    }
    
    # Safely create model factory
    try:
        from model_classes import ModelFactory
        model_factory = ModelFactory()
    except Exception as e:
        logger.error(f"Could not import ModelFactory: {e}. Using direct model creation.")
        model_factory = None
    
    # Store feature importance from each fold
    feature_importance = {}
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # For storing the trained model
    best_model = None
    best_score = float('inf')
    best_scaler = None
    
    try:
        # Loop through each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold+1}/{n_splits}")
            
            try:
                # Check if indices are valid
                if max(train_idx) >= len(X) or max(test_idx) >= len(X):
                    logger.error(f"Invalid indices in fold {fold+1}: train_idx max={max(train_idx)}, test_idx max={max(test_idx)}, len(X)={len(X)}")
                    continue
                
                # Get train/test split for this fold
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Ensure sufficient test samples
                if len(X_test) < 2:
                    logger.warning(f"Test set too small in fold {fold+1}. Skipping this fold.")
                    continue
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create and train model
                if model_factory is not None:
                    model = model_factory.create_model(model_type, params)
                else:
                    # Direct model creation based on type
                    if model_type == 'mlp':
                        from sklearn.neural_network import MLPRegressor
                        # Modify MLPRegressor parameters to avoid convergence issues
                        mlp_params = {
                            'hidden_layer_sizes': (64, 32),
                            'activation': 'relu',
                            'solver': 'adam',
                            'alpha': 0.001,  # Increased regularization
                            'max_iter': 2000,  # Increased iterations
                            'early_stopping': True,
                            'validation_fraction': 0.15,
                            'n_iter_no_change': 20,
                            'tol': 1e-4,  # More relaxed tolerance
                            'random_state': 42
                        }
                        # Override with any user-provided params
                        if params:
                            mlp_params.update(params)
                        model = MLPRegressor(**mlp_params)
                    elif model_type == 'gbm':
                        from sklearn.ensemble import GradientBoostingRegressor
                        gbm_params = {
                            'n_estimators': 100,
                            'learning_rate': 0.1,
                            'max_depth': 3,
                            'random_state': 42
                        }
                        if params:
                            gbm_params.update(params)
                        model = GradientBoostingRegressor(**gbm_params)
                    # Add other model types as needed
                    else:
                        logger.error(f"Unsupported model type: {model_type}")
                        continue
                
                # Train model with error handling
                try:
                    model.fit(X_train_scaled, y_train)
                except Exception as e:
                    logger.error(f"Error training model in fold {fold+1}: {str(e)}")
                    continue
                
                # Make predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Calculate metrics safely
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_rmse = np.sqrt(train_mse)
                
                # Safe R² calculation
                train_r2 = safe_r2_score(y_train, y_train_pred)
                
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_rmse = np.sqrt(test_mse)
                test_r2 = safe_r2_score(y_test, y_test_pred)
                
                # Store metrics
                cv_results['train_mse'].append(train_mse)
                cv_results['train_rmse'].append(train_rmse)
                cv_results['train_r2'].append(train_r2)
                
                cv_results['test_mse'].append(test_mse)
                cv_results['test_rmse'].append(test_rmse)
                cv_results['test_r2'].append(test_r2)
                
                logger.info(f"Fold {fold+1} metrics - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                
                # Track best model
                if test_rmse < best_score:
                    best_score = test_rmse
                    best_model = model
                    best_scaler = scaler
                    
                # Get feature importance if available
                try:
                    if hasattr(model, 'feature_importances_'):
                        # For tree-based models
                        importances = model.feature_importances_
                        for i, col in enumerate(X.columns):
                            if col not in feature_importance:
                                feature_importance[col] = []
                            feature_importance[col].append(importances[i])
                    elif model_type == 'mlp':
                        # For MLPRegressor, use weights as approximate importance
                        weights = np.abs(model.coefs_[0])
                        importances = np.sum(weights, axis=1) / np.sum(weights)
                        for i, col in enumerate(X.columns):
                            if col not in feature_importance:
                                feature_importance[col] = []
                            feature_importance[col].append(importances[i])
                except Exception as e:
                    logger.warning(f"Could not calculate feature importance: {e}")
                    
            except Exception as e:
                logger.error(f"Error in fold {fold+1}: {str(e)}")
                continue
                
        # Calculate average metrics
        for metric in cv_results:
            if cv_results[metric]:
                cv_results[f'avg_{metric}'] = np.mean(cv_results[metric])
                cv_results[f'std_{metric}'] = np.std(cv_results[metric])
            else:
                cv_results[f'avg_{metric}'] = np.nan
                cv_results[f'std_{metric}'] = np.nan
        
        # Average feature importance
        avg_importance = {}
        for col in feature_importance:
            if feature_importance[col]:
                avg_importance[col] = np.mean(feature_importance[col])
        
        # Sort by importance
        sorted_importance = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Return results
        if best_model is not None and best_scaler is not None:
            logger.info("Cross-validation completed successfully")
            return {
                'model': best_model,
                'scaler': best_scaler
            }, cv_results, sorted_importance
        else:
            logger.error("No valid model was trained during cross-validation")
            return None, cv_results, sorted_importance
            
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return None, {'error': str(e)}, {}


def train_model_simple_split(X, y, model_type, params=None, test_size=0.3):
    """
    Train a model using a simple train/test split instead of cross-validation.
    Use this for very small datasets.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    logger.info(f"Training model with simple split: {model_type}, test_size={test_size}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    if model_type == 'mlp':
        from sklearn.neural_network import MLPRegressor
        mlp_params = {
            'hidden_layer_sizes': (32, 16),  # Smaller network for small datasets
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,  # Stronger regularization
            'max_iter': 2000,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'tol': 1e-4,
            'random_state': 42
        }
        if params:
            mlp_params.update(params)
        model = MLPRegressor(**mlp_params)
    # Add other model types as needed
    
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
        feature_importance = {}
        if model_type == 'mlp':
            weights = np.abs(model.coefs_[0])
            importances = np.sum(weights, axis=1) / np.sum(weights)
            for i, col in enumerate(X.columns):
                feature_importance[col] = importances[i]
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'model': model,
            'scaler': scaler
        }, cv_results, sorted_importance
        
    except Exception as e:
        logger.error(f"Error in simple split training: {str(e)}")
        return None, {'error': str(e)}, {}
    
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
        return safe_r2_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Error calculating R² score: {e}")
        return fallback_value

def fix_add_missing_mlp_attributes(model):
    """
    Add missing attributes to MLPRegressor if they don't exist.
    This fixes the '_best_coefs' attribute error.
    """
    if model.__class__.__name__ == 'MLPRegressor':
        if not hasattr(model, '_best_coefs') and hasattr(model, 'coefs_'):
            model._best_coefs = model.coefs_
        if not hasattr(model, '_best_intercepts') and hasattr(model, 'intercepts_'):
            model._best_intercepts = model.intercepts_
    return model


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
    print(f"\n--- FEATURE IMPORTANCE ANALYSIS FOR {country} - {tenor_name} ---")
    
    if mlp_model is None or x_data.empty or len(y_data) == 0:
        print("Error: Cannot generate feature importance report without a trained model and data")
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
    print("Calculating feature correlations...")
    correlations = {}
    for column in x_data.columns:
        correlation = x_data[column].corr(y_data)
        correlations[column] = correlation
        print(f"  {column}: correlation = {correlation:.4f}")
    
    # Sort by absolute correlation values
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    importance_results['correlations'] = {k: round(v, 4) for k, v in sorted_correlations}
    
    # 2. Calculate permutation importance
    try:
        print("\nCalculating permutation importance...")
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
            print(f"  {feature_name}: importance = {importance_score:.4f} ± {importance_std:.4f}")
        
        # Sort by importance score
        sorted_importance = sorted(perm_importance.items(), 
                                 key=lambda x: abs(x[1]['score']), 
                                 reverse=True)
        
        importance_results['permutation_importance'] = {
            k: {'score': round(v['score'], 4), 'std': round(v['std'], 4)} 
            for k, v in sorted_importance
        }
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
        importance_results['permutation_importance'] = {'error': str(e)}
    
    # 3. Analyze feature weights for MLPRegressor
    # Note: This is an approximation as MLP weights don't directly correspond to feature importance
    try:
        print("\nAnalyzing neural network weights...")
        
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
            print(f"  {feature}: weight importance = {normalized_weights[i]:.4f}")
        
        # Sort by weight importance
        sorted_weights = sorted(weight_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        importance_results['weight_importance'] = {k: round(float(v), 4) for k, v in sorted_weights}
    except Exception as e:
        print(f"Error analyzing neural network weights: {e}")
        importance_results['weight_importance'] = {'error': str(e)}
    
    # 4. Combine results from different methods to create a consensus ranking
    print("\nGenerating consensus feature ranking...")
    
    # Initialize feature scores
    feature_scores = {feature: {'score': 0, 'methods': {}} for feature in x_data.columns}
    
    # Add correlation scores
    for feature, corr in sorted_correlations:
        feature_scores[feature]['methods']['correlation'] = abs(corr)
    
    # Add permutation importance scores
    if 'error' not in importance_results['permutation_importance']:
        for feature, data in sorted_importance:
            feature_scores[feature]['methods']['permutation'] = abs(data['score'])
    
    # Add weight importance scores
    if 'error' not in importance_results.get('weight_importance', {'error': True}):
        for feature, weight in sorted_weights:
            feature_scores[feature]['methods']['weight'] = abs(weight)
    
    # Calculate consensus scores (average of available methods)
    for feature in feature_scores:
        method_scores = feature_scores[feature]['methods'].values()
        if method_scores:
            feature_scores[feature]['score'] = sum(method_scores) / len(method_scores)
        else:
            feature_scores[feature]['score'] = 0
    
    # Sort features by consensus score
    consensus_ranking = sorted(feature_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Create consensus ranking table
    importance_results['consensus_ranking'] = []
    
    print(f"{'Feature':<30} {'Consensus':<10} {'Correlation':<12} {'Permutation':<12} {'Weight':<10}")
    print(f"{'-'*30} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    
    for feature, data in consensus_ranking:
        methods = data['methods']
        consensus = data['score']
        corr = methods.get('correlation', float('nan'))
        perm = methods.get('permutation', float('nan'))
        weight = methods.get('weight', float('nan'))
        
        print(f"{feature:<30} {consensus:10.4f} {corr:12.4f} {perm:12.4f} {weight:10.4f}")
        
        importance_results['consensus_ranking'].append({
            'feature': feature,
            'consensus_score': round(consensus, 4),
            'correlation': round(corr, 4) if not np.isnan(corr) else None,
            'permutation': round(perm, 4) if not np.isnan(perm) else None,
            'weight': round(weight, 4) if not np.isnan(weight) else None
        })
    
    # 5. Create visualizations
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
        
        print(f"\nFeature importance visualization saved to {country}_{tenor_name}_feature_importance.png")
        
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
        
        print(f"Correlation heatmap saved to {country}_{tenor_name}_correlation_heatmap.png")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # 6. Export feature importance to CSV
    try:
        importance_df = pd.DataFrame(importance_results['consensus_ranking'])
        importance_df.to_csv(f"{country}_{tenor_name}_feature_importance.csv", index=False)
        print(f"Feature importance data saved to {country}_{tenor_name}_feature_importance.csv")
    except Exception as e:
        print(f"Error exporting feature importance to CSV: {e}")
    
    print("--- END OF FEATURE IMPORTANCE ANALYSIS ---\n")
    
    return importance_results

def fetch_all_data_sources():
    """
    Fetch all required data sources from Bloomberg and Macrobond.
    
    Returns:
        dict: Dictionary containing all data sources
    """
    logger = logging.getLogger(__name__)
    logger.info("Fetching all data sources")
    
    data_sources = {}
    
    try:
        # Define date range for data fetching
        dt_from = config.DEFAULT_DATE_FROM
        dt_to = config.DEFAULT_DATE_TO
        
        # 1. Fetch yield data for different tenors
        yield_data = {}
        
        # 2-year bond yields
        yld_2yr = get_bloomberg_date(
            list(config.bond_yield_tickers['2yr'].keys()), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_DAILY_PERIODICITY
        )
        yld_2yr = yld_2yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_2yr'])
        yld_2yr_ann = yld_2yr.resample('M').mean()
        yield_data['yld_2yr'] = yld_2yr
        yield_data['yld_2yr_ann'] = yld_2yr_ann
        
        # 5-year bond yields
        yld_5yr = get_bloomberg_date(
            list(config.bond_yield_tickers['5yr'].keys()), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_DAILY_PERIODICITY
        )
        yld_5yr = yld_5yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_5yr'])
        yld_5yr_ann = yld_5yr.resample('M').mean()
        yield_data['yld_5yr'] = yld_5yr
        yield_data['yld_5yr_ann'] = yld_5yr_ann
        
        # 10-year bond yields
        yld_10yr = get_bloomberg_date(
            list(config.bond_yield_tickers['10yr'].keys()), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_DAILY_PERIODICITY
        )
        yld_10yr = yld_10yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_10yr'])
        yld_10yr_ann = yld_10yr.resample('M').mean()
        yield_data['yld_10yr'] = yld_10yr
        yield_data['yld_10yr_ann'] = yld_10yr_ann
        
        # 30-year bond yields
        yld_30yr = get_bloomberg_date(
            list(config.bond_yield_tickers['30yr'].keys()), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_DAILY_PERIODICITY
        )
        yld_30yr = yld_30yr.rename(columns=config.COLUMN_MAPPINGS['bond_yield_30yr'])
        yld_30yr_ann = yld_30yr.resample('M').mean()
        yield_data['yld_30yr'] = yld_30yr
        yield_data['yld_30yr_ann'] = yld_30yr_ann
        
        data_sources['yield_data'] = yield_data
        
        # 2. Fetch policy rate data
        pol_rat = get_bloomberg_date(
            list(config.pol_rat_tickers.keys()), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_DAILY_PERIODICITY
        )
        pol_rat = pol_rat.rename(columns=config.COLUMN_MAPPINGS['policy_rates'])
        pol_rat = pol_rat.resample('M').mean()
        data_sources['policy_rates'] = pol_rat
        
        # 3. Fetch economic activity tracker data
        act_track = get_bloomberg_date(
            list(config.act_track_tickers.keys()), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_MONTHLY_PERIODICITY
        )
        act_track = act_track.rename(columns=config.COLUMN_MAPPINGS['activity'])
        if not act_track.empty:
            act_track.index = act_track.index.to_period("M").to_timestamp("M")
            act_track = act_track.resample('M').first().ffill()
            act_track.index = pd.DatetimeIndex(act_track.index.strftime('%Y-%m-%d'))
        data_sources['activity'] = act_track
        
        # 4. Fetch inflation data
        try:
            mb_instance = Macrobond()
            cpi_inf = mb_instance.FetchSeries(list(config.cpi_inf_tickers.keys()))
            cpi_inf = cpi_inf.rename(columns=config.COLUMN_MAPPINGS['cpi_forecast'])
            cpi_inf = cpi_inf.resample('M').mean()
            cpi_inf = cpi_inf.pct_change(periods=12) * 100  # Convert to year-over-year percentage change
            data_sources['inflation'] = cpi_inf
        except Exception as e:
            logger.error(f"Error fetching inflation data: {e}")
            data_sources['inflation'] = pd.DataFrame()
        
        try:
            mb_instance = Macrobond()
            unemployment_rate = mb_instance.FetchSeries(list(config.unemployment_tickers.keys()))
            unemployment_rate = unemployment_rate.rename(columns=config.COLUMN_MAPPINGS['unemployment_rate'])
            unemployment_rate = unemployment_rate.resample('M').mean()
            data_sources['unemployment_rate'] = unemployment_rate
        except Exception as e:
            logger.error(f"Error fetching unemlpoyment data: {e}")
            data_sources['unemployment_rate'] = pd.DataFrame()

        try:
            mb_instance = Macrobond()
            iip_gdp = mb_instance.FetchSeries(list(config.iip_gdp_tickers.keys()))
            iip_gdp = iip_gdp.rename(columns=config.COLUMN_MAPPINGS['iip_gdp'])
            iip_gdp = iip_gdp.resample('M').mean()
            data_sources['iip_gdp'] = iip_gdp
        except Exception as e:
            logger.error(f"Error fetching unemlpoyment data: {e}")
            data_sources['iip_gdp'] = pd.DataFrame()

        # 5. Fetch credit rating data and calculate consolidated risk rating
        try:
            # Fetch Moody's, Fitch, and S&P ratings
            mb_instance = Macrobond()
            m_rating = mb_instance.FetchSeries(list(config.moodys_rating_tickers.keys()))
            m_rating.index.name = "Date"
            m_rating = m_rating.rename(columns=config.COLUMN_MAPPINGS['moody_ratings'])
            m_rating = m_rating.resample('M').mean()
            
            f_rating = mb_instance.FetchSeries(list(config.fitch_rating_tickers.keys()))
            f_rating.index.name = "Date"
            f_rating = f_rating.rename(columns=config.COLUMN_MAPPINGS['fitch_ratings'])
            f_rating = f_rating.resample('M').mean()
            
            s_rating = mb_instance.FetchSeries(list(config.sp_rating_tickers.keys()))
            s_rating.index.name = "Date"
            s_rating = s_rating.rename(columns=config.COLUMN_MAPPINGS['sp_ratings'])
            s_rating = s_rating.resample('M').mean()
            
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
                data_sources['risk_rating'] = risk_rating
            else:
                logger.warning("Cannot calculate risk rating: one or more rating agencies' data is missing")
                data_sources['risk_rating'] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching credit rating data: {e}")
            data_sources['risk_rating'] = pd.DataFrame()
        

        # 6. Collect debt as a share of GDP data
        # try:
        #     mb_instance = Macrobond()
        #     debt_gdp = mb_instance.FetchSeries(list(config.debt_gdp_tickers.keys()))
        #     debt_gdp = debt_gdp.rename(columns=config.COLUMN_MAPPINGS['debt_gdp_tickers'])
        #     debt_gdp = debt_gdp.resample('M').mean().bfill().ffill()
        #     data_sources['debt_gdp'] = debt_gdp
        # except Exception as e:
        #     logger.error(f"Error fetching inflation data: {e}")
        #     data_sources['debt_gdp'] = pd.DataFrame()
        
        # logger.info("Successfully fetched all data sources")
        
    except Exception as e:
        logger.error(f"Error in fetch_all_data_sources: {e}")
        raise
    
    return data_sources

def forward_fill_to_current_date(df, freq='M'):
    """
    Forward-fill a DataFrame to the current date
    
    Parameters:
        df: DataFrame - DataFrame to forward-fill
        freq: str - Frequency for date_range ('M' for monthly, 'D' for daily)
        
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

# Helper function to select the best model based on test RMSE
def select_best_model(model_results_dict, country, tenor_name, output_dir=None):
    """
    Select the best model based on a comprehensive set of evaluation criteria.
    Now also saves the best model to disk.
    
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
        output_dir = config.MODEL_DIR
        
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
    model_scores = {}
    
    # Define weights for different metrics (adjust as needed)
    weights = {
        'test_rmse': 1.0,        # Primary metric (lower is better)
        'test_r2': 0.8,          # Secondary metric (higher is better)
        'train_rmse': 0.5,       # Training error (lower is better)
        'overfitting': 0.7,      # Difference between train and test error (lower is better)
        'simplicity': 0.3,       # Model complexity penalty (higher is better)
        'economic_validity': 0.6, # How well model aligns with economic theory
    }
    
    # Model complexity rankings (lower is simpler/better)
    complexity_ranking = {
        'elasticnet': 1,    # Linear models are simplest
        'ridge': 1,
        'lasso': 1,
        'nelson_siegel': 2, # Domain-specific models
        'gbm': 3,           # Tree-based models
        'randomforest': 3,
        'xgboost': 3,
        'mlp': 4,           # Neural networks
        'gp': 4,            # Gaussian Process
        'ensemble': 5       # Ensemble methods most complex
    }
    
    # Calculate scores for each model
    for model_type, result in successful_models.items():
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
        
        
        # Calculate weighted score
        total_score = sum(score_components[metric] * weights[metric] for metric in score_components)
        
        # Store for comparison
        model_scores[model_type] = {
            'total_score': total_score,
            'components': score_components
        }
        
        logger.info(f"Model {model_type} score: {total_score:.4f} (components: {score_components})")
    
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
        
        # SAVE THE BEST MODEL - NEW ADDITION
        best_model_data = model_results_dict[best_model_type]

        # Extra verification of feature names
        feature_names = extract_feature_names(best_model_data, country, tenor_name)
        
        # Save the model
        if 'model' in best_model_data and 'scaler' in best_model_data:
            model_filename = f"{country}_{tenor_name}_best_model.pkl"
            model_path = os.path.join(output_dir, model_filename)

            print(best_model_data)
            
            # Get feature names if available
            if 'feature_details' in best_model_data:
                if 'feature_columns' in best_model_data['feature_details']:
                    feature_names = best_model_data['feature_details']['feature_columns']


            print(feature_names)
            
            # Create a package with all necessary components
            model_package = {
                'model': best_model_data['model'],
                'scaler': best_model_data['scaler'],
                'model_type': best_model_type,
                'performance': best_model_data.get('metrics', {}),
                'feature_names': feature_names,
                'creation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            model_package['feature_names'] = feature_names
            
            # Save using joblib
            try:
                joblib.dump(model_package, model_path)
                logger.info(f"Saved best model ({best_model_type}) for {country} - {tenor_name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving best model: {str(e)}")
        else:
            logger.warning(f"Cannot save best model ({best_model_type}) - missing required components")
        
        return best_model_type
    else:
        logger.warning("Could not calculate scores for any models")
        return None