"""
Advanced model evaluation utilities for yield curve modeling.
This module contains functions for evaluating model performance and
analyzing residuals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import logging
import matplotlib.dates as mdates
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def evaluate_model_performance(model, X, y, scaler=None, model_name=None, output_dir=None):
    """
    Comprehensive evaluation of model performance with multiple metrics.
    
    Parameters:
        model: Trained model with predict method
        X (DataFrame): Feature data
        y (Series): Target data
        scaler: Optional scaler for X
        model_name (str): Name of the model for reporting
        output_dir (str): Directory to save visualization outputs
        
    Returns:
        dict: Performance metrics
    """
    if model is None or X is None or X.empty or y is None or len(y) == 0:
        logger.error("Cannot evaluate model with empty data")
        return None
    
    # Prepare model name for reporting
    if model_name is None:
        model_name = type(model).__name__
    
    logger.info(f"Evaluating performance for model: {model_name}")
    
    # Scale features if needed
    X_eval = X.copy()
    if scaler is not None:
        X_eval = scaler.transform(X_eval)
    
    # Generate predictions
    predictions = model.predict(X_eval)
    
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'actual': y,
        'predicted': predictions,
        'error': y - predictions,
        'abs_error': np.abs(y - predictions),
        'squared_error': (y - predictions) ** 2
    })
    
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        results_df.index = y.index
    
    # Calculate common metrics
    metrics = {}
    
    # Root Mean Squared Error (RMSE)
    metrics['rmse'] = np.sqrt(mean_squared_error(y, predictions))
    
    # Mean Absolute Error (MAE)
    metrics['mae'] = mean_absolute_error(y, predictions)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    non_zero_mask = y != 0
    if non_zero_mask.any():
        metrics['mape'] = mean_absolute_percentage_error(
            y[non_zero_mask], predictions[non_zero_mask]
        ) * 100  # Convert to percentage
    else:
        metrics['mape'] = np.nan
    
    # Coefficient of Determination (R²)
    metrics['r2'] = r2_score(y, predictions)
    
    # Adjusted R² (accounting for number of features)
    n = len(y)
    p = X.shape[1]
    metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
    
    # Residual statistics
    residuals = y - predictions
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    
    # Additional error statistics
    metrics['mean_error'] = np.mean(results_df['error'])
    metrics['median_error'] = np.median(results_df['error'])
    metrics['max_abs_error'] = np.max(results_df['abs_error'])
    metrics['error_std'] = np.std(results_df['error'])
    
    # Percentage of predictions within different error bounds
    for threshold in [0.1, 0.25, 0.5, 1.0]:
        within_threshold = (results_df['abs_error'] <= threshold).mean() * 100
        metrics[f'within_{threshold}'] = within_threshold
    
    # Log key metrics
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"R²: {metrics['r2']:.4f} (Adjusted R²: {metrics['adj_r2']:.4f})")
    logger.info(f"Mean Error: {metrics['mean_error']:.4f}")
    logger.info(f"% Within 0.25: {metrics['within_0.25']:.2f}%")
    
    # Create visualizations if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{model_name}_actual_vs_predicted.png'))
        plt.close()
        
        # Residuals plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residuals Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{model_name}_residuals.png'))
        plt.close()
        
        # Time series of actual vs predicted if data has datetime index
        if isinstance(results_df.index, pd.DatetimeIndex):
            plt.figure(figsize=(12, 6))
            plt.plot(results_df.index, results_df['actual'], label='Actual', linewidth=2)
            plt.plot(results_df.index, results_df['predicted'], label='Predicted', linewidth=2, linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f'{model_name}: Actual vs Predicted Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_time_series.png'))
            plt.close()
            
            # Error over time
            plt.figure(figsize=(12, 6))
            plt.plot(results_df.index, results_df['error'], color='red')
            plt.axhline(y=0, color='k', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Error')
            plt.title(f'{model_name}: Prediction Error Over Time')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_error_time_series.png'))
            plt.close()
        
        # Error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{model_name}_error_distribution.png'))
        plt.close()
        
        # Q-Q plot for normality check
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name}: Q-Q Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{model_name}_qq_plot.png'))
        plt.close()
        
        # Save results DataFrame to CSV
        results_df.to_csv(os.path.join(output_dir, f'{model_name}_predictions.csv'))
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                      for k, v in metrics.items()}, f, indent=4)
    
    return metrics, results_df

def perform_residual_analysis(results_df, model_name=None, output_dir=None):
    """
    Analyze prediction residuals for patterns, autocorrelation, and heteroscedasticity.
    
    Parameters:
        results_df (DataFrame): DataFrame containing actual, predicted, and error columns
        model_name (str): Name of the model for reporting
        output_dir (str): Directory to save visualization outputs
        
    Returns:
        dict: Analysis results
    """
    if results_df is None or 'error' not in results_df.columns:
        logger.error("Invalid results DataFrame for residual analysis")
        return None
    
    # Prepare model name for reporting
    if model_name is None:
        model_name = "Model"
    
    logger.info(f"Performing residual analysis for {model_name}")
    
    residuals = results_df['error']
    
    # Initialize results dictionary
    analysis = {}
    
    # Basic statistics
    analysis['mean'] = residuals.mean()
    analysis['std'] = residuals.std()
    analysis['min'] = residuals.min()
    analysis['max'] = residuals.max()
    analysis['median'] = residuals.median()
    
    # Normality test (Shapiro-Wilk)
    shapiro_test = stats.shapiro(residuals)
    analysis['shapiro_stat'] = shapiro_test[0]
    analysis['shapiro_p'] = shapiro_test[1]
    analysis['is_normal'] = shapiro_test[1] > 0.05
    
    # Autocorrelation test (Durbin-Watson)
    if isinstance(results_df.index, pd.DatetimeIndex):
        # For time series data
        try:
            dw_stat = sm.stats.stattools.durbin_watson(residuals)
            analysis['durbin_watson'] = dw_stat
            analysis['has_autocorrelation'] = dw_stat < 1.5 or dw_stat > 2.5
        except Exception as e:
            logger.warning(f"Could not calculate Durbin-Watson statistic: {str(e)}")
            analysis['durbin_watson'] = None
            analysis['has_autocorrelation'] = None
        
        # Ljung-Box test for autocorrelation
        try:
            lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10])
            analysis['ljung_box_stat'] = lb_test[0][0]
            analysis['ljung_box_p'] = lb_test[1][0]
            analysis['has_autocorrelation_lb'] = lb_test[1][0] < 0.05
        except Exception as e:
            logger.warning(f"Could not calculate Ljung-Box test: {str(e)}")
            analysis['ljung_box_stat'] = None
            analysis['ljung_box_p'] = None
            analysis['has_autocorrelation_lb'] = None
    
    # Heteroscedasticity test (Breusch-Pagan)
    try:
        # Create model for heteroscedasticity test
        X = sm.add_constant(results_df['predicted'])
        model = sm.OLS(residuals**2, X).fit()
        bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, X)
        
        analysis['bp_stat'] = bp_test[0]
        analysis['bp_p'] = bp_test[1]
        analysis['has_heteroscedasticity'] = bp_test[1] < 0.05
    except Exception as e:
        logger.warning(f"Could not calculate Breusch-Pagan test: {str(e)}")
        analysis['bp_stat'] = None
        analysis['bp_p'] = None
        analysis['has_heteroscedasticity'] = None
    
    # Check for residual patterns
    # Correlation between residuals and predicted values
    analysis['resid_pred_corr'] = np.corrcoef(residuals, results_df['predicted'])[0, 1]
    analysis['has_pattern'] = abs(analysis['resid_pred_corr']) > 0.1
    
    # Log key findings
    logger.info(f"Residual mean: {analysis['mean']:.4f}")
    logger.info(f"Normality: {'Yes' if analysis['is_normal'] else 'No'} (p={analysis['shapiro_p']:.4f})")
    
    if 'has_autocorrelation' in analysis and analysis['has_autocorrelation'] is not None:
        logger.info(f"Autocorrelation: {'Yes' if analysis['has_autocorrelation'] else 'No'} (DW={analysis['durbin_watson']:.4f})")
    
    if 'has_heteroscedasticity' in analysis and analysis['has_heteroscedasticity'] is not None:
        logger.info(f"Heteroscedasticity: {'Yes' if analysis['has_heteroscedasticity'] else 'No'} (p={analysis['bp_p']:.4f})")
    
    logger.info(f"Residual pattern: {'Yes' if analysis['has_pattern'] else 'No'} (corr={analysis['resid_pred_corr']:.4f})")
    
    # Create visualizations if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Autocorrelation plot
        if isinstance(results_df.index, pd.DatetimeIndex):
            plt.figure(figsize=(10, 6))
            plot_acf(residuals, lags=20, alpha=0.05, title=f'{model_name}: Residual Autocorrelation')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{model_name}_autocorrelation.png'))
            plt.close()
            
            # Partial autocorrelation plot
            plt.figure(figsize=(10, 6))
            plot_pacf(residuals, lags=20, alpha=0.05, title=f'{model_name}: Partial Autocorrelation')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{model_name}_partial_autocorr.png'))
            plt.close()
        
        # Save analysis results to JSON
        with open(os.path.join(output_dir, f'{model_name}_residual_analysis.json'), 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.float32, np.float64, np.float_)) else v 
                      for k, v in analysis.items()}, f, indent=4)
    
    return analysis

def evaluate_forecast_accuracy(forecasts, actuals, horizons=None, model_name=None, output_dir=None):
    """
    Evaluate the accuracy of forecasts at different horizons.
    
    Parameters:
        forecasts (DataFrame): Forecasted values with datetime index
        actuals (Series): Actual values with datetime index
        horizons (list): List of forecast horizons to evaluate (in periods)
        model_name (str): Name of the model for reporting
        output_dir (str): Directory to save visualization outputs
        
    Returns:
        DataFrame: Forecast accuracy metrics by horizon
    """
    if forecasts is None or actuals is None:
        logger.error("Cannot evaluate forecast accuracy with empty data")
        return None
    
    # Prepare model name for reporting
    if model_name is None:
        model_name = "Model"
    
    # Default horizons if not specified
    if horizons is None:
        horizons = [1, 3, 6, 12]  # 1-month, 3-month, 6-month, 1-year horizons
    
    logger.info(f"Evaluating forecast accuracy for {model_name} at {len(horizons)} horizons")
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=horizons, columns=['rmse', 'mae', 'mape', 'r2'])
    
    # Evaluate each horizon
    for h in horizons:
        # Shift actuals to align with forecasts at horizon h
        shifted_actuals = actuals.shift(-h)
        
        # Find common indices
        common_idx = forecasts.index.intersection(shifted_actuals.dropna().index)
        
        if len(common_idx) > 0:
            # Extract values for this horizon
            forecast_values = forecasts.loc[common_idx].values
            actual_values = shifted_actuals.loc[common_idx].values
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
            mae = mean_absolute_error(actual_values, forecast_values)
            
            # Avoid division by zero in MAPE
            non_zero_mask = actual_values != 0
            if non_zero_mask.any():
                mape = np.mean(np.abs((actual_values[non_zero_mask] - forecast_values[non_zero_mask]) / 
                                    actual_values[non_zero_mask])) * 100
            else:
                mape = np.nan
            
            r2 = r2_score(actual_values, forecast_values)
            
            # Store results
            results.loc[h, 'rmse'] = rmse
            results.loc[h, 'mae'] = mae
            results.loc[h, 'mape'] = mape
            results.loc[h, 'r2'] = r2
            results.loc[h, 'n_observations'] = len(common_idx)
            
            logger.info(f"Horizon {h} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        else:
            logger.warning(f"No common data points for horizon {h}")
            results.loc[h, 'rmse'] = np.nan
            results.loc[h, 'mae'] = np.nan
            results.loc[h, 'mape'] = np.nan
            results.loc[h, 'r2'] = np.nan
            results.loc[h, 'n_observations'] = 0
    
    # Create visualizations if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot metrics by horizon
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE
        plt.subplot(2, 2, 1)
        plt.plot(results.index, results['rmse'], 'o-', linewidth=2)
        plt.xlabel('Forecast Horizon')
        plt.ylabel('RMSE')
        plt.title('RMSE by Horizon')
        plt.grid(True, alpha=0.3)
        
        # Plot MAE
        plt.subplot(2, 2, 2)
        plt.plot(results.index, results['mae'], 'o-', linewidth=2)
        plt.xlabel('Forecast Horizon')
        plt.ylabel('MAE')
        plt.title('MAE by Horizon')
        plt.grid(True, alpha=0.3)
        
        # Plot MAPE
        plt.subplot(2, 2, 3)
        plt.plot(results.index, results['mape'], 'o-', linewidth=2)
        plt.xlabel('Forecast Horizon')
        plt.ylabel('MAPE (%)')
        plt.title('MAPE by Horizon')
        plt.grid(True, alpha=0.3)
        
        # Plot R²
        plt.subplot(2, 2, 4)
        plt.plot(results.index, results['r2'], 'o-', linewidth=2)
        plt.xlabel('Forecast Horizon')
        plt.ylabel('R²')
        plt.title('R² by Horizon')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_forecast_accuracy.png'))
        plt.close()
        
        # Save results to CSV
        results.to_csv(os.path.join(output_dir, f'{model_name}_forecast_accuracy.csv'))
    
    return results

def compare_models(models_results, X=None, y=None, output_dir=None, economic_metrics=None):
    """
    Enhanced function to compare performance of multiple models with additional metrics.
    
    Parameters:
        models_results (dict): Dictionary of model results, where keys are model names
                              and values are dictionaries of performance metrics
        X (DataFrame): Optional feature data for additional testing
        y (Series): Optional target data for additional testing
        output_dir (str): Directory to save visualization outputs
        economic_metrics (dict): Optional economic relationship metrics
        
    Returns:
        DataFrame: Comprehensive comparison of model performance
    """
    if not models_results:
        logger.error("Cannot compare models with empty results")
        return None
    
    logger.info(f"Comparing {len(models_results)} models")
    
    # Initialize comparison DataFrame
    metrics_to_compare = [
        'rmse', 'mae', 'r2', 'adj_r2', 'mape',     # Standard metrics
        'within_0.25', 'within_0.5',               # Error bounds
        'residual_mean', 'residual_std'            # Residual analysis
    ]
    
    columns = metrics_to_compare + ['model_type', 'complexity', 'economic_validity']
    
    # Initialize comparison DataFrame
    comparison = pd.DataFrame(index=models_results.keys(), columns=columns)
    
    # Extract metrics for each model
    for model_name, results in models_results.items():
        for metric in metrics_to_compare:
            if metric in results:
                comparison.loc[model_name, metric] = results[metric]
            else:
                comparison.loc[model_name, metric] = np.nan
    
    # Rank models by RMSE (lower is better)
    comparison['rmse_rank'] = comparison['rmse'].rank()
    
    # Rank models by R² (higher is better)
    comparison['r2_rank'] = comparison['r2'].rank(ascending=False)
    
    # Calculate average rank
    comparison['avg_rank'] = (comparison['rmse_rank'] + comparison['r2_rank']) / 2
    
    # Sort by average rank
    comparison = comparison.sort_values('avg_rank')

    complexity_map = {
        'ElasticNet': 1,
        'Ridge': 1,
        'Lasso': 1,
        'YieldCurveNelsonSiegel': 2,
        'GradientBoostingRegressor': 3,
        'RandomForestRegressor': 3,
        'XGBRegressor': 3,
        'MLPRegressor': 4,
        'GaussianProcessYieldModel': 4,
        'EnsembleModel': 5,
        'StackingEnsemble': 5
    }
    
    # Extract metrics for each model
    for model_name, results in models_results.items():
        # Get model type if available
        model_type = (
            type(results.get('model')).__name__ 
            if results.get('model') is not None 
            else results.get('model_type', 'Unknown')
        )
        comparison.loc[model_name, 'model_type'] = model_type
        
        # Assign complexity score
        comparison.loc[model_name, 'complexity'] = complexity_map.get(model_type, 3)
        
        # Extract standard metrics
        for metric in metrics_to_compare:
            if metric in results:
                comparison.loc[model_name, metric] = results[metric]
            elif 'metrics' in results and metric in results['metrics']:
                comparison.loc[model_name, metric] = results['metrics'][metric]
            elif 'metrics' in results and 'test' in results['metrics'] and metric in results['metrics']['test']:
                comparison.loc[model_name, metric] = results['metrics']['test'][metric]
            else:
                comparison.loc[model_name, metric] = np.nan
    
    # Identify best model
    best_model = comparison.index[0]
    logger.info(f"Best model: {best_model}")
    logger.info(f"Best model RMSE: {comparison.loc[best_model, 'rmse']:.4f}")
    logger.info(f"Best model R²: {comparison.loc[best_model, 'r2']:.4f}")

    economic_validity_score = None
        
        # Use provided economic metrics if available
    if economic_metrics and model_name in economic_metrics:
            economic_validity_score = economic_metrics[model_name]
        # Otherwise try to calculate from feature importance
    elif 'feature_importance' in results and results['feature_importance']:
            feature_imp = results['feature_importance']
            
            # Check for key economic relationships
            expected_relationships = {
                'pol_rat': 1,       # Policy rate should have positive effect
                'cpi_inf': 1,       # Inflation should have positive effect
                'unemployment': -1,  # Unemployment should have negative effect
                'rating': 1         # Risk rating should have positive effect (higher rating = lower risk)
            }
            
            valid_count = 0
            total_count = 0
            
            for prefix, expected_sign in expected_relationships.items():
                matching_features = [f for f in feature_imp.keys() if prefix in f.lower()]
                if matching_features:
                    total_count += 1
                    avg_imp = np.mean([feature_imp[f] for f in matching_features])
                    if (avg_imp > 0 and expected_sign > 0) or (avg_imp < 0 and expected_sign < 0):
                        valid_count += 1
            
            if total_count > 0:
                economic_validity_score = valid_count / total_count
        
        # Store economic validity score
    if economic_validity_score is not None:
            comparison.loc[model_name, 'economic_validity'] = economic_validity_score
    else:
            comparison.loc[model_name, 'economic_validity'] = np.nan

    # Calculate rank for multiple metrics with weights
    rank_weights = {
        'rmse': 1.0,           # Primary metric
        'r2': 0.8,             # Secondary metric
        'economic_validity': 0.7,  # Economic validity is important
        'complexity': 0.3      # Simplicity bonus (lower complexity is better)
    }
    
    # Calculate ranks
    for metric, weight in rank_weights.items():
        if metric in comparison.columns and not comparison[metric].isna().all():
            if metric in ['rmse', 'mae', 'mape', 'complexity']:
                # For these metrics, lower is better
                comparison[f'{metric}_rank'] = comparison[metric].rank()
            else:
                # For other metrics, higher is better
                comparison[f'{metric}_rank'] = comparison[metric].rank(ascending=False)
    
    # Calculate weighted average rank
    rank_columns = [f'{metric}_rank' for metric in rank_weights.keys() 
                   if f'{metric}_rank' in comparison.columns]
    
    if rank_columns:
        # Filter out NaN values per model
        weighted_ranks = []
        for idx in comparison.index:
            model_ranks = comparison.loc[idx, rank_columns]
            model_weights = [rank_weights[col.replace('_rank', '')] 
                            for col in rank_columns if not pd.isna(model_ranks[col])]
            model_rank_values = [model_ranks[col] for col in rank_columns if not pd.isna(model_ranks[col])]
            
            if model_rank_values and model_weights:
                # Calculate weighted rank
                weighted_rank = sum(r * w for r, w in zip(model_rank_values, model_weights)) / sum(model_weights)
                weighted_ranks.append((idx, weighted_rank))
            else:
                weighted_ranks.append((idx, float('inf')))
        
        # Assign weighted ranks
        for idx, rank in weighted_ranks:
            comparison.loc[idx, 'weighted_rank'] = rank
        
        # Sort by weighted rank
        comparison = comparison.sort_values('weighted_rank')
    else:
        # Fallback to sorting by RMSE if no ranks calculated
        comparison = comparison.sort_values('rmse')

    # Identify multiple "best" models for different criteria
    best_models = {}
    
    # Best overall model (by weighted rank)
    if 'weighted_rank' in comparison.columns:
        best_models['overall'] = comparison.index[0]
    # Best by RMSE
    if 'rmse' in comparison.columns and not comparison['rmse'].isna().all():
        best_models['rmse'] = comparison['rmse'].idxmin()
    # Best by R²
    if 'r2' in comparison.columns and not comparison['r2'].isna().all():
        best_models['r2'] = comparison['r2'].idxmax()
    # Most economically valid
    if 'economic_validity' in comparison.columns and not comparison['economic_validity'].isna().all():
        best_models['economic_validity'] = comparison['economic_validity'].idxmax()
    # Simplest good model (top 3 performance but lowest complexity)
    if 'rmse' in comparison.columns and 'complexity' in comparison.columns:
        top3 = comparison.nsmallest(3, 'rmse')
        if not top3.empty:
            best_models['simplest_good'] = top3['complexity'].idxmin()
    
    # Log best models
    logger.info("Best models by category:")
    for category, model_name in best_models.items():
        logger.info(f"  Best {category}: {model_name}")
        if category in ['overall', 'rmse']:
            if 'rmse' in comparison.columns:
                logger.info(f"    RMSE: {comparison.loc[model_name, 'rmse']:.4f}")
            if 'r2' in comparison.columns:
                logger.info(f"    R²: {comparison.loc[model_name, 'r2']:.4f}")
        
    # Add a model recommendation based on all factors
    final_recommendation = best_models.get('overall', best_models.get('rmse', comparison.index[0]))
    logger.info(f"Final recommended model: {final_recommendation}")
    
    # Create visualizations if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create bar charts for each metric
        for metric in metrics_to_compare:
            if metric in comparison.columns and not comparison[metric].isna().all():
                plt.figure(figsize=(10, 6))
                
                # Sort by the current metric
                if metric in ['rmse', 'mae', 'mape', 'residual_std']:
                    # Lower is better
                    sorted_df = comparison.sort_values(metric)
                    color_map = plt.cm.RdYlGn_r
                else:
                    # Higher is better
                    sorted_df = comparison.sort_values(metric, ascending=False)
                    color_map = plt.cm.RdYlGn
                
                # Plot bar chart
                bars = plt.bar(sorted_df.index, sorted_df[metric], color=color_map(np.linspace(0, 1, len(sorted_df))))
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.4f}',
                             ha='center', va='bottom', rotation=0)
                
                plt.xlabel('Model')
                plt.ylabel(metric.upper())
                plt.title(f'Model Comparison by {metric.upper()}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.grid(True, alpha=0.3, axis='y')
                plt.savefig(os.path.join(output_dir, f'model_comparison_{metric}.png'))
                plt.close()
        
        # [MODIFICATION: ENHANCED VISUALIZATION AND REPORTING]
        # Create a radar chart comparing top models
        try:
            if len(comparison) >= 2:
                top_models = comparison.head(min(5, len(comparison)))
                
                # Select metrics for radar chart
                radar_metrics = ['rmse', 'r2', 'economic_validity']
                radar_metrics = [m for m in radar_metrics if m in comparison.columns]
                
                if radar_metrics:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # Normalize metrics to 0-1 range for radar chart
                    normalized_data = {}
                    for metric in radar_metrics:
                        data = top_models[metric].values
                        if metric in ['rmse', 'mae', 'mape', 'complexity']:
                            # For metrics where lower is better, invert values
                            if np.max(data) != np.min(data):  # Avoid division by zero
                                normalized_data[metric] = 1 - (data - np.min(data)) / (np.max(data) - np.min(data))
                            else:
                                normalized_data[metric] = np.ones_like(data)
                        else:
                            # For metrics where higher is better
                            if np.max(data) != np.min(data):  # Avoid division by zero
                                normalized_data[metric] = (data - np.min(data)) / (np.max(data) - np.min(data))
                            else:
                                normalized_data[metric] = np.ones_like(data)
                    
                    # Number of metrics
                    N = len(radar_metrics)
                    
                    # Create angles for radar chart
                    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
                    # Close the circle
                    angles += angles[:1]
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                    
                    # Add data for each model
                    for i, model_name in enumerate(top_models.index):
                        values = [normalized_data[metric][i] for metric in radar_metrics]
                        # Close the loop
                        values += values[:1]
                        
                        # Plot the model
                        ax.plot(angles, values, linewidth=2, label=model_name)
                        ax.fill(angles, values, alpha=0.1)
                    
                    # Add metric labels
                    plt.xticks(angles[:-1], radar_metrics, fontsize=12)
                    
                    # Add legend
                    plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.1))
                    
                    plt.title('Model Comparison Radar Chart', fontsize=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'model_comparison_radar.png'))
                    plt.close()
                    
                    logger.info(f"Created radar chart visualization at {os.path.join(output_dir, 'model_comparison_radar.png')}")
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
        
        # Create a summary table figure
        try:
            # Select key metrics for summary table
            summary_metrics = ['rmse', 'r2', 'economic_validity', 'complexity', 'weighted_rank']
            summary_metrics = [m for m in summary_metrics if m in comparison.columns]
            
            if summary_metrics:
                # Create a summary table for top models
                top_models_summary = comparison[summary_metrics].head(min(8, len(comparison)))
                
                # Format the table data
                cell_text = []
                for i, idx in enumerate(top_models_summary.index):
                    row = top_models_summary.loc[idx].tolist()
                    # Format numbers
                    formatted_row = []
                    for j, val in enumerate(row):
                        if summary_metrics[j] in ['rmse', 'r2', 'economic_validity', 'weighted_rank']:
                            formatted_row.append(f"{val:.4f}" if pd.notnull(val) else "N/A")
                        else:
                            formatted_row.append(f"{val:.1f}" if pd.notnull(val) else "N/A")
                    cell_text.append(formatted_row)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, len(top_models_summary) * 0.5 + 2))
                ax.axis('off')
                
                # Create table
                table = plt.table(
                    cellText=cell_text,
                    rowLabels=top_models_summary.index,
                    colLabels=summary_metrics,
                    cellLoc='center',
                    loc='center'
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Highlight the best model
                if final_recommendation in top_models_summary.index:
                    row_idx = list(top_models_summary.index).index(final_recommendation)
                    for j in range(len(summary_metrics)):
                        cell = table[(row_idx + 1, j)]
                        cell.set_facecolor('#aaffaa')
                
                plt.title('Model Performance Summary', fontsize=15, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'model_comparison_summary.png'))
                plt.close()
                
                logger.info(f"Created summary table visualization at {os.path.join(output_dir, 'model_comparison_summary.png')}")
        except Exception as e:
            logger.error(f"Error creating summary table: {str(e)}")
        
        # [MODIFICATION: SAVE COMPREHENSIVE COMPARISON REPORT]
        # Save all comparison data to CSV with recommendations
        try:
            # Add recommendation flags to comparison dataframe
            for category, model_name in best_models.items():
                comparison[f'best_{category}'] = comparison.index == model_name
            
            comparison['final_recommendation'] = comparison.index == final_recommendation
            
            # Save to CSV
            comparison.to_csv(os.path.join(output_dir, 'model_comparison_full.csv'))
            
            # Create a more readable summary CSV
            summary_df = pd.DataFrame({
                'Model': list(best_models.values()),
                'Category': list(best_models.keys()),
                'RMSE': [comparison.loc[m, 'rmse'] if 'rmse' in comparison.columns else np.nan 
                        for m in best_models.values()],
                'R²': [comparison.loc[m, 'r2'] if 'r2' in comparison.columns else np.nan 
                      for m in best_models.values()],
                'Economic Validity': [comparison.loc[m, 'economic_validity'] if 'economic_validity' in comparison.columns else np.nan 
                                    for m in best_models.values()],
                'Complexity': [comparison.loc[m, 'complexity'] if 'complexity' in comparison.columns else np.nan 
                              for m in best_models.values()]
            })
            
            # Add final recommendation row
            summary_df = pd.concat([
                summary_df,
                pd.DataFrame({
                    'Model': [final_recommendation],
                    'Category': ['FINAL RECOMMENDATION'],
                    'RMSE': [comparison.loc[final_recommendation, 'rmse'] if 'rmse' in comparison.columns else np.nan],
                    'R²': [comparison.loc[final_recommendation, 'r2'] if 'r2' in comparison.columns else np.nan],
                    'Economic Validity': [comparison.loc[final_recommendation, 'economic_validity'] 
                                        if 'economic_validity' in comparison.columns else np.nan],
                    'Complexity': [comparison.loc[final_recommendation, 'complexity'] 
                                  if 'complexity' in comparison.columns else np.nan]
                })
            ])
            
            # Save summary
            summary_df.to_csv(os.path.join(output_dir, 'model_recommendations.csv'), index=False)
            logger.info(f"Saved model recommendations to {os.path.join(output_dir, 'model_recommendations.csv')}")
        except Exception as e:
            logger.error(f"Error saving comparison data: {str(e)}")
    
    # [MODIFICATION: ADD RECOMMENDATION TO RETURN VALUE]
    # Update return value to include the recommendations
    return {
        'comparison_df': comparison,
        'best_models': best_models,
        'recommended_model': final_recommendation
    }