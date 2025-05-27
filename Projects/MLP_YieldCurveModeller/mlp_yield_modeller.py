import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import traceback
import joblib
import datetime

# Project imports
import config
from data_worker import train_evaluate_model, fetch_all_data_sources, save_consolidated_fitted_values, forward_fill_to_current_date, select_best_model
from data_visualizer import create_model_summary_visualization
from overlap_worker import create_data_availability_summary
from model_tester import configure_diagnostic_logging, extract_feature_names
from forecast_generator import get_forecast_data_for_modelling
from sklearn.preprocessing import LabelEncoder
from model_training import build_ensemble_model
from macrobond import Macrobond
import bloomberg

# Module level logger (will be configured in main)
logger = logging.getLogger(__name__)


def initialize_api_clients():
    """
    Initialize and return API client instances.
    
    Returns:
        tuple: (macrobond_client, bloomberg_client, label_encoder)
    """
    try:
        mb = Macrobond()
        logger.info("Macrobond API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Macrobond client: {e}")
        mb = None
    
    try:
        bbg = bloomberg.Bloomberg()
        logger.info("Bloomberg API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Bloomberg client: {e}")
        bbg = None
    
    # Initialize label encoder
    label_encoder = LabelEncoder()
    
    return mb, bbg, label_encoder


def setup_output_directories():
    """
    Create necessary output directories if they don't exist.
    """
    directories = [
        config.MODEL_DIR,
        config.VISUALIZATION_DIR,
        config.OUTPUT_DIR
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created/verified directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


def parse_command_line_arguments():
    """
    Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Yield Curve Modeling Pipeline')
    parser.add_argument('--models', nargs='+', 
                       default=['elasticnet', 'mlp', 'gbm', 'gp'],
                       choices=['mlp', 'enhanced_mlp', 'gbm', 'elasticnet', 'gp', 'ensemble'],
                       help='Model types to train')
    parser.add_argument('--use-ensembles', action='store_true', 
                       help='Create ensembles of different model types')
    parser.add_argument('--countries', nargs='+', 
                       help='Specific countries to process (default: all)')
    parser.add_argument('--tenors', nargs='+', 
                       help='Specific tenors to process (default: all)')
    parser.add_argument('--optimize', action='store_true', 
                       help='Perform hyperparameter optimization')
    parser.add_argument('--compare-models', action='store_true', 
                       help='Compare different model types')
    parser.add_argument('--run-sensitivity', action='store_true', 
                       help='Run sensitivity analysis on trained models')
    parser.add_argument('--sensitivity-output', choices=['csv', 'plot', 'both'], 
                       default='plot', help='Output format for sensitivity analysis')
    parser.add_argument('--shock-amount', type=float, default=0.5, 
                       help='Amount to shock features by in sensitivity analysis')
    parser.add_argument('--shock-type', choices=['absolute', 'relative', 'percentage'], 
                       default='absolute', help='Type of shock to apply in sensitivity analysis')
    
    return parser.parse_args()


def fetch_and_prepare_data(mb_client, bbg_client):
    """
    Fetch and prepare all required data sources.
    
    Parameters:
        mb_client: Macrobond API client
        bbg_client: Bloomberg API client
        
    Returns:
        dict: Dictionary containing all prepared data sources
    """
    logger.info("Fetching bond yield and economic data...")
    
    # Pass API clients to data fetching function
    data_sources = fetch_all_data_sources(mb_client=mb_client, bbg_client=bbg_client)
    
    # Unpack and prepare data sources
    yield_data = data_sources['yield_data']
    pol_rat = forward_fill_to_current_date(data_sources['policy_rates'])
    cpi_inf = forward_fill_to_current_date(data_sources['inflation'])
    act_track = forward_fill_to_current_date(data_sources['activity'])
    risk_rating = forward_fill_to_current_date(data_sources['risk_rating'])
    unemployment_rate = forward_fill_to_current_date(data_sources['unemployment_rate'])
    iip_gdp = forward_fill_to_current_date(data_sources['iip_gdp'])
    
    # Get historical forecasts from forecast_generator
    logger.info("Generating historical forecasts...")
    historical_forecasts = get_forecast_data_for_modelling(
        forecast_horizons=config.FORECAST_HORIZONS,
        start_date=config.DEFAULT_HISTORICAL_FORECAST_START,
        mb_client=mb_client
    )
    
    # Store yield data in lists for easier iteration
    yield_list = [
        yield_data['yld_2yr_ann'], 
        yield_data['yld_5yr_ann'], 
        yield_data['yld_10yr_ann'], 
        yield_data['yld_30yr_ann']
    ]
    
    return {
        'yield_data': yield_data,
        'yield_list': yield_list,
        'pol_rat': pol_rat,
        'cpi_inf': cpi_inf,
        'act_track': act_track,
        'risk_rating': risk_rating,
        'unemployment_rate': unemployment_rate,
        'iip_gdp': iip_gdp,
        'historical_forecasts': historical_forecasts
    }


def run_data_diagnostics(country_list, data_dict, tenor_names):
    """
    Run data diagnostics and create availability summary.
    
    Parameters:
        country_list: List of countries to analyze
        data_dict: Dictionary containing all data sources
        tenor_names: List of tenor names
        
    Returns:
        dict: Data availability summary
    """
    logger.info("Running data diagnostics...")
    
    data_availability = create_data_availability_summary(
        country_list=country_list,
        country_code_mapping=config.country_list_mapping,
        yield_list=data_dict['yield_list'],
        yield_names=tenor_names,
        pol_rat=data_dict['pol_rat'],
        cpi_inf=data_dict['cpi_inf'],
        act_track=data_dict['act_track'],
        risk_rating=data_dict['risk_rating'],
        historical_forecasts=data_dict['historical_forecasts'],
        unemployment_rate=data_dict['unemployment_rate'],
        iip_gdp=data_dict['iip_gdp']
    )
    
    return data_availability


def select_models_for_tenor(tenor_idx, args):
    """
    Select appropriate models for a given tenor.
    
    Parameters:
        tenor_idx: Index of the tenor (0 for 2yr, 1 for 5yr, etc.)
        args: Command line arguments
        
    Returns:
        list: List of model types to try for this tenor
    """
    tenor_model_types = args.models.copy()
    
    # For 2yr tenor, we may want to remove models that require multiple tenor points
    if tenor_idx == 0:
        logger.info("Processing 2yr tenor - using standard model types")
    
    # Remove duplicates while preserving order
    tenor_model_types = list(dict.fromkeys(tenor_model_types))
    
    # Reorder to ensure simpler models are tried first
    if 'elasticnet' in tenor_model_types and tenor_model_types[0] != 'elasticnet':
        tenor_model_types.remove('elasticnet')
        tenor_model_types.insert(0, 'elasticnet')
    
    return tenor_model_types


def train_models_for_tenor(country, tenor_name, tenor_idx, tenor_data, data_dict, 
                          country_predicted_yields, tenor_model_types, args):
    """
    Train all model types for a specific country and tenor.
    
    Parameters:
        country: Country name
        tenor_name: Tenor name (e.g., 'yld_2yr')
        tenor_idx: Tenor index
        tenor_data: Yield data for this tenor
        data_dict: Dictionary containing all data sources
        country_predicted_yields: Previously predicted yields for this country
        tenor_model_types: List of model types to train
        args: Command line arguments
        
    Returns:
        dict: Results for each model type
    """
    model_results_by_type = {}
    use_yield_curve_models = (tenor_idx > 0)
    
    for model_type in tenor_model_types:
        logger.info(f"Training {model_type} model for {country} - {tenor_name}")
        
        try:
            # Train the model
            model_result = train_evaluate_model(
                country=country,
                tenor_name=tenor_name,
                country_code_mapping=config.country_list_mapping,
                tenor_data=tenor_data,
                pol_rat=data_dict['pol_rat'],
                cpi_inf=data_dict['cpi_inf'],
                act_track=data_dict['act_track'],
                model_type=model_type,
                risk_rating=data_dict['risk_rating'],
                historical_forecasts=data_dict['historical_forecasts'],
                unemployment_rate=data_dict['unemployment_rate'],
                iip_gdp=data_dict['iip_gdp'],
                predicted_yields=country_predicted_yields,
                use_advanced_training=True,
                compare_models=args.compare_models,
                use_yield_curve_models=use_yield_curve_models
            )
            
        except Exception as e:
            logger.error(f"Error training {model_type} model for {country} - {tenor_name}: {e}")
            logger.exception("Full traceback:")
            model_result = {
                'status': f'Failed - Exception in {model_type}',
                'error': str(e),
                'metrics': {}
            }
        
        model_results_by_type[model_type] = model_result
    
    return model_results_by_type


def create_ensemble_model(country, tenor_name, model_results_by_type, args):
    """
    Create ensemble model if requested and multiple successful models exist.
    
    Parameters:
        country: Country name
        tenor_name: Tenor name
        model_results_by_type: Dictionary of model results by type
        args: Command line arguments
        
    Returns:
        dict: Updated model results including ensemble if created
    """
    if not args.use_ensembles:
        return model_results_by_type
    
    successful_models = {m_type: res for m_type, res in model_results_by_type.items()
                        if res.get('status', '').startswith('Success')}
    
    if len(successful_models) > 1:
        logger.info(f"Creating ensemble model for {country} - {tenor_name}")
        try:
            ensemble_result = build_ensemble_model(
                country=country,
                tenor_name=tenor_name,
                model_results=successful_models,
                weighted_by_performance=True
            )
            
            model_results_by_type['ensemble'] = ensemble_result
            
        except Exception as e:
            logger.error(f"Error creating ensemble for {country} - {tenor_name}: {e}")
            logger.exception("Full traceback:")
    
    return model_results_by_type


def update_predicted_yields(country, tenor_name, model_result, predicted_yields_by_country):
    """
    Update predicted yields dictionary with successful model results.
    
    Parameters:
        country: Country name
        tenor_name: Tenor name
        model_result: Model result dictionary
        predicted_yields_by_country: Dictionary to update
    """
    if (model_result.get('status', '').startswith('Success') and 
        'predictions' in model_result):
        
        if country not in predicted_yields_by_country:
            predicted_yields_by_country[country] = {}
        
        if tenor_name not in predicted_yields_by_country[country]:
            predicted_yields_by_country[country][tenor_name] = {}
        
        # Map tenor name to years
        tenor_years_map = {
            'yld_2yr': 2, 'yld_5yr': 5, 'yld_10yr': 10, 'yld_30yr': 30
        }
        
        model_type = model_result.get('best_model_type', 'unknown')
        predicted_yields_by_country[country][tenor_name][model_type] = {
            'predictions': model_result['predictions'],
            'tenor_years': tenor_years_map.get(tenor_name, 0),
            'current_value': (model_result['predictions'].iloc[-1] 
                            if not model_result['predictions'].empty else None)
        }


def save_best_models(results_summary):
    """
    Save the best models with proper feature names and metadata.
    
    Parameters:
        results_summary: Dictionary containing all results
    """
    logger.info("Saving best models with explicit feature names")
    
    for country in results_summary['best_models']:
        for tenor, model_type in results_summary['best_models'][country].items():
            try:
                # Get the specific model result
                model_result = results_summary['model_results'][country][tenor]
                
                # Ensure model and scaler exist
                if 'model' not in model_result or 'scaler' not in model_result:
                    logger.warning(f"Missing model or scaler for {country} - {tenor}")
                    continue
                
                # Create output directory
                country_dir = os.path.join(config.MODEL_DIR, country)
                os.makedirs(country_dir, exist_ok=True)
                
                # Construct model path
                model_path = os.path.join(country_dir, f"{country}_{tenor}_best_model.pkl")
                
                # Extract feature names
                try:
                    feature_names = extract_feature_names(model_result, country, tenor)
                except Exception as e:
                    logger.warning(f"Could not extract feature names for {country} - {tenor}: {e}")
                    # Fallback to feature_details if available
                    feature_names = model_result.get('feature_details', {}).get('feature_columns', [])
                
                # Create model package
                model_package = {
                    'model': model_result['model'],
                    'scaler': model_result['scaler'],
                    'model_type': model_type,
                    'feature_names': feature_names,
                    'creation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'performance': model_result.get('metrics', {}),
                    'country': country,
                    'tenor': tenor
                }
                
                # Save model package
                joblib.dump(model_package, model_path)
                logger.info(f"Saved best model for {country} - {tenor}: {model_type}")
                
                # Log feature names for verification
                if feature_names:
                    logger.debug(f"Feature names for {country} - {tenor}: {feature_names}")
                else:
                    logger.warning(f"No feature names found for {country} - {tenor}")
                    
            except Exception as e:
                logger.error(f"Error saving best model for {country} - {tenor}: {e}")
                logger.exception("Full traceback:")


def generate_summary_outputs(results_summary, models_df, country_list):
    """
    Generate summary outputs and visualizations.
    
    Parameters:
        results_summary: Dictionary containing all results
        models_df: List of model result dictionaries
        country_list: List of countries processed
    """
    if models_df:
        # Convert results to DataFrame and save
        models_results_df = pd.DataFrame(models_df)
        
        # Create output directory for each country
        for country in country_list:
            country_output_dir = os.path.join(config.OUTPUT_DIR, country)
            os.makedirs(country_output_dir, exist_ok=True)
            
            # Filter results for this country
            country_results = models_results_df[models_results_df['Country'] == country]
            if not country_results.empty:
                model_summary_path = os.path.join(country_output_dir, 'model_results_summary.csv')
                country_results.to_csv(model_summary_path, index=False)
                logger.info(f"Model results summary saved for {country}")
        
        # Save overall summary
        overall_summary_path = os.path.join(config.OUTPUT_DIR, 'overall_model_results_summary.csv')
        models_results_df.to_csv(overall_summary_path, index=False)
        logger.info(f"Overall model results summary saved to '{overall_summary_path}'")
        
        # Create summary visualization
        try:
            create_model_summary_visualization(results_summary)
            logger.info("Created model summary visualization")
        except Exception as e:
            logger.error(f"Error creating model summary visualization: {e}")
            logger.exception("Full traceback:")


def main():
    """
    Enhanced main function for yield curve modeling and forecasting pipeline.
    Supports multiple model types, ensembles, and smarter model selection strategies.
    """
    # Single point of logging configuration
    logger_instance = config.configure_logging()
    logger.info("Starting enhanced yield curve modeling and forecasting pipeline")
    
    try:
        # Setup output directories
        setup_output_directories()
        
        # Initialize API clients once at the start
        mb_client, bbg_client, label_encoder = initialize_api_clients()
        
        if mb_client is None:
            logger.error("Failed to initialize Macrobond client - some functionality may be limited")
        if bbg_client is None:
            logger.error("Failed to initialize Bloomberg client - some functionality may be limited")
        
        # Parse command line arguments
        args = parse_command_line_arguments()
        
        # Use specified countries or all configured countries
        country_list = args.countries if args.countries else config.country_list
        
        # Use specified tenors or all configured tenors
        tenor_names = args.tenors if args.tenors else ['yld_2yr', 'yld_5yr', 'yld_10yr', 'yld_30yr']
        
        # Initialize results tracking
        results_summary = {
            'data_overlap': None,
            'data_availability': None,
            'model_results': {},
            'success_count': 0,
            'failure_count': 0,
            'best_models': {}
        }
        
        # Fetch and prepare all data
        data_dict = fetch_and_prepare_data(mb_client, bbg_client)
        
        # Run data diagnostics
        data_availability = run_data_diagnostics(country_list, data_dict, tenor_names)
        results_summary['data_availability'] = data_availability
        
        # Initialize tracking variables
        models_df = []
        predicted_yields_by_country = {}
        
        # Train and evaluate models for each country and tenor
        logger.info("Training and evaluating models in order of maturity...")
        
        for country in country_list:
            logger.info(f"Processing country: {country}")
            
            # Initialize results for this country
            results_summary['model_results'][country] = {}
            predicted_yields_by_country[country] = {}
            results_summary['best_models'][country] = {}
            
            # Process tenors in order from shortest to longest
            for tenor_idx, tenor_name in enumerate(tenor_names):
                if tenor_idx < len(data_dict['yield_list']):
                    tenor_data = data_dict['yield_list'][tenor_idx]
                else:
                    logger.error(f"No data found for tenor {tenor_name}")
                    continue
                
                logger.info(f"Processing {country} - {tenor_name}")
                
                # Select models for this tenor
                tenor_model_types = select_models_for_tenor(tenor_idx, args)
                logger.info(f"Will try these model types for {country} - {tenor_name}: {tenor_model_types}")
                
                # Get previously predicted yields for this country
                country_predicted_yields = predicted_yields_by_country.get(country, {})
                
                # Train models for this tenor
                model_results_by_type = train_models_for_tenor(
                    country, tenor_name, tenor_idx, tenor_data, data_dict,
                    country_predicted_yields, tenor_model_types, args
                )
                
                # Create ensemble if requested
                model_results_by_type = create_ensemble_model(
                    country, tenor_name, model_results_by_type, args
                )
                
                # Select the best model
                logger.info(f"Selecting best model for {country} - {tenor_name} from {len(model_results_by_type)} trained models")
                
                best_model_type = select_best_model(
                    model_results_by_type,
                    country=country,
                    tenor_name=tenor_name,
                    output_dir=os.path.join(config.MODEL_DIR, country)
                )
                
                if best_model_type:
                    logger.info(f"Best model for {country} - {tenor_name}: {best_model_type}")
                    
                    # Store the best model's results
                    best_result = model_results_by_type[best_model_type]
                    best_result['best_model_type'] = best_model_type
                    results_summary['model_results'][country][tenor_name] = best_result
                    results_summary['best_models'][country][tenor_name] = best_model_type
                    
                    # Update predicted yields
                    update_predicted_yields(country, tenor_name, best_result, predicted_yields_by_country)
                    
                    # Update success count
                    results_summary['success_count'] += 1
                    
                else:
                    logger.warning(f"No successful models for {country} - {tenor_name}")
                    
                    # Store fallback result
                    if model_results_by_type:
                        first_model_type = list(model_results_by_type.keys())[0]
                        results_summary['model_results'][country][tenor_name] = model_results_by_type[first_model_type]
                    else:
                        results_summary['model_results'][country][tenor_name] = {
                            'status': 'Failed - No Models Trained',
                            'error': 'No models were successfully trained',
                            'metrics': {}
                        }
                    
                    results_summary['failure_count'] += 1
                
                # Add results to summary DataFrame
                for model_type, model_result in model_results_by_type.items():
                    models_df.append({
                        'Country': country,
                        'Tenor': tenor_name,
                        'Model_Type': model_type,
                        'Status': model_result.get('status', 'Unknown'),
                        'Train_RMSE': model_result.get('metrics', {}).get('train', {}).get('rmse'),
                        'Train_R2': model_result.get('metrics', {}).get('train', {}).get('r2'),
                        'Test_RMSE': model_result.get('metrics', {}).get('test', {}).get('rmse'),
                        'Test_R2': model_result.get('metrics', {}).get('test', {}).get('r2'),
                        'Is_Best_Model': model_type == best_model_type,
                        'Error': model_result.get('error')
                    })
        
        # Generate summary outputs
        generate_summary_outputs(results_summary, models_df, country_list)
        
        # Save best models
        save_best_models(results_summary)
        
        # Log final statistics
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Total models: {results_summary['success_count'] + results_summary['failure_count']}")
        logger.info(f"Successful models: {results_summary['success_count']}")
        logger.info(f"Failed models: {results_summary['failure_count']}")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Critical error in main pipeline: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()