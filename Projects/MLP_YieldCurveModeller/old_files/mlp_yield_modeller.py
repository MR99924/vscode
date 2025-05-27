import sys
import os
import pandas as pd
import config
from data_worker import train_evaluate_model, fetch_all_data_sources, save_consolidated_fitted_values, forward_fill_to_current_date, select_best_model
from data_visualizer import create_model_summary_visualization
from overlap_worker import create_data_availability_summary
from model_tester import configure_diagnostic_logging, extract_feature_names
from forecast_generator import get_forecast_data_for_modelling
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model_training import build_ensemble_model
from model_classes import save_model_package
import matplotlib.pyplot as plt
import seaborn as sns
from macrobond import Macrobond
import logging
import argparse
import traceback
import joblib, datetime

sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\MLP_YieldCurveModeller')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize API connectors
mb = Macrobond()
label_encoder = LabelEncoder()

def main():
    """
    Enhanced main function for yield curve modeling and forecasting pipeline.
    Supports multiple model types, ensembles, and smarter model selection strategies.
    """
    # Configure logging
    logger = config.configure_logging()
    logger.info("Starting enhanced yield curve modeling and forecasting pipeline")
    
    # Create directories for outputs if they don't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    
    # Add command-line argument parsing for flexibility
    parser = argparse.ArgumentParser(description='Yield Curve Modeling Pipeline')
    parser.add_argument('--models', nargs='+', 
                  default=['elasticnet', 'mlp', 'gbm', 'gp'],#, 'nelson_siegel', 'xgboost'],
                  choices=['mlp', 'enhanced_mlp', 'gbm', 'elasticnet', 'gp', 'ensemble'],#'nelson_seigel', 'xgboost'],
                  help='Model types to train')
    parser.add_argument('--use-ensembles', action='store_true', help='Create ensembles of different model types')
    parser.add_argument('--countries', nargs='+', help='Specific countries to process (default: all)')
    parser.add_argument('--tenors', nargs='+', help='Specific tenors to process (default: all)')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--compare-models', action='store_true', help='Compare different model types')
    parser.add_argument('--run-sensitivity', action='store_true', help='Run sensitivity analysis on trained models')
    parser.add_argument('--sensitivity-output', choices=['csv', 'plot', 'both'], default='plot', 
                    help='Output format for sensitivity analysis')
    parser.add_argument('--shock-amount', type=float, default=0.5, help='Amount to shock features by in sensitivity analysis')
    parser.add_argument('--shock-type', choices=['absolute', 'relative', 'percentage'], default='absolute',
                    help='Type of shock to apply in sensitivity analysis')
    args = parser.parse_args()
    
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
    
    try:
        # Step 1: Fetch all required data
        logger.info("Fetching bond yield and economic data...")
        data_sources = fetch_all_data_sources()
        
        # Unpack data sources
        yield_data = data_sources['yield_data']
        pol_rat = forward_fill_to_current_date(data_sources['policy_rates'])
        cpi_inf = forward_fill_to_current_date(data_sources['inflation'])
        act_track = forward_fill_to_current_date(data_sources['activity'])
        risk_rating = forward_fill_to_current_date(data_sources['risk_rating'])
        unemployment_rate = forward_fill_to_current_date(data_sources['unemployment_rate'])
        iip_gdp = forward_fill_to_current_date(data_sources['iip_gdp'])
        
        # Step 2: Get historical forecasts from forecast_generator
        logger.info("Generating historical forecasts...")
        historical_forecasts = get_forecast_data_for_modelling(
            forecast_horizons=config.FORECAST_HORIZONS,
            start_date=config.DEFAULT_HISTORICAL_FORECAST_START
        )
        
        # Store yield data in lists for easier iteration
        yield_list = [
            yield_data['yld_2yr_ann'], 
            yield_data['yld_5yr_ann'], 
            yield_data['yld_10yr_ann'], 
            yield_data['yld_30yr_ann']
        ]
        
        # Step 3: Run data diagnostics (can be uncommented if needed)
        logger.info("Running data diagnostics...")
        
        # Analyze data availability
        data_availability = create_data_availability_summary(
            country_list=country_list,
            country_code_mapping=config.country_list_mapping,
            yield_list=yield_list,
            yield_names=tenor_names,
            pol_rat=pol_rat,
            cpi_inf=cpi_inf,
            act_track=act_track,
            risk_rating=risk_rating,
            historical_forecasts=historical_forecasts,
            unemployment_rate=unemployment_rate,
            iip_gdp=iip_gdp
        )
        results_summary['data_availability'] = data_availability
        
        # Step 4: Train and evaluate models in order from shortest to longest maturity
        logger.info("Training and evaluating models in order of maturity...")
        
        # Initialize a DataFrame to store results
        models_df = []
        
        # Dictionary to store model predictions by country and tenor
        predicted_yields_by_country = {}
        
        # Iterate through countries
        for country in country_list:
            logger.info(f"Processing country: {country}")
            
            # Initialize results for this country
            results_summary['model_results'][country] = {}
            predicted_yields_by_country[country] = {}
            results_summary['best_models'][country] = {}
            
            # Process tenors in order from shortest to longest
            for tenor_idx, tenor_name in enumerate(tenor_names):
                if tenor_idx < len(yield_list):
                    tenor_data = yield_list[tenor_idx]
                else:
                    logger.error(f"No data found for tenor {tenor_name}")
                    continue
                        
                logger.info(f"Processing {country} - {tenor_name}")
                
                # [MODIFICATION: TENOR MODEL TYPES SELECTION]
                # Replace the existing conditional logic for model selection with:
                tenor_model_types = args.models.copy()  # Make a copy to avoid modifying the original list

                # For 2yr tenor, we may want to remove 'nelson_siegel' as it requires multiple tenor points
                if tenor_idx == 0:# and 'nelson_siegel' in tenor_model_types and not args.models:
                    # Only remove if using default models (not explicitly specified by user)
                    logger.info("Removed nelson_siegel model for 2yr tenor (requires multiple tenor points)")

                # Make sure we try each model exactly once
                tenor_model_types = list(dict.fromkeys(tenor_model_types))

                # Reordering can stay to ensure simpler models are tried first (for better chance of success)
                if 'elasticnet' in tenor_model_types and tenor_model_types[0] != 'elasticnet':
                    tenor_model_types.remove('elasticnet')
                    tenor_model_types.insert(0, 'elasticnet')

                logger.info(f"Will try these model types for {country} - {tenor_name}: {tenor_model_types}")

                # For longer tenors, consider using yield-curve specific models
                use_yield_curve_models = (tenor_idx > 0)
                
                # Pass predicted yields from shorter tenors if available
                country_predicted_yields = predicted_yields_by_country.get(country, {})
                        
                # Train models of each type
                model_results_by_type = {}
        
                for model_type in tenor_model_types:
                    logger.info(f"Training {model_type} model for {country} - {tenor_name}")
                    
                    try:
                        # For backwards compatibility, use train_evaluate_mlp for mlp models
                        if model_type == 'mlp':
                            model_result = train_evaluate_model(
                                country=country,
                                tenor_name=tenor_name,
                                country_code_mapping=config.country_list_mapping,
                                tenor_data=tenor_data,
                                pol_rat=pol_rat,
                                cpi_inf=cpi_inf,
                                act_track=act_track,
                                risk_rating=risk_rating,
                                historical_forecasts=historical_forecasts,
                                unemployment_rate=unemployment_rate,
                                iip_gdp=iip_gdp,
                                predicted_yields=country_predicted_yields,
                                model_type=model_type,
                                use_advanced_training=True,
                                compare_models=args.compare_models
                            )
                        else:
                            # For other model types, use the new function
                            model_result = train_evaluate_model(
                                country=country,
                                tenor_name=tenor_name,
                                country_code_mapping=config.country_list_mapping,
                                tenor_data=tenor_data,
                                pol_rat=pol_rat,
                                cpi_inf=cpi_inf,
                                act_track=act_track,
                                model_type=model_type,
                                risk_rating=risk_rating,
                                historical_forecasts=historical_forecasts,
                                unemployment_rate=unemployment_rate,
                                iip_gdp=iip_gdp,
                                predicted_yields=country_predicted_yields,
                                use_advanced_training=True,
                                compare_models=args.compare_models,
                                use_yield_curve_models=use_yield_curve_models
                            )
                    except Exception as e:
                        logger.error(f"Error training {model_type} model for {country} - {tenor_name}: {e}")
                        traceback.print_exc()
                        model_result = {
                            'status': f'Failed - Exception in {model_type}',
                            'error': str(e),
                            'metrics': {}
                        }

                    
                    # Store the results regardless of success or failure
                    model_results_by_type[model_type] = model_result
                    
                    # If successful, store predictions for use by longer-tenor models
                    if model_result.get('status', '').startswith('Success') and 'predictions' in model_result:
                        if tenor_name not in predicted_yields_by_country[country]:
                            predicted_yields_by_country[country][tenor_name] = {}
                        
                        predicted_yields_by_country[country][tenor_name][model_type] = {
                            'predictions': model_result['predictions'],
                            'tenor_years': 2 if tenor_name == 'yld_2yr' else (
                                        5 if tenor_name == 'yld_5yr' else (
                                        10 if tenor_name == 'yld_10yr' else 30)),
                            'current_value': model_result['predictions'].iloc[-1] if not model_result['predictions'].empty else None
                        }
                        
                        # MODIFIED: If we got a successful model, use it and update the success count
                        results_summary['model_results'][country][tenor_name] = model_result
                        results_summary['model_results'][country][tenor_name]['best_model_type'] = model_type
                        results_summary['best_models'][country][tenor_name] = model_type
                        results_summary['success_count'] += 1
                        
                        # MODIFIED: Break out of the loop since we have a successful model
                        logger.info(f"Got successful {model_type} model for {country} - {tenor_name}. Using this model.")
                        
                
                # Create ensemble if requested and if we have multiple successful models
                if args.use_ensembles:
                    successful_models = {m_type: res for m_type, res in model_results_by_type.items() 
                                    if res.get('status', '').startswith('Success')}
                    
                    if len(successful_models) > 1:
                        logger.info(f"Creating ensemble model for {country} - {tenor_name}")
                        try:
                            # This requires you to implement build_ensemble_model
                            ensemble_result = build_ensemble_model(
                                country=country,
                                tenor_name=tenor_name,
                                model_results=successful_models,
                                weighted_by_performance=True
                            )
                            
                            # Store ensemble results
                            model_results_by_type['ensemble'] = ensemble_result
                            
                            # If successful, store predictions
                            if ensemble_result.get('status', '').startswith('Success') and 'predictions' in ensemble_result:
                                if tenor_name not in predicted_yields_by_country[country]:
                                    predicted_yields_by_country[country][tenor_name] = {}
                                
                                predicted_yields_by_country[country][tenor_name]['ensemble'] = {
                                    'predictions': ensemble_result['predictions'],
                                    'tenor_years': 2 if tenor_name == 'yld_2yr' else (
                                                5 if tenor_name == 'yld_5yr' else (
                                                10 if tenor_name == 'yld_10yr' else 30)),
                                    'current_value': ensemble_result['predictions'].iloc[-1] if not ensemble_result['predictions'].empty else None
                                }
                        except Exception as e:
                            logger.error(f"Error creating ensemble for {country} - {tenor_name}: {e}")
                            traceback.print_exc()
                
                # Select the best model based on performance metrics
                logger.info(f"Selecting best model for {country} - {tenor_name} from {len(model_results_by_type)} trained models")

                # Use the existing select_best_model function from data_worker.py
                best_model_type = select_best_model(
                                    model_results_by_type,
                                    country=country,
                                    tenor_name=tenor_name,
                                    output_dir=os.path.join(config.MODEL_DIR,f"\\{country}\\", country)
                                )

                if best_model_type:
                    logger.info(f"Best model for {country} - {tenor_name}: {best_model_type}")
                    
                    # Store the best model's results
                    results_summary['model_results'][country][tenor_name] = model_results_by_type[best_model_type]
                    results_summary['model_results'][country][tenor_name]['best_model_type'] = best_model_type
                    results_summary['best_models'][country][tenor_name] = best_model_type
                    
                    # Update success count
                    results_summary['success_count'] += 1
                    
                    # Store predicted yields for use in longer tenor models
                    if model_results_by_type[best_model_type].get('status', '').startswith('Success') and 'predictions' in model_results_by_type[best_model_type]:
                        if tenor_name not in predicted_yields_by_country[country]:
                            predicted_yields_by_country[country][tenor_name] = {}
                        
                        predicted_yields_by_country[country][tenor_name][best_model_type] = {
                            'predictions': model_results_by_type[best_model_type]['predictions'],
                            'tenor_years': 2 if tenor_name == 'yld_2yr' else (
                                        5 if tenor_name == 'yld_5yr' else (
                                        10 if tenor_name == 'yld_10yr' else 30)),
                            'current_value': model_results_by_type[best_model_type]['predictions'].iloc[-1] if not model_results_by_type[best_model_type]['predictions'].empty else None
                        }
                else:
                    logger.warning(f"No successful models for {country} - {tenor_name}")
                    
                    # Store the first model's results as a fallback
                    if model_results_by_type:
                        first_model_type = list(model_results_by_type.keys())[0]
                        results_summary['model_results'][country][tenor_name] = model_results_by_type[first_model_type]
                    else:
                        results_summary['model_results'][country][tenor_name] = {
                            'status': 'Failed - No Models Trained',
                            'error': 'No models were successfully trained',
                            'metrics': {}
                        }
                    
                    # Update failure count
                    results_summary['failure_count'] += 1

                # Add to results DataFrame for all models
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
        
        
                        
        # Step 5: Generate summary outputs
        if models_df:
            # Convert results to DataFrame and save
            models_results_df = pd.DataFrame(models_df)
            model_summary_path = os.path.join(config.OUTPUT_DIR, f"\\{country}\\", 'model_results_summary.csv')
            models_results_df.to_csv(model_summary_path, index=False)
            logger.info(f"Model results summary saved to '{model_summary_path}'")
            
            # Create summary visualization
            try:
                create_model_summary_visualization(results_summary)
                logger.info("Created model summary visualization")
            except Exception as e:
                logger.error(f"Error creating model summary visualization: {e}")
                traceback.print_exc()
        
        # Step 6: Generate a list of successful models
        logger.info("Generating unified forecasts and visualizations...")
        
        # Get list of successfully trained models
        successful_models = []
        for country in results_summary['model_results']:
            for tenor in results_summary['model_results'][country]:
                if results_summary['model_results'][country][tenor].get('status', '').startswith('Success'):
                    model_type = results_summary['model_results'][country][tenor].get('best_model_type', 'mlp')
                    successful_models.append((country, tenor, model_type))
        
        
        # Step 7: Create yield curve visualization for countries with successful models across tenors
        logger.info("Creating yield curve visualizations...")

        countries_with_all_tenors = []
        for country in results_summary['model_results']:
            has_all_tenors = all(
                tenor in results_summary['model_results'][country] and 
                results_summary['model_results'][country][tenor].get('status', '').startswith('Success')
                for tenor in tenor_names
            )
            if has_all_tenors:
                countries_with_all_tenors.append(country)

        logger.info(f"Total models: {results_summary['success_count'] + results_summary['failure_count']}")
        logger.info(f"Successful models: {results_summary['success_count']}")
        logger.info(f"Failed models: {results_summary['failure_count']}")

    except Exception as e:
        logger.error(f"Error saving models: {e}")

    try:
        logger.info("Saving best models with explicit feature names")

        for country in results_summary['best_models']:
            for tenor, model_type in results_summary['best_models'][country].items():
                try:
                    # Get the specific model result
                    model_result = results_summary['model_results'][country][tenor]

                    # Ensure model and scaler exist
                    if 'model' in model_result and 'scaler' in model_result:
                        # Create output directory
                        country_dir = os.path.join(config.MODEL_DIR, country)
                        os.makedirs(country_dir, exist_ok=True)

                        # Construct model path
                        model_path = os.path.join(country_dir, f"{country}_{tenor}_best_model.pkl")

                        # Use extract_feature_names to get feature names
                        try:
                            feature_names = extract_feature_names(model_result, country, tenor)
                        except Exception as e:
                            logger.warning(f"Could not extract feature names: {e}")
                                     # Extract feature names
                            feature_names = model_result['feature_details']['feature_columns']

                        # Create model package
                        model_package = {
                            'model': model_result['model'],
                            'scaler': model_result['scaler'],
                            'model_type': model_type,
                            'feature_names': feature_names,
                            'creation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'performance': model_result.get('metrics', {})
                        }

                        # Save model package
                        joblib.dump(model_package, model_path)
                        logger.info(f"Saved best model for {country} - {tenor}: {model_type}")

                        # Print feature names for verification
                        if feature_names:
                            logger.info(f"Feature names for {country} - {tenor}: {feature_names}")
                        else:
                            logger.warning(f"No feature names found for {country} - {tenor}")

                except Exception as e:
                    logger.error(f"Error saving best model for {country} - {tenor}: {e}")

        logger.info("Completed saving best models")

    except Exception as e:
        logger.error(f"Unexpected error during final model saving: {e}")

    return results_summary

if __name__ == "__main__":
    main()




