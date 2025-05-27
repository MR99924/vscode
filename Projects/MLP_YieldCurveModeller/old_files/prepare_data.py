import datetime
import numpy as np
import os
import sys
import config
import json
import pandas as pd
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from macrobond import Macrobond
from sklearn.preprocessing import LabelEncoder
from feature_engineering import forward_fill_to_current_date, select_features, combine_features, create_yield_curve_features

# Initialize API connectors
mb = Macrobond()
label_encoder = LabelEncoder()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_predictions = {}

def prepare_data(country, tenor_name, country_code_mapping, tenor, pol_rat, cpi_inf, act_track, iip_gdp,
                 risk_rating, historical_forecasts, unemployment_rate, predicted_yields=None,
                 forward_fill=True, enhance_features=False, handle_missing='ffill', feature_selection=None):
    """
    Prepare data for modeling by selecting and combining relevant features.
    Enhanced with improved missing data handling and feature engineering.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        country_code_mapping: dict - Mapping from country names to country codes
        tenor: DataFrame - Yield data for the specified tenor
        pol_rat: DataFrame - Policy rate data
        cpi_inf: DataFrame - Inflation data
        act_track: DataFrame - Economic activity tracker data
        risk_rating: DataFrame - Risk rating data
        historical_forecasts: dict - Historical forecasts from forecast_generator
        debt_gdp: DataFrame - Debt to GDP ratio data (optional)
        unemployment_rate: DataFrame - Unemployment rate data
        predicted_yields: dict - Dictionary containing predicted yields for shorter tenors
        forward_fill: bool - Whether to forward fill missing data
        enhance_features: bool - Whether to create additional engineered features
        handle_missing: str - Method for handling missing values ('ffill', 'drop', or 'impute')
        feature_selection: dict - Parameters for feature selection (if None, no selection is performed)
        
    Returns:
        tuple: (x, y, feature_details) where:
            - x contains features
            - y contains target values
            - feature_details is a dict with information about feature sources
    """
    # Initialize feature details dictionary
    feature_details = {
        'country': country,
        'tenor': tenor_name,
        'feature_sources': [],
        'date_ranges': {},
        'feature_counts': {},
        'data_quality': {},
        'feature_engineering': {}
    }
    
    logger.info(f"Preparing data for {country} - {tenor_name}")

    # Initialize feature DataFrames list and columns list
    feature_dfs = []
    all_columns = []

    # Get country code
    if country not in country_code_mapping:
        logger.error(f"Country {country} not found in country_code_mapping")
        return pd.DataFrame(), pd.Series(), feature_details
    
    country_code = country_code_mapping[country]
    logger.info(f"Country code: {country_code}")
    
    # Extract forecast horizon from tenor name (2yr, 5yr, 10yr, 30yr)
    forecast_horizon = tenor_name.replace('yld_', '').replace('yr', '')
    logger.info(f"Forecast horizon: {forecast_horizon}")

    # Process historical forecasts for this tenor if available
    if historical_forecasts is not None:
        # Get the historical forecast DataFrame
        historical_df = historical_forecasts.get("historical_df", pd.DataFrame())
        
        if not historical_df.empty:
            # Define column names based on tenor
            growth_forecast_col = f"gdp_forecast_{forecast_horizon}m_{country_code}"
            inflation_forecast_col = f"cpi_forecast_{forecast_horizon}m_{country_code}"
            
            # Find matching columns (using startswith to handle potential naming variations)
            forecast_cols = [col for col in historical_df.columns if 
                            (col.startswith(f"gdp_forecast_{forecast_horizon}") and col.endswith(f"_{country_code}")) or
                            (col.startswith(f"cpi_forecast_{forecast_horizon}") and col.endswith(f"_{country_code}"))]
            
            # Check if we found any matching columns
            if forecast_cols:
                logger.info(f"Found {len(forecast_cols)} matching forecast columns: {forecast_cols}")
                forecast_data = historical_df[forecast_cols].copy()
                
                # Forward fill forecasts to current date if requested
                if forward_fill and not forecast_data.empty:
                    forecast_data = forward_fill_to_current_date(forecast_data)
                
                # Add forecast data quality metrics
                if not forecast_data.empty:
                    missing_pct = forecast_data.isna().mean().mean() * 100
                    feature_details['data_quality'][f'historical_forecasts_{forecast_horizon}'] = {
                        'missing_pct': missing_pct,
                        'date_range': f"{forecast_data.index.min().strftime('%Y-%m-%d')} to {forecast_data.index.max().strftime('%Y-%m-%d')}",
                        'columns': forecast_cols
                    }
                
                feature_dfs.append(forecast_data)
                all_columns.extend(forecast_cols)
            else:
                # Try alternative naming patterns
                alt_growth_col = f"Growth_{forecast_horizon}yr"
                alt_inflation_col = f"Inflation_{forecast_horizon}yr"
                
                country_forecasts = historical_forecasts.get("countries", {}).get(country, {})
                if country_forecasts and isinstance(country_forecasts, pd.DataFrame):
                    forecast_cols = [col for col in country_forecasts.columns if 
                                     col.startswith(alt_growth_col) or col.startswith(alt_inflation_col)]
                    
                    if forecast_cols:
                        logger.info(f"Found {len(forecast_cols)} alternative forecast columns: {forecast_cols}")
                        forecast_data = country_forecasts[forecast_cols].copy()
                        
                        # Forward fill forecasts to current date if requested
                        if forward_fill and not forecast_data.empty:
                            forecast_data = forward_fill_to_current_date(forecast_data)
                        
                        # Add forecast data quality metrics
                        if not forecast_data.empty:
                            missing_pct = forecast_data.isna().mean().mean() * 100
                            feature_details['data_quality'][f'historical_forecasts_alt_{forecast_horizon}'] = {
                                'missing_pct': missing_pct,
                                'date_range': f"{forecast_data.index.min().strftime('%Y-%m-%d')} to {forecast_data.index.max().strftime('%Y-%m-%d')}",
                                'columns': forecast_cols
                            }
                        
                        feature_dfs.append(forecast_data)
                        all_columns.extend(forecast_cols)
                    else:
                        logger.warning(f"No matching forecast columns found for {country} - {tenor_name}")
                else:
                    logger.warning(f"No forecast data available for {country}")
        else:
            logger.warning("Historical forecast DataFrame is empty")
    else:
        logger.warning("No historical forecast data provided")
    
    # Get target variable (yield for the specified tenor and country)
    yield_column = f"{tenor_name}_{country_code}"
    if yield_column not in tenor.columns:
        logger.error(f"Yield column {yield_column} not found in tenor data")
        return pd.DataFrame(), pd.Series(), feature_details
    
    y = tenor[yield_column].dropna()
    feature_details['target_column'] = yield_column
    
    if not y.empty:
        logger.info(f"Target yield data: {yield_column}")
        logger.info(f"  Date range: {y.index.min().strftime('%Y-%m-%d')} to {y.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"  Number of observations: {len(y)}")
        
        # Add target data quality metrics
        feature_details['data_quality']['target'] = {
            'missing_pct': (1 - len(y) / len(tenor)) * 100 if len(tenor) > 0 else 0,
            'date_range': f"{y.index.min().strftime('%Y-%m-%d')} to {y.index.max().strftime('%Y-%m-%d')}",
            'count': len(y)
        }
    else:
        logger.error(f"No valid data found for {yield_column}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    feature_details['date_ranges']['target'] = {
        'start': y.index.min().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'end': y.index.max().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'count': len(y)
    }
    
    # Add predicted yields from shorter maturities as features
    if predicted_yields is not None:
        logger.info("Checking for predicted yields from shorter maturities:")
        
        # For 5yr model, include predicted 2yr yield
        if tenor_name == 'yld_5yr' and 'yld_2yr' in predicted_yields:
            pred_2yr = predicted_yields['yld_2yr']
            if not isinstance(pred_2yr, pd.Series):
                logger.warning(f"Predicted 2yr yield is not a pandas Series")
            elif pred_2yr.empty:
                logger.warning(f"Predicted 2yr yield Series is empty")
            else:
                logger.info(f"Adding predicted 2yr yield as feature")
                pred_df = pd.DataFrame({'pred_yld_2yr': pred_2yr})
                
                # Forward fill if needed and requested
                if forward_fill:
                    pred_df = forward_fill_to_current_date(pred_df)
                
                feature_dfs.append(pred_df)
                all_columns.append('pred_yld_2yr')
                
                # Add to feature details
                feature_details['feature_sources'].append('predicted_2yr')
                feature_details['feature_counts']['predicted_2yr'] = 1
                feature_details['date_ranges']['predicted_2yr'] = {
                    'start': pred_df.index.min().strftime('%Y-%m-%d') if not pred_df.empty else 'N/A',
                    'end': pred_df.index.max().strftime('%Y-%m-%d') if not pred_df.empty else 'N/A',
                    'count': len(pred_df)
                }
                
                # Add data quality metrics
                feature_details['data_quality']['predicted_2yr'] = {
                    'missing_pct': pred_df.isna().mean().mean() * 100,
                    'date_range': f"{pred_df.index.min().strftime('%Y-%m-%d')} to {pred_df.index.max().strftime('%Y-%m-%d')}",
                    'count': len(pred_df)
                }
        
        # For 10yr model, include predicted 5yr yield
        elif tenor_name == 'yld_10yr' and 'yld_5yr' in predicted_yields:
            pred_5yr = predicted_yields['yld_5yr']
            if not isinstance(pred_5yr, pd.Series):
                logger.warning(f"Predicted 5yr yield is not a pandas Series")
            elif pred_5yr.empty:
                logger.warning(f"Predicted 5yr yield Series is empty")
            else:
                logger.info(f"Adding predicted 5yr yield as feature")
                pred_df = pd.DataFrame({'pred_yld_5yr': pred_5yr})
                
                # Forward fill if needed and requested
                if forward_fill:
                    pred_df = forward_fill_to_current_date(pred_df)
                
                feature_dfs.append(pred_df)
                all_columns.append('pred_yld_5yr')
                
                # Add to feature details
                feature_details['feature_sources'].append('predicted_5yr')
                feature_details['feature_counts']['predicted_5yr'] = 1
                feature_details['date_ranges']['predicted_5yr'] = {
                    'start': pred_5yr.index.min().strftime('%Y-%m-%d') if not pred_5yr.empty else 'N/A',
                    'end': pred_5yr.index.max().strftime('%Y-%m-%d') if not pred_5yr.empty else 'N/A',
                    'count': len(pred_5yr)
                }
                
                # Add data quality metrics
                feature_details['data_quality']['predicted_5yr'] = {
                    'missing_pct': pred_df.isna().mean().mean() * 100,
                    'date_range': f"{pred_df.index.min().strftime('%Y-%m-%d')} to {pred_df.index.max().strftime('%Y-%m-%d')}",
                    'count': len(pred_df)
                }
        
        # For 30yr model, include predicted 10yr yield
        elif tenor_name == 'yld_30yr' and 'yld_10yr' in predicted_yields:
            pred_10yr = predicted_yields['yld_10yr']
            if not isinstance(pred_10yr, pd.Series):
                logger.warning(f"Predicted 10yr yield is not a pandas Series")
            elif pred_10yr.empty:
                logger.warning(f"Predicted 10yr yield Series is empty")
            else:
                logger.info(f"Adding predicted 10yr yield as feature")
                pred_df = pd.DataFrame({'pred_yld_10yr': pred_10yr})
                
                # Forward fill if needed and requested
                if forward_fill:
                    pred_df = forward_fill_to_current_date(pred_df)
                
                feature_dfs.append(pred_df)
                all_columns.append('pred_yld_10yr')
                
                # Add to feature details
                feature_details['feature_sources'].append('predicted_10yr')
                feature_details['feature_counts']['predicted_10yr'] = 1
                feature_details['date_ranges']['predicted_10yr'] = {
                    'start': pred_10yr.index.min().strftime('%Y-%m-%d') if not pred_10yr.empty else 'N/A',
                    'end': pred_10yr.index.max().strftime('%Y-%m-%d') if not pred_10yr.empty else 'N/A',
                    'count': len(pred_10yr)
                }
                
                # Add data quality metrics
                feature_details['data_quality']['predicted_10yr'] = {
                    'missing_pct': pred_df.isna().mean().mean() * 100,
                    'date_range': f"{pred_df.index.min().strftime('%Y-%m-%d')} to {pred_df.index.max().strftime('%Y-%m-%d')}",
                    'count': len(pred_df)
                }
            
        else:
            logger.info(f"No applicable predicted yields for {tenor_name}")
    
    # Select features based on tenor and organize data sources
    source_dfs = {
        'policy_rates': pol_rat,
        'inflation': cpi_inf,
        'activity': act_track,
        'risk_rating': risk_rating,
        'unemployment_rate': unemployment_rate,
        'iip_gdp': iip_gdp
    }
    
    # Forward fill source DataFrames if requested
    if forward_fill:
        for source_name, source_df in source_dfs.items():
            if source_df is not None and not source_df.empty:
                source_dfs[source_name] = forward_fill_to_current_date(source_df)
                logger.info(f"Forward-filled {source_name} data")
    
    # Determine if country is an emerging market (affects feature selection)
    is_emerging_market = country in config.emerging_markets
    
    # Select appropriate sources based on tenor
    if tenor_name == 'yld_2yr':
        # For 2-year yields, use policy rates, inflation, and activity
        sources = ['policy_rates', 'inflation', 'activity', 'unemployment_rate']
        logger.info(f"Model type: {tenor_name} - Using policy rates, inflation, and economic activity")
    elif tenor_name == 'yld_5yr':
        # For 5-year yields, use policy rates, inflation, risk ratings, and debt/GDP
        sources = ['policy_rates', 'inflation', 'risk_rating', 'unemployment_rate']
        # if 'debt_gdp' in source_dfs and source_dfs['debt_gdp'] is not None and not source_dfs['debt_gdp'].empty:
        #     sources.append('debt_gdp')
        logger.info(f"Model type: {tenor_name} - Using policy rates, inflation, risk ratings, and debt/GDP")
    elif tenor_name in ['yld_10yr', 'yld_30yr']:
        # For longer-term yields, use all available features
        sources = ['policy_rates', 'inflation', 'activity', 'risk_rating', 'unemployment_rate']
        # if 'debt_gdp' in source_dfs and source_dfs['debt_gdp'] is not None and not source_dfs['debt_gdp'].empty:
        #     sources.append('debt_gdp')
        logger.info(f"Model type: {tenor_name} - Using policy rates, inflation, activity, risk ratings, and debt/GDP")
    else:
        logger.error(f"Unsupported tenor: {tenor_name}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    # Add US policy rate influence for emerging markets
    if is_emerging_market and source_dfs['policy_rates'] is not None:
        pol_rat_df = source_dfs['policy_rates']
        if 'pol_rat_us' in pol_rat_df.columns:
            logger.info(f"Adding US policy rate influence for emerging market: {country}")
            us_pol_df = pd.DataFrame({'pol_rat_us_influence': pol_rat_df['pol_rat_us']})
            
            # Forward fill if requested
            if forward_fill:
                us_pol_df = forward_fill_to_current_date(us_pol_df)
            
            feature_dfs.append(us_pol_df)
            all_columns.append('pol_rat_us_influence')
    
    feature_details['feature_sources'] = sources
    logger.info(f"Using feature sources: {', '.join(sources)}")
    
    # For each source, extract the relevant columns and add to features
    for source_name in sources:
        logger.info(f"Processing source: {source_name}")
        source_df = source_dfs.get(source_name)
        
        if source_df is None or source_df.empty:
            logger.warning(f"Source {source_name} is empty or None")
            continue
            
        country_cols = [col for col in source_df.columns if col.endswith(f"_{country_code}")]
        
        # For some sources like policy rates, also include global factors
        if source_name == 'policy_rates':
            # Include US policy rate for all countries
            if 'pol_rat_us' in source_df.columns and 'pol_rat_us' not in country_cols:
                country_cols.append('pol_rat_us')
            
            # Include ECB policy rate for European countries
            if hasattr(config, 'european_countries') and country in config.european_countries:
                if 'pol_rat_ecb' in source_df.columns and 'pol_rat_ecb' not in country_cols:
                    country_cols.append('pol_rat_ecb')
        
        logger.info(f"Selected {len(country_cols)} columns from {source_name}: {country_cols}")
        
        if country_cols:
            source_data = source_df[country_cols].copy()
            all_columns.extend(country_cols)
            
            if not source_data.empty:
                logger.info(f"Date range: {source_data.index.min().strftime('%Y-%m-%d')} to {source_data.index.max().strftime('%Y-%m-%d')}")
                logger.info(f"Number of observations: {len(source_data)}")
                logger.info(f"Missing values: {source_data.isna().mean().mean() * 100:.2f}%")
            else:
                logger.warning("No data after column filtering")
            
            # Add to feature details
            feature_details['feature_counts'][source_name] = len(country_cols)
            feature_details['date_ranges'][source_name] = {
                'start': source_data.index.min().strftime('%Y-%m-%d') if not source_data.empty else 'N/A',
                'end': source_data.index.max().strftime('%Y-%m-%d') if not source_data.empty else 'N/A',
                'count': len(source_data),
                'columns': country_cols
            }
            
            feature_dfs.append(source_data)
        else:
            logger.warning(f"No columns found for {country} in {source_name}")
    
    if not feature_dfs:
        logger.error(f"No features found for {country} - {tenor_name}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    # Combine selected features
    x = pd.concat(feature_dfs, axis=1)
    logger.info(f"Combined features shape: {x.shape}")
    
    feature_details['total_features'] = x.shape[1]
    feature_details['feature_columns'] = list(x.columns)
    feature_details['combined_date_range'] = {
        'start': x.index.min().strftime('%Y-%m-%d') if not x.empty else 'N/A',
        'end': x.index.max().strftime('%Y-%m-%d') if not x.empty else 'N/A',
        'count': len(x)
    }
    
    # Save raw data before missing value handling for diagnostics
    x_raw = x.copy()
    
    # Handle missing values according to specified method
    if handle_missing == 'drop':
        # Drop rows with NaN values
        x_clean = x.dropna()
        logger.info(f"Dropping rows with missing values. Remaining: {len(x_clean)}/{len(x)} ({len(x_clean)/len(x)*100:.1f}%)")
    elif handle_missing == 'impute':
        # Impute missing values with column median
        x_clean = x.copy()
        for col in x_clean.columns:
            if x_clean[col].isna().any():
                median = x_clean[col].median()
                na_count = x_clean[col].isna().sum()
                x_clean[col] = x_clean[col].fillna(median)
                logger.info(f"Imputed {na_count} missing values in {col} with median ({median:.4f})")
    else:  # Default to forward fill then backfill
        # Forward fill then backfill to handle missing values
        x_clean = x.ffill().bfill()
        logger.info("Applied forward fill and backfill to handle missing values")
    
    # Calculate missing data statistics for each column
    missing_stats = pd.DataFrame({
        'missing_count': x_raw.isna().sum(),
        'missing_pct': x_raw.isna().mean() * 100,
        'missing_before': x_raw.isna().sum(),
        'missing_after': x_clean.isna().sum(),
    })
    missing_stats = missing_stats.sort_values('missing_pct', ascending=False)
    
    # Store missing data statistics in feature details
    feature_details['data_quality']['missing_stats'] = missing_stats.to_dict()
    feature_details['data_quality']['total_missing_before'] = int(x_raw.isna().sum().sum())
    feature_details['data_quality']['total_missing_after'] = int(x_clean.isna().sum().sum())
    feature_details['data_quality']['rows_before'] = len(x_raw)
    feature_details['data_quality']['rows_after'] = len(x_clean)
    
    feature_details['clean_date_range'] = {
        'start': x_clean.index.min().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'end': x_clean.index.max().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'count': len(x_clean)
    }
    
    # Ensure x and y have overlapping indices
    common_idx = x_clean.index.intersection(y.index)
    logger.info(f"Data overlap: {len(common_idx)} dates")
    
    if len(common_idx) == 0:
        logger.error(f"No overlapping data found between features and target")
        
        # Add diagnostics about the intersection
        feature_details['overlap_diagnosis'] = {
            'target_earliest': y.index.min().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
            'target_latest': y.index.max().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
            'features_earliest': x_clean.index.min().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
            'features_latest': x_clean.index.max().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
            'target_count': len(y),
            'features_count': len(x_clean),
            'common_count': 0,
            'issue': 'No date overlap between features and target'
        }
        
        return pd.DataFrame(), pd.Series(), feature_details
    
    x_final = x_clean.loc[common_idx]
    y_final = y.loc[common_idx]
    
    logger.info(f"Final dataset: {len(y_final)} observations with {x_final.shape[1]} features")

    # Create additional engineered features if requested
    if enhance_features:
        logger.info("NOT creating engineered features")
        
        # Create standard engineered features
        #eng_features = engineer_features(x_final, country=country_code, tenor=tenor_name)
        
        # Create yield curve features
        logger.info("Creating yield curve-specific features")
        yield_features = create_yield_curve_features(tenor, [country_code])
        
        # Only keep rows that overlap with x_final
        if not yield_features.empty:
            yield_features = yield_features.loc[yield_features.index.intersection(x_final.index)]
        
        # Create lagged features for key indicators
        logger.info("Creating lagged features")
        key_indicators = []
        
        # Add policy rate for country
        pol_rate_col = f"pol_rat_{country_code}"
        if pol_rate_col in x_final.columns:
            key_indicators.append(pol_rate_col)
        
        # Add inflation for country
        inflation_col = f"cpi_inf_{country_code}"
        if inflation_col in x_final.columns:
            key_indicators.append(inflation_col)
        
        # Add unemployment for country
        unemployment_col = f"unemployment_rate_{country_code}"
        if unemployment_col in x_final.columns:
            key_indicators.append(unemployment_col)
                
        # Combine all features
        logger.info("Combining all features")
        enhanced_df = pd.concat([x_final, yield_features], axis=1)
        
        # Combine with original features
        if not enhanced_df.empty:
            x_final = combine_features(x_final, enhanced_df, handle_missing=handle_missing)
            
            # Update feature details
            feature_details['enhanced_features'] = list(enhanced_df.columns)
            feature_details['feature_engineering']['count'] = len(enhanced_df.columns)
            feature_details['feature_engineering']['types'] = {
                'yield_curve': len(yield_features.columns),
            }
            logger.info(f"Added {len(enhanced_df.columns)} engineered features. Final feature count: {len(x_final.columns)}")
    
    # Perform feature selection if requested
    if feature_selection is not None:
        logger.info("Performing feature selection")
        
        # Default selection parameters
        method = feature_selection.get('method', 'correlation')
        threshold = feature_selection.get('threshold', 0.05)
        top_n = feature_selection.get('top_n', None)
        
        # Select features
        x_selected = select_features(x_final, y_final, method=method, threshold=threshold, top_n=top_n)
        
        logger.info(f"Selected {x_selected.shape[1]}/{x_final.shape[1]} features using {method} method")
        
        # Update feature details
        feature_details['feature_selection'] = {
            'method': method,
            'threshold': threshold,
            'top_n': top_n,
            'original_count': x_final.shape[1],
            'selected_count': x_selected.shape[1],
            'selected_features': list(x_selected.columns)
        }
        
        # Replace x_final with selected features
        x_final = x_selected
    
    # Calculate feature correlations with target
    correlations = {}
    for col in x_final.columns:
        correlations[col] = abs(x_final[col].corr(y_final))
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Log top correlated features
    logger.info("Top 5 correlated features:")
    for i, (feature, corr) in enumerate(sorted_correlations[:5]):
        logger.info(f"  {i+1}. {feature}: {corr:.4f}")
    
    # Add to feature details
    feature_details['feature_correlations'] = {k: round(v, 4) for k, v in sorted_correlations}
    
    # Check data overlap between features and target
    common_indices = x_final.index.intersection(y_final.index)
    logger.info(f"Overlapping date range: {common_indices.min().strftime('%Y-%m-%d')} to {common_indices.max().strftime('%Y-%m-%d')}")
    logger.info(f"Number of overlapping points: {len(common_indices)}")
    logger.info(f"Overlap percentage of target: {len(common_indices) / len(y) * 100:.2f}%")
    logger.info(f"Overlap percentage of clean features: {len(common_indices) / len(x_clean) * 100:.2f}%")
    
    feature_details['overlap_diagnosis'] = {
        'target_earliest': y.index.min().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'target_latest': y.index.max().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'features_earliest': x_clean.index.min().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'features_latest': x_clean.index.max().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'overlapping_earliest': common_indices.min().strftime('%Y-%m-%d') if len(common_indices) > 0 else 'N/A',
        'overlapping_latest': common_indices.max().strftime('%Y-%m-%d') if len(common_indices) > 0 else 'N/A',
        'target_count': len(y),
        'features_count': len(x_clean),
        'common_count': len(common_indices),
        'issue': 'None'
    }
    
    # Print final dataset information
    logger.info(f"Final dataset shape: {x_final.shape[0]} rows × {x_final.shape[1]} columns")
    logger.info(f"Date range: {x_final.index.min().strftime('%Y-%m-%d')} to {x_final.index.max().strftime('%Y-%m-%d')}")
    
    # Check feature quality by examining spread and variance
    logger.info("Analyzing feature quality")
    feature_stats = pd.DataFrame({
        'min': x_final.min(),
        'max': x_final.max(),
        'mean': x_final.mean(),
        'median': x_final.median(),
        'std': x_final.std(),
        'var': x_final.var(),
        'skew': x_final.skew(),
        'kurt': x_final.kurtosis(),
        'corr_with_target': [x_final[col].corr(y_final) for col in x_final.columns]
    })
    
    # Identify low-variance features that might not be useful
    low_var_features = feature_stats[feature_stats['var'] < 0.01].index.tolist()
    if low_var_features:
        logger.warning(f"Found {len(low_var_features)} features with very low variance")
        for feat in low_var_features:
            logger.warning(f"  Low variance feature: {feat}, var={feature_stats.loc[feat, 'var']:.6f}")
    
    # Store feature quality statistics
    feature_details['feature_quality'] = {
        'feature_stats': feature_stats.to_dict(),
        'low_variance_features': low_var_features
    }
    
    # Check target distribution
    target_stats = {
        'min': float(y_final.min()),
        'max': float(y_final.max()),
        'mean': float(y_final.mean()),
        'median': float(y_final.median()),
        'std': float(y_final.std()),
        'skew': float(y_final.skew()),
        'kurt': float(y_final.kurtosis())
    }
    
    feature_details['target_stats'] = target_stats
    logger.info(f"Target statistics: mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}, range=[{target_stats['min']:.2f}, {target_stats['max']:.2f}]")
    
    # Save data preparation results to file if directory exists
    try:
        # Create a unique identifier for this data preparation
        prep_id = f"{country}_{tenor_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Ensure results directory exists
        results_dir = os.path.join('results', 'data_prep')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save feature details as JSON
        with open(os.path.join(results_dir, f"{prep_id}_details.json"), 'w') as f:
            # Use a custom encoder for numpy/pandas types
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif pd.isna(obj):
                        return None
                    return super(NpEncoder, self).default(obj)
            
            # Remove non-serializable elements from feature details
            serializable_details = feature_details.copy()
            if 'feature_stats' in serializable_details.get('feature_quality', {}):
                del serializable_details['feature_quality']['feature_stats']
            
            json.dump(serializable_details, f, cls=NpEncoder, indent=2)
        
        logger.info(f"Saved data preparation details to {os.path.join(results_dir, f'{prep_id}_details.json')}")
    except Exception as e:
        logger.warning(f"Could not save data preparation results: {str(e)}")
    
    if not x_final.empty and len(y_final) > 0:
    # Double-check that x and y have same length
        if len(x_final) != len(y_final):
            logger.warning(f"Dimension mismatch between features ({len(x_final)}) and target ({len(y_final)})")
            # Find common indices
            common_idx = x_final.index.intersection(y_final.index)
            logger.info(f"Using {len(common_idx)} common data points")
            
            # Restrict to common indices
            x_final = x_final.loc[common_idx]
            y_final = y_final.loc[common_idx]
        
        # Ensure all feature values are valid
        if x_final.isnull().any().any():
            logger.warning("Found NaN values in final features. Applying final cleaning.")
            x_final = x_final.fillna(method='ffill').fillna(method='bfill')
            # As a last resort, fill remaining NaNs with column means
            for col in x_final.columns:
                if x_final[col].isnull().any():
                    x_final[col] = x_final[col].fillna(x_final[col].mean() if not pd.isna(x_final[col].mean()) else 0)

        # Check for infinite values
        if np.any(np.isinf(x_final.values)):
            logger.warning("Found infinite values in final features. Replacing with large finite values.")
            x_final = x_final.replace([np.inf, -np.inf], [1e9, -1e9])
        
        # Log final dimensions to confirm consistency
        logger.info(f"Final dataset shape: {x_final.shape[0]} rows × {x_final.shape[1]} columns")
        logger.info(f"Final target length: {len(y_final)}")
        
        # Add data dimension info to feature_details
        feature_details['feature_columns'] = list(x_final.columns)

        
        logger.info(f"Feature columns for {country} - {tenor_name}: {feature_details['feature_columns']}")


    return x_final, y_final, feature_details