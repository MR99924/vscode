import datetime
import numpy as np
import os
import json
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Project imports
import config
from feature_engineering import forward_fill_to_current_date, select_features, combine_features, create_yield_curve_features

# Module level logger (will be configured by main application)
logger = logging.getLogger(__name__)


def prepare_data(country, tenor_name, country_code_mapping, tenor, pol_rat, cpi_inf, act_track, iip_gdp,
                 risk_rating, historical_forecasts, unemployment_rate, predicted_yields,
                 forward_fill=True, enhance_features=False, handle_missing='ffill', feature_selection=None,
                 mb_client=None, label_encoder=None):
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
        iip_gdp: DataFrame - IIP to GDP ratio data
        risk_rating: DataFrame - Risk rating data
        historical_forecasts: dict - Historical forecasts from forecast_generator
        unemployment_rate: DataFrame - Unemployment rate data
        predicted_yields: dict - Dictionary containing predicted yields for shorter tenors
        forward_fill: bool - Whether to forward fill missing data
        enhance_features: bool - Whether to create additional engineered features
        handle_missing: str - Method for handling missing values ('ffill', 'drop', or 'impute')
        feature_selection: dict - Parameters for feature selection (if None, no selection is performed)
        mb_client: Macrobond client instance (passed from calling function)
        label_encoder: LabelEncoder instance (passed from calling function if needed)
        
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
    historical_forecast_data = _process_historical_forecasts(
        historical_forecasts, forecast_horizon, country_code, 
        forward_fill, feature_details
    )
    
    if historical_forecast_data is not None:
        feature_dfs.append(historical_forecast_data['df'])
        all_columns.extend(historical_forecast_data['columns'])
    
    # Get target variable (yield for the specified tenor and country)
    y, target_details = _prepare_target_variable(tenor, tenor_name, country_code)
    if y.empty:
        logger.error(f"No valid target data found for {country} - {tenor_name}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    # Update feature details with target information
    feature_details.update(target_details)
    
    # Add predicted yields from shorter maturities as features
    predicted_yield_data = _process_predicted_yields(
        predicted_yields, tenor_name, forward_fill, feature_details
    )
    
    if predicted_yield_data is not None:
        feature_dfs.append(predicted_yield_data['df'])
        all_columns.extend(predicted_yield_data['columns'])
    
    # Prepare source data dictionaries
    source_dfs = _prepare_source_dataframes(
        pol_rat, cpi_inf, act_track, risk_rating, unemployment_rate, 
        iip_gdp, forward_fill
    )
    
    # Select appropriate features based on tenor and country characteristics
    feature_sources_data = _select_and_process_features(
        tenor_name, country, country_code, source_dfs, 
        feature_details, forward_fill
    )
    
    if feature_sources_data:
        feature_dfs.extend(feature_sources_data)
    
    if not feature_dfs:
        logger.error(f"No features found for {country} - {tenor_name}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    # Combine all features
    x = pd.concat(feature_dfs, axis=1)
    logger.info(f"Combined features shape: {x.shape}")
    
    # Update feature details
    _update_feature_details_combined(x, feature_details)
    
    # Handle missing values
    x_clean = _handle_missing_values(x, handle_missing, feature_details)
    
    # Ensure data overlap between features and target
    x_final, y_final = _ensure_data_overlap(x_clean, y, feature_details)
    
    if x_final.empty or y_final.empty:
        logger.error(f"No overlapping data found between features and target")
        return pd.DataFrame(), pd.Series(), feature_details
    
    # Create enhanced features if requested
    if enhance_features:
        x_final = _create_enhanced_features(
            x_final, y_final, tenor, country_code, tenor_name, 
            handle_missing, feature_details
        )
    
    # Perform feature selection if requested
    if feature_selection is not None:
        x_final = _perform_feature_selection(
            x_final, y_final, feature_selection, feature_details
        )
    
    # Calculate and store feature correlations
    _calculate_feature_correlations(x_final, y_final, feature_details)
    
    # Final data quality checks and cleanup
    x_final, y_final = _final_data_cleanup(x_final, y_final, feature_details)
    
    # Save preparation results
    _save_preparation_results(country, tenor_name, feature_details)
    
    logger.info(f"Data preparation completed for {country} - {tenor_name}")
    logger.info(f"Final dataset: {len(y_final)} observations with {x_final.shape[1]} features")
    
    return x_final, y_final, feature_details


def _process_historical_forecasts(historical_forecasts, forecast_horizon, country_code, 
                                forward_fill, feature_details):
    """Process historical forecasts data and return DataFrame if available."""
    if historical_forecasts is None:
        logger.warning("No historical forecast data provided")
        return None
    
    # Get the historical forecast DataFrame
    historical_df = historical_forecasts.get("historical_df", pd.DataFrame())
    
    if historical_df.empty:
        logger.warning("Historical forecast DataFrame is empty")
        return None
    
    # Define column names based on tenor
    growth_forecast_col = f"gdp_forecast_{forecast_horizon}m_{country_code}"
    inflation_forecast_col = f"cpi_forecast_{forecast_horizon}m_{country_code}"
    
    # Find matching columns
    forecast_cols = [col for col in historical_df.columns if 
                    (col.startswith(f"gdp_forecast_{forecast_horizon}") and col.endswith(f"_{country_code}")) or
                    (col.startswith(f"cpi_forecast_{forecast_horizon}") and col.endswith(f"_{country_code}"))]
    
    if forecast_cols:
        logger.info(f"Found {len(forecast_cols)} matching forecast columns: {forecast_cols}")
        forecast_data = historical_df[forecast_cols].copy()
        
        # Forward fill forecasts if requested
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
        
        return {'df': forecast_data, 'columns': forecast_cols}
    
    # Try alternative naming patterns
    alt_growth_col = f"Growth_{forecast_horizon}yr"
    alt_inflation_col = f"Inflation_{forecast_horizon}yr"
    
    country_forecasts = historical_forecasts.get("countries", {}).get(country_code, {})
    if country_forecasts and isinstance(country_forecasts, pd.DataFrame):
        forecast_cols = [col for col in country_forecasts.columns if 
                        col.startswith(alt_growth_col) or col.startswith(alt_inflation_col)]
        
        if forecast_cols:
            logger.info(f"Found {len(forecast_cols)} alternative forecast columns: {forecast_cols}")
            forecast_data = country_forecasts[forecast_cols].copy()
            
            if forward_fill and not forecast_data.empty:
                forecast_data = forward_fill_to_current_date(forecast_data)
            
            return {'df': forecast_data, 'columns': forecast_cols}
    
    logger.warning(f"No matching forecast columns found")
    return None


def _prepare_target_variable(tenor, tenor_name, country_code):
    """Prepare target variable and return series with details."""
    yield_column = f"{tenor_name}_{country_code}"
    
    if yield_column not in tenor.columns:
        logger.error(f"Yield column {yield_column} not found in tenor data")
        return pd.Series(), {}
    
    y = tenor[yield_column].dropna()
    
    target_details = {
        'target_column': yield_column,
        'date_ranges': {
            'target': {
                'start': y.index.min().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
                'end': y.index.max().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
                'count': len(y)
            }
        },
        'data_quality': {
            'target': {
                'missing_pct': (1 - len(y) / len(tenor)) * 100 if len(tenor) > 0 else 0,
                'date_range': f"{y.index.min().strftime('%Y-%m-%d')} to {y.index.max().strftime('%Y-%m-%d')}" if not y.empty else 'N/A',
                'count': len(y)
            }
        }
    }
    
    if not y.empty:
        logger.info(f"Target yield data: {yield_column}")
        logger.info(f"  Date range: {y.index.min().strftime('%Y-%m-%d')} to {y.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"  Number of observations: {len(y)}")
    
    return y, target_details


def _process_predicted_yields(predicted_yields, tenor_name, forward_fill, feature_details):
    """Process predicted yields from shorter maturities as features."""
    if predicted_yields is None:
        return None
    
    logger.info("Checking for predicted yields from shorter maturities")
    
    # Map tenor names to their predecessor tenors
    tenor_hierarchy = {
        'yld_5yr': 'yld_2yr',
        'yld_10yr': 'yld_5yr',
        'yld_30yr': 'yld_10yr'
    }
    
    if tenor_name not in tenor_hierarchy:
        logger.info(f"No applicable predicted yields for {tenor_name}")
        return None
    
    predecessor_tenor = tenor_hierarchy[tenor_name]
    
    if predecessor_tenor not in predicted_yields:
        logger.info(f"No predicted {predecessor_tenor} yield available")
        return None
    
    pred_yield = predicted_yields[predecessor_tenor]
    
    if not isinstance(pred_yield, pd.Series) or pred_yield.empty:
        logger.warning(f"Predicted {predecessor_tenor} yield is not valid")
        return None
    
    logger.info(f"Adding predicted {predecessor_tenor} yield as feature")
    column_name = f"pred_{predecessor_tenor}"
    pred_df = pd.DataFrame({column_name: pred_yield})
    
    # Forward fill if requested
    if forward_fill:
        pred_df = forward_fill_to_current_date(pred_df)
    
    # Update feature details
    feature_details['feature_sources'].append(f'predicted_{predecessor_tenor}')
    feature_details['feature_counts'][f'predicted_{predecessor_tenor}'] = 1
    feature_details['date_ranges'][f'predicted_{predecessor_tenor}'] = {
        'start': pred_df.index.min().strftime('%Y-%m-%d') if not pred_df.empty else 'N/A',
        'end': pred_df.index.max().strftime('%Y-%m-%d') if not pred_df.empty else 'N/A',
        'count': len(pred_df)
    }
    
    # Add data quality metrics
    feature_details['data_quality'][f'predicted_{predecessor_tenor}'] = {
        'missing_pct': pred_df.isna().mean().mean() * 100,
        'date_range': f"{pred_df.index.min().strftime('%Y-%m-%d')} to {pred_df.index.max().strftime('%Y-%m-%d')}",
        'count': len(pred_df)
    }
    
    return {'df': pred_df, 'columns': [column_name]}


def _prepare_source_dataframes(pol_rat, cpi_inf, act_track, risk_rating, 
                              unemployment_rate, iip_gdp, forward_fill):
    """Prepare and forward fill source dataframes."""
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
    
    return source_dfs


def _select_and_process_features(tenor_name, country, country_code, source_dfs, 
                               feature_details, forward_fill):
    """Select appropriate features based on tenor and process them."""
    # Determine if country is an emerging market
    is_emerging_market = hasattr(config, 'emerging_markets') and country in config.emerging_markets
    
    # Select appropriate sources based on tenor
    sources = _get_sources_for_tenor(tenor_name)
    
    # Add US policy rate influence for emerging markets
    feature_dfs = []
    if is_emerging_market and source_dfs['policy_rates'] is not None:
        us_influence_df = _add_us_policy_rate_influence(
            source_dfs['policy_rates'], forward_fill, country
        )
        if us_influence_df is not None:
            feature_dfs.append(us_influence_df)
    
    feature_details['feature_sources'] = sources
    logger.info(f"Using feature sources: {', '.join(sources)}")
    
    # Process each source
    for source_name in sources:
        source_data = _process_feature_source(
            source_name, source_dfs[source_name], country_code, 
            feature_details
        )
        if source_data is not None:
            feature_dfs.append(source_data)
    
    return feature_dfs


def _get_sources_for_tenor(tenor_name):
    """Get appropriate data sources based on tenor."""
    if tenor_name == 'yld_2yr':
        sources = ['policy_rates', 'inflation', 'activity', 'unemployment_rate']
        logger.info(f"Model type: {tenor_name} - Using policy rates, inflation, and economic activity")
    elif tenor_name == 'yld_5yr':
        sources = ['policy_rates', 'inflation', 'risk_rating', 'unemployment_rate']
        logger.info(f"Model type: {tenor_name} - Using policy rates, inflation, and risk ratings")
    elif tenor_name in ['yld_10yr', 'yld_30yr']:
        sources = ['policy_rates', 'inflation', 'activity', 'risk_rating', 'unemployment_rate']
        logger.info(f"Model type: {tenor_name} - Using all available features")
    else:
        logger.error(f"Unsupported tenor: {tenor_name}")
        return []
    
    return sources


def _add_us_policy_rate_influence(pol_rat_df, forward_fill, country):
    """Add US policy rate influence for emerging markets."""
    if 'pol_rat_us' in pol_rat_df.columns:
        logger.info(f"Adding US policy rate influence for emerging market: {country}")
        us_pol_df = pd.DataFrame({'pol_rat_us_influence': pol_rat_df['pol_rat_us']})
        
        if forward_fill:
            us_pol_df = forward_fill_to_current_date(us_pol_df)
        
        return us_pol_df
    
    return None


def _process_feature_source(source_name, source_df, country_code, feature_details):
    """Process a single feature source and return DataFrame."""
    if source_df is None or source_df.empty:
        logger.warning(f"Source {source_name} is empty or None")
        return None
    
    logger.info(f"Processing source: {source_name}")
    
    # Get country-specific columns
    country_cols = [col for col in source_df.columns if col.endswith(f"_{country_code}")]
    
    # Add global factors for certain sources
    if source_name == 'policy_rates':
        # Include US policy rate for all countries
        if 'pol_rat_us' in source_df.columns and 'pol_rat_us' not in country_cols:
            country_cols.append('pol_rat_us')
        
        # Include ECB policy rate for European countries
        if (hasattr(config, 'european_countries') and 
            country_code in getattr(config, 'european_countries', []) and
            'pol_rat_ecb' in source_df.columns and 'pol_rat_ecb' not in country_cols):
            country_cols.append('pol_rat_ecb')
    
    logger.info(f"Selected {len(country_cols)} columns from {source_name}: {country_cols}")
    
    if not country_cols:
        logger.warning(f"No columns found for country in {source_name}")
        return None
    
    source_data = source_df[country_cols].copy()
    
    if not source_data.empty:
        logger.info(f"Date range: {source_data.index.min().strftime('%Y-%m-%d')} to {source_data.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"Number of observations: {len(source_data)}")
        logger.info(f"Missing values: {source_data.isna().mean().mean() * 100:.2f}%")
        
        # Update feature details
        feature_details['feature_counts'][source_name] = len(country_cols)
        feature_details['date_ranges'][source_name] = {
            'start': source_data.index.min().strftime('%Y-%m-%d'),
            'end': source_data.index.max().strftime('%Y-%m-%d'),
            'count': len(source_data),
            'columns': country_cols
        }
        
        return source_data
    
    logger.warning("No data after column filtering")
    return None


def _update_feature_details_combined(x, feature_details):
    """Update feature details with combined feature information."""
    feature_details['total_features'] = x.shape[1]
    feature_details['feature_columns'] = list(x.columns)
    feature_details['combined_date_range'] = {
        'start': x.index.min().strftime('%Y-%m-%d') if not x.empty else 'N/A',
        'end': x.index.max().strftime('%Y-%m-%d') if not x.empty else 'N/A',
        'count': len(x)
    }


def _handle_missing_values(x, handle_missing, feature_details):
    """Handle missing values according to specified method."""
    x_raw = x.copy()
    
    if handle_missing == 'drop':
        x_clean = x.dropna()
        logger.info(f"Dropping rows with missing values. Remaining: {len(x_clean)}/{len(x)} ({len(x_clean)/len(x)*100:.1f}%)")
    elif handle_missing == 'impute':
        x_clean = x.copy()
        for col in x_clean.columns:
            if x_clean[col].isna().any():
                median = x_clean[col].median()
                na_count = x_clean[col].isna().sum()
                x_clean[col] = x_clean[col].fillna(median)
                logger.info(f"Imputed {na_count} missing values in {col} with median ({median:.4f})")
    else:  # Default to forward fill then backfill
        x_clean = x.ffill().bfill()
        logger.info("Applied forward fill and backfill to handle missing values")
    
    # Calculate and store missing data statistics
    _calculate_missing_statistics(x_raw, x_clean, feature_details)
    
    return x_clean


def _calculate_missing_statistics(x_raw, x_clean, feature_details):
    """Calculate missing data statistics."""
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


def _ensure_data_overlap(x_clean, y, feature_details):
    """Ensure features and target have overlapping data."""
    common_idx = x_clean.index.intersection(y.index)
    logger.info(f"Data overlap: {len(common_idx)} dates")
    
    if len(common_idx) == 0:
        logger.error("No overlapping data found between features and target")
        
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
        
        return pd.DataFrame(), pd.Series()
    
    x_final = x_clean.loc[common_idx]
    y_final = y.loc[common_idx]
    
    # Update clean date range in feature details
    feature_details['clean_date_range'] = {
        'start': x_final.index.min().strftime('%Y-%m-%d') if not x_final.empty else 'N/A',
        'end': x_final.index.max().strftime('%Y-%m-%d') if not x_final.empty else 'N/A',
        'count': len(x_final)
    }
    
    # Add overlap diagnosis
    feature_details['overlap_diagnosis'] = {
        'target_earliest': y.index.min().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'target_latest': y.index.max().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'features_earliest': x_clean.index.min().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'features_latest': x_clean.index.max().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'overlapping_earliest': common_idx.min().strftime('%Y-%m-%d') if len(common_idx) > 0 else 'N/A',
        'overlapping_latest': common_idx.max().strftime('%Y-%m-%d') if len(common_idx) > 0 else 'N/A',
        'target_count': len(y),
        'features_count': len(x_clean),
        'common_count': len(common_idx),
        'issue': 'None'
    }
    
    logger.info(f"Final dataset: {len(y_final)} observations with {x_final.shape[1]} features")
    
    return x_final, y_final


def _create_enhanced_features(x_final, y_final, tenor, country_code, tenor_name, 
                            handle_missing, feature_details):
    """Create additional engineered features if requested."""
    logger.info("Creating enhanced features")
    
    # Create yield curve features
    logger.info("Creating yield curve-specific features")
    yield_features = create_yield_curve_features(tenor, [country_code])
    
    # Only keep rows that overlap with x_final
    if not yield_features.empty:
        yield_features = yield_features.loc[yield_features.index.intersection(x_final.index)]
    
    # Combine all enhanced features
    logger.info("Combining enhanced features")
    enhanced_dfs = [x_final]
    
    if not yield_features.empty:
        enhanced_dfs.append(yield_features)
    
    if len(enhanced_dfs) > 1:
        enhanced_df = pd.concat(enhanced_dfs, axis=1)
        x_final = combine_features(x_final, enhanced_df, handle_missing=handle_missing)
        
        # Update feature details
        feature_details['enhanced_features'] = list(yield_features.columns)
        feature_details['feature_engineering']['count'] = len(yield_features.columns)
        feature_details['feature_engineering']['types'] = {
            'yield_curve': len(yield_features.columns),
        }
        logger.info(f"Added {len(yield_features.columns)} engineered features. Final feature count: {len(x_final.columns)}")
    
    return x_final


def _perform_feature_selection(x_final, y_final, feature_selection, feature_details):
    """Perform feature selection if requested."""
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
    
    return x_selected


def _calculate_feature_correlations(x_final, y_final, feature_details):
    """Calculate and store feature correlations with target."""
    correlations = {}
    for col in x_final.columns:
        try:
            corr = x_final[col].corr(y_final)
            correlations[col] = abs(corr) if not pd.isna(corr) else 0.0
        except Exception as e:
            logger.warning(f"Could not calculate correlation for {col}: {e}")
            correlations[col] = 0.0
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Log top correlated features
    logger.info("Top 5 correlated features:")
    for i, (feature, corr) in enumerate(sorted_correlations[:5]):
        logger.info(f"  {i+1}. {feature}: {corr:.4f}")
    
    # Add to feature details
    feature_details['feature_correlations'] = {k: round(v, 4) for k, v in sorted_correlations}


def _final_data_cleanup(x_final, y_final, feature_details):
    """Perform final data quality checks and cleanup."""
    # Check data overlap and log statistics
    common_indices = x_final.index.intersection(y_final.index)
    logger.info(f"Overlapping date range: {common_indices.min().strftime('%Y-%m-%d')} to {common_indices.max().strftime('%Y-%m-%d')}")
    logger.info(f"Number of overlapping points: {len(common_indices)}")
    logger.info(f"Overlap percentage of target: {len(common_indices) / len(y_final) * 100:.2f}%")
    logger.info(f"Overlap percentage of features: {len(common_indices) / len(x_final) * 100:.2f}%")
    
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
                mean_val = x_final[col].mean() if not pd.isna(x_final[col].mean()) else 0
                x_final[col] = x_final[col].fillna(mean_val)

    # Check for infinite values
    if np.any(np.isinf(x_final.values)):
        logger.warning("Found infinite values in final features. Replacing with large finite values.")
        x_final = x_final.replace([np.inf, -np.inf], [1e9, -1e9])
    
    # Log final dimensions to confirm consistency
    logger.info(f"Final dataset shape: {x_final.shape[0]} rows × {x_final.shape[1]} columns")
    logger.info(f"Final target length: {len(y_final)}")
    
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
    
    # Add final data dimension info to feature_details
    feature_details['feature_columns'] = list(x_final.columns)
    logger.info(f"Feature columns: {feature_details['feature_columns']}")
    
    return x_final, y_final


def _save_preparation_results(country, tenor_name, feature_details):
    """Save data preparation results to file if directory exists."""
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