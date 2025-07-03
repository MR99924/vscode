"""
Forecast Generator Module

This module handles the creation of unified forecasts for growth and inflation
by combining historical data with long-term projections through smooth transitions.
It provides functionality for generating both current forecasts and historical point-in-time
forecasts to support yield curve modeling.
"""

import os
import logging
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# Import from project modules using relative imports
import config
from macrobond_fixed import Macrobond
import bloomberg_fixed
from feature_engineering import forward_fill_to_current_date

# Initialize logging
logger = logging.getLogger(__name__)

def get_bloomberg_data(bbg_client: bloomberg_fixed.Bloomberg, tickers: List[str], 
                      date_from: Union[str, pd.Timestamp], 
                      date_to: Union[str, pd.Timestamp], field: str = "PX_LAST", 
                      periodicity: str = "DAILY") -> pd.DataFrame:
    """
    Fetch data from Bloomberg for the given tickers and date range.
    
    Parameters:
        bbg_client: Already initialized Bloomberg client
        tickers: List of Bloomberg tickers
        date_from: Start date for data fetch
        date_to: End date for data fetch
        field: Bloomberg field to fetch (default: "PX_LAST")
        periodicity: Data frequency (default: "DAILY")
        
    Returns:
        DataFrame with ticker data
    """
    df = bbg_client.historicalRequest(
        tickers, field, date_from, date_to, periodicity=periodicity,
        nonTradingDayFillOption="ALL_CALENDAR_DAYS", nonTradingDayFillMethod="PREVIOUS_VALUE"
    )
    
    # Pivot the dataframe for easier manipulation
    df = pd.pivot_table(df, values='bbergvalue', index='bbergdate', columns='bbergsymbol')
    
    # Return only the requested tickers columns
    return df[tickers]

def apply_backfill_after_first_observation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply backfill but only after the first valid observation for each column.
    This preserves the integrity of the historical data.
    
    Parameters:
        df: DataFrame to process
        
    Returns:
        DataFrame with backfilled values after first observation
    """
    for column in df.columns:
        first_valid_index = df[column].first_valid_index()
        if first_valid_index is not None:
            df.loc[first_valid_index:, column] = df.loc[first_valid_index:, column].bfill()
    return df

def s_curve(x: float, steepness: float = 8.0) -> float:
    """
    S-curve function for smooth transitions between forecast regimes.
    
    Parameters:
        x: Input value between 0 and 1 representing position in transition
        steepness: Controls the steepness of the transition curve (default: 8.0)
            Higher values create a sharper transition
    
    Returns:
        Value between 0 and 1 representing the weight of the target regime
    """
    return 1 / (1 + np.exp(-steepness * (x - 0.5)))

def unify_forecasts(immediate_forecast: Union[np.ndarray, pd.Series], 
                   long_term_forecast: Union[np.ndarray, pd.Series], 
                   decay_period: int) -> np.ndarray:
    """
    Unify short and long term forecasts with an s-curve transition.
    
    Parameters:
        immediate_forecast: The near-term forecast
        long_term_forecast: The long-term forecast
        decay_period: The number of periods over which to transition
    
    Returns:
        np.array: The unified forecast
    """
    logger.info("Starting unify_forecasts function")
    
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(immediate_forecast, pd.Series):
        immediate_forecast = immediate_forecast.values
    if isinstance(long_term_forecast, pd.Series):
        long_term_forecast = long_term_forecast.values
    
    # Ensure inputs are numpy arrays
    immediate_forecast = np.array(immediate_forecast)
    long_term_forecast = np.array(long_term_forecast)
    
    logger.debug(f"Immediate forecast shape: {immediate_forecast.shape}")
    logger.debug(f"Long-term forecast shape: {long_term_forecast.shape}")
    
    # Initialize result array
    result = np.zeros_like(long_term_forecast)
    
    # Find last valid value in immediate forecast
    last_valid_idx = -1
    for i in range(len(immediate_forecast)):
        if not np.isnan(immediate_forecast[i]) and immediate_forecast[i] is not None:
            last_valid_idx = i
    
    if last_valid_idx == -1:
        logger.warning("No valid data found in immediate forecast. Using long-term forecast directly.")
        logger.debug(f"Last valid index: {last_valid_idx}")
        # If no valid data in immediate forecast, return long-term forecast
        return long_term_forecast
    
    # Get the last valid immediate forecast value
    start_value = immediate_forecast[last_valid_idx]
    logger.debug(f"Transition start value: {start_value}")

    result[0] = start_value
    last_valid_idx = 0
    
    # Calculate available points for transition
    available_points = len(result) - (last_valid_idx + 1)
    transition_length = min(decay_period, available_points)
    logger.debug(f"Transition length: {transition_length}")
    
    if transition_length <= 0:
        logger.warning("No room for transition, returning immediate forecast")
        return immediate_forecast
    
    # Apply transition using s-curve
    for i in range(transition_length):
        pos = last_valid_idx + 1 + i
        
        if pos >= len(result):
            break
            
        # Calculate weight using s-curve (normalized position in transition)
        weight = s_curve(i / transition_length)
        
        # Get target value from long-term forecast
        current_target_idx = min(pos, len(long_term_forecast) - 1)
        current_target = long_term_forecast[current_target_idx]
        
        # Calculate blended value based on position in transition
        result[pos] = start_value * (1 - weight) + current_target * weight
        
        logger.debug(f"Post-transition point {i}: {result[pos]} from lt_idx {current_target_idx}")
    
    # Copy remaining long-term forecast values after transition
    for i in range(last_valid_idx + 1 + transition_length, len(result)):
        # Ensure we don't exceed array bounds when accessing long-term forecast
        lt_idx = min(i, len(long_term_forecast) - 1)
        result[i] = long_term_forecast[lt_idx]
    
    return result

def demonstrate_unified_forecasts(
    mb_client: Macrobond,
    country_list: List[str], 
    country_code_mapping: Dict[str, str], 
    growth_forecast: pd.DataFrame, 
    growth_forecast_lt: pd.DataFrame, 
    cpi_forecast: pd.DataFrame, 
    cpi_target: pd.DataFrame, 
    cutoff_date: Optional[Union[str, pd.Timestamp]] = None, 
    wind_back_years: int = 5, 
    target_end_date: str = '2060-12-31',
    forecast_horizon: int = 360, 
    full_forecast_horizon: int = 432, 
    growth_decay_period: int = 60, 
    inflation_decay_period: int = 36
) -> Dict[str, Dict[str, Any]]:
    """
    Create unified forecasts for growth and inflation for each country.
    Includes full historical data series and supports historical point-in-time forecasting.
    
    Parameters:
        mb_client: Already initialized Macrobond client
        country_list: List of countries to process
        country_code_mapping: Mapping from country names to country codes
        growth_forecast: Historical GDP growth data
        growth_forecast_lt: Long-term GDP growth forecast data
        cpi_forecast: Historical inflation data
        cpi_target: Inflation target data
        cutoff_date: Date to use as the end of historical data (optional)
        wind_back_years: Number of years to wind back for transition (default: 5)
        target_end_date: End date for forecasts (default: '2060-12-31')
        forecast_horizon: Near-term forecast horizon in months (default: 360)
        full_forecast_horizon: Full forecast horizon in months (default: 432)
        growth_decay_period: Transition period for growth in months (default: 60)
        inflation_decay_period: Transition period for inflation in months (default: 36)
    
    Returns:
        Dictionary of unified forecasts by country
    """
    logger.info(f"Forecast horizon: {forecast_horizon}")
    logger.info(f"Full forecast horizon: {full_forecast_horizon}")
    logger.info(f"Growth decay period: {growth_decay_period}")
    
    if cutoff_date:
        cutoff_date = pd.Timestamp(cutoff_date)
        logger.info(f"Using cutoff date: {cutoff_date}")
        
    results = {}
    
    # Calculate the target end date
    target_end_date = pd.Timestamp(target_end_date)
    
    # Print available columns for debugging
    logger.debug(f"Available growth columns: {growth_forecast.columns}")
    logger.debug(f"Available CPI columns: {cpi_forecast.columns}")
    
    for country in country_list:
        logger.info(f"Processing country: {country}")
        country_code = country_code_mapping[country]
        logger.info(f"Processing country: {country} (code:{country_code})")
        
        results[country] = {
            'growth': None,
            'inflation': None,
            'transition_points': {}
        }
        
        try:
            # Find GDP column for this country
            growth_col = f"gdp_{country_code}"
            lt_growth_col = f"gdp_lt_{country_code}"
            
            # Check if columns exist
            if growth_col not in growth_forecast.columns:
                logger.warning(f"Column {growth_col} not found in growth_forecast")
                logger.debug(f"Available columns: {growth_forecast.columns}")
                continue
                
            # Get full historical GDP data (not just the last value)
            # Make a copy to avoid modifying the original
            historical_growth = growth_forecast[growth_col].copy().dropna()
            
            # If cutoff_date is provided, only use historical data up to that date
            if cutoff_date:
                if isinstance(historical_growth.index, pd.DatetimeIndex):
                    # Filter historical data based on cutoff date
                    historical_growth = historical_growth[historical_growth.index <= cutoff_date]
                    logger.info(f"Filtered historical growth data to {len(historical_growth)} points up to {cutoff_date}")
                else:
                    logger.warning(f"Cannot apply cutoff_date to non-datetime index for {country}")
            
            # Get long-term forecast data - also filter by cutoff date
            if lt_growth_col in growth_forecast_lt.columns:
                lt_growth = growth_forecast_lt[lt_growth_col].copy().dropna()
                # Filter long-term forecasts to only include those available at cutoff_date
                if cutoff_date and isinstance(lt_growth.index, pd.DatetimeIndex):
                    lt_growth = lt_growth[lt_growth.index <= cutoff_date]
                    # If we have no long-term forecast available at this date, create a default one
                    if lt_growth.empty:
                        logger.info(f"No long-term growth data available at {cutoff_date}, creating default forecast")
                        # Use the last available historical growth rate as a starting point
                        if not historical_growth.empty:
                            default_growth_rate = historical_growth.iloc[-1]
                        else:
                            # Default growth rate if no historical data
                            default_growth_rate = config.DEFAULT_GROWTH_RATE
                        lt_growth = pd.Series([default_growth_rate] * full_forecast_horizon)
            else:
                lt_growth = None
            
            if historical_growth.empty:
                logger.warning(f"No historical growth data for {country}")
                continue
                
            if lt_growth is None or lt_growth.empty:
                logger.warning(f"No long-term growth data for {country}")
                # Create a default constant forecast based on recent historical data
                if not historical_growth.empty:
                    # Use average of last 3 years or whatever is available
                    recent_years = min(36, len(historical_growth))
                    default_growth_rate = historical_growth.iloc[-recent_years:].mean()
                    logger.info(f"Using average of last {recent_years} months: {default_growth_rate} as default")
                else:
                    default_growth_rate = config.DEFAULT_GROWTH_RATE
                    
                lt_growth = pd.Series([default_growth_rate] * full_forecast_horizon)
                
            # Print diagnostics
            logger.debug(f"\n{country} - Growth Data:")
            logger.debug(f"Historical data points: {len(historical_growth)}")
            logger.debug(f"Historical data sample: {historical_growth.head()}")
            logger.debug(f"Historical data last point: {historical_growth.iloc[-1] if not historical_growth.empty else None}")
            logger.debug(f"LT forecast points: {len(lt_growth)}")
            logger.debug(f"LT growth sample: {lt_growth.head()}")
            
            # Wind back years for transition (to avoid data issues)
            # Only wind back if we have enough historical data
            if wind_back_years > 0 and len(historical_growth) > wind_back_years * 12:
                historical_end_idx = len(historical_growth) - (wind_back_years * 12)
                transition_historical = historical_growth.iloc[:historical_end_idx]
                logger.info(f"Using data up to {transition_historical.index[-1] if not transition_historical.empty else None} for transition")
            else:
                transition_historical = historical_growth
            
            # Convert to arrays for processing
            historical_growth_array = historical_growth.values
            transition_historical_array = transition_historical.values
            lt_growth_array = lt_growth.values
            
            # Calculate required length based on dates
            if isinstance(historical_growth.index, pd.DatetimeIndex):
                start_date = historical_growth.index[0]
                # For historical forecasting, extend only to a reasonable future date from cutoff_date
                if cutoff_date:
                    # Extend forecast 30 years beyond cutoff date
                    forecast_end_date = cutoff_date + pd.DateOffset(years=30)
                    complete_date_range = pd.date_range(start=start_date, end=forecast_end_date, freq='ME')
                else:
                    complete_date_range = pd.date_range(start=start_date, end=target_end_date, freq='ME')
                
                # Determine how many points we need for a complete series
                required_length = len(complete_date_range)
                
                # Use the unified forecast only up to the required length
                unified_growth = np.zeros(required_length)
                unified_growth[:] = np.nan  # Initialize with NaN
                
                # Copy historical data
                for i in range(len(historical_growth)):
                    if i < required_length:
                        unified_growth[i] = historical_growth.iloc[i]
                
                # Calculate transition start index - this is where forecasting begins
                if isinstance(historical_growth.index, pd.DatetimeIndex):
                    current_date = pd.Timestamp.now()
                    transition_idx = historical_growth.index.get_indexer([current_date], method='nearest')[0]

                for i in range(len(historical_growth)):
                    if i < required_length:
                        unified_growth[i] = historical_growth.iloc[i]
                             
                # Prepare the immediate forecast for transition
                immediate_horizon = min(forecast_horizon, len(transition_historical_array))
                if immediate_horizon > 0:
                    immediate_forecast = transition_historical_array[-immediate_horizon:]
                else:
                    # If we don't have enough transition data, use what we have
                    default_value = historical_growth.iloc[-1] if not historical_growth.empty else config.DEFAULT_GROWTH_RATE
                    immediate_forecast = np.full(forecast_horizon, default_value)
                
                # Generate transition forecast
                transition_forecast = unify_forecasts(
                    immediate_forecast,
                    lt_growth_array,
                    decay_period=growth_decay_period
                )
                
                # Fill the rest of the array with transition forecast
                remaining_length = required_length - transition_idx
                for i in range(min(remaining_length, len(transition_forecast))):
                    if transition_idx + i < required_length:
                        unified_growth[transition_idx + i] = transition_forecast[i]
                
                # If we still haven't reached the target end date, extend with the last value
                for i in range(transition_idx + len(transition_forecast), required_length):
                    if i > 0 and i-1 < len(unified_growth):  # Safety check
                        unified_growth[i] = unified_growth[i-1]
            else:
                # Fallback approach for non-datetime indices
                default_value = transition_historical_array[-1] if len(transition_historical_array) > 0 else config.DEFAULT_GROWTH_RATE
                
                if len(transition_historical_array) >= forecast_horizon:
                    forecast_data = transition_historical_array[-forecast_horizon:]
                else:
                    forecast_data = np.full(forecast_horizon, default_value)
                
                unified_growth = np.concatenate([
                    historical_growth_array,
                    unify_forecasts(
                        forecast_data,
                        lt_growth_array,
                        decay_period=growth_decay_period
                    )
                ])
                
                # Trim or extend to required length if needed
                if required_length > 0:
                    if len(unified_growth) > required_length:
                        unified_growth = unified_growth[:required_length]
                    elif len(unified_growth) < required_length:
                        extension = np.full(required_length - len(unified_growth), 
                                           unified_growth[-1] if len(unified_growth) > 0 else config.DEFAULT_GROWTH_RATE)
                        unified_growth = np.concatenate([unified_growth, extension])
            
            # Store the result
            results[country]['growth'] = unified_growth
            # Store the transition point index
            results[country]['transition_points']['growth_transition_idx'] = len(historical_growth) - 1
            
            # Now repeat for inflation data
            inflation_col = f"cpi_inf_{country_code}"
            target_inflation_col = f"cpi_target_{country_code}"
            
            # Check if columns exist
            if inflation_col not in cpi_forecast.columns:
                logger.warning(f"Column {inflation_col} not found in cpi_forecast")
                continue
                
            # Get full historical inflation data
            historical_inflation = cpi_forecast[inflation_col].copy().dropna()
            
            # If cutoff_date is provided, only use historical data up to that date
            if cutoff_date:
                if isinstance(historical_inflation.index, pd.DatetimeIndex):
                    historical_inflation = historical_inflation[historical_inflation.index <= cutoff_date]
                    logger.info(f"Filtered historical inflation data to {len(historical_inflation)} points up to {cutoff_date}")
                else:
                    logger.warning(f"Cannot apply cutoff_date to non-datetime index for {country}")
            
            # Wind back years for transition (to avoid data issues)
            if wind_back_years > 0 and len(historical_inflation) > wind_back_years * 12:
                inflation_end_idx = len(historical_inflation) - (wind_back_years * 12)
                transition_inflation_hist = historical_inflation.iloc[:inflation_end_idx]
            else:
                transition_inflation_hist = historical_inflation
            
            # Get inflation target data - also filter by cutoff date
            if target_inflation_col in cpi_target.columns:
                target_inflation = cpi_target[target_inflation_col].copy().dropna()
                # Filter targets to only include those available at cutoff_date
                if cutoff_date and isinstance(target_inflation.index, pd.DatetimeIndex):
                    target_inflation = target_inflation[target_inflation.index <= cutoff_date]
                    # If no target available at this date, create a default one
                    if target_inflation.empty:
                        logger.info(f"No inflation target available at {cutoff_date}, creating default target")
                        if not historical_inflation.empty:
                            # Use average of last 3 years or whatever is available
                            recent_years = min(36, len(historical_inflation))
                            default_target = historical_inflation.iloc[-recent_years:].mean()
                        else:
                            default_target = config.DEFAULT_INFLATION_RATE
                        target_inflation = pd.Series([default_target] * full_forecast_horizon)
            else:
                target_inflation = None
            
            if historical_inflation.empty:
                logger.warning(f"No historical inflation data for {country}")
                continue
                
            if target_inflation is None or target_inflation.empty:
                logger.warning(f"No inflation target data for {country}")
                # Create a default target based on recent historical data
                if not historical_inflation.empty:
                    # Use average of last 3 years or whatever is available
                    recent_years = min(36, len(historical_inflation))
                    default_target = historical_inflation.iloc[-recent_years:].mean()
                    logger.info(f"Using average of last {recent_years} months: {default_target} as default target")
                else:
                    default_target = config.DEFAULT_INFLATION_RATE
                    
                target_inflation = pd.Series([default_target] * full_forecast_horizon)
                
            # Print diagnostics for inflation
            logger.debug(f"\n{country} - Inflation Data:")
            logger.debug(f"Historical inflation points: {len(historical_inflation)}")
            logger.debug(f"Historical inflation sample: {historical_inflation.head()}")
            logger.debug(f"Historical inflation last point: {historical_inflation.iloc[-1] if not historical_inflation.empty else None}")
            logger.debug(f"Target inflation points: {len(target_inflation)}")
            logger.debug(f"Target inflation sample: {target_inflation.head()}")
            
            # Convert to arrays for processing
            historical_inflation_array = historical_inflation.values
            transition_inflation_array = transition_inflation_hist.values
            target_inflation_array = target_inflation.values
            
            # Use same approach as for growth to create aligned series
            if isinstance(historical_inflation.index, pd.DatetimeIndex):
                start_date = historical_inflation.index[0]
                # For historical forecasting, extend only to a reasonable future date from cutoff_date
                if cutoff_date:
                    # Extend forecast 30 years beyond cutoff date
                    forecast_end_date = cutoff_date + pd.DateOffset(years=30)
                    complete_date_range = pd.date_range(start=start_date, end=forecast_end_date, freq='ME')
                else:
                    complete_date_range = pd.date_range(start=start_date, end=target_end_date, freq='ME')
                
                unified_length = len(complete_date_range)
                unified_inflation = np.zeros(unified_length)
                unified_inflation[:] = np.nan
                
                # Copy historical data
                for i in range(len(historical_inflation)):
                    if i < unified_length:
                        unified_inflation[i] = historical_inflation.iloc[i]
                
                # Calculate transition start index
                transition_idx = len(historical_inflation)
                
                # Prepare immediate forecast
                immediate_horizon_inf = min(forecast_horizon, len(transition_inflation_array))
                if immediate_horizon_inf > 0:
                    immediate_inflation = transition_inflation_array[-immediate_horizon_inf:]
                else:
                    # If we don't have enough transition data, use what we have
                    default_value = historical_inflation.iloc[-1] if not historical_inflation.empty else config.DEFAULT_INFLATION_RATE
                    immediate_inflation = np.full(forecast_horizon, default_value)
                
                # Generate transition forecast
                transition_inflation = unify_forecasts(
                    immediate_inflation,
                    target_inflation_array,
                    decay_period=inflation_decay_period
                )
                
                # Fill with transition forecast
                for i in range(min(len(transition_inflation), unified_length - transition_idx)):
                    if transition_idx + i < unified_length:
                        unified_inflation[transition_idx + i] = transition_inflation[i]
                
                # Extend if needed
                for i in range(transition_idx + len(transition_inflation), unified_length):
                    if i > 0 and i-1 < len(unified_inflation):
                        unified_inflation[i] = unified_inflation[i-1]
            else:
                # Fallback approach for non-datetime indices
                default_value = transition_inflation_array[-1] if len(transition_inflation_array) > 0 else config.DEFAULT_INFLATION_RATE
                
                if len(transition_inflation_array) >= forecast_horizon:
                    forecast_data = transition_inflation_array[-forecast_horizon:]
                else:
                    forecast_data = np.full(forecast_horizon, default_value)
                
                unified_inflation = np.concatenate([
                    historical_inflation_array,
                    unify_forecasts(
                        forecast_data,
                        target_inflation_array,
                        decay_period=inflation_decay_period
                    )
                ])
                
                # Trim or extend to match required length
                if required_length > 0:
                    if len(unified_inflation) > required_length:
                        unified_inflation = unified_inflation[:required_length]
                    elif len(unified_inflation) < required_length:
                        extension = np.full(required_length - len(unified_inflation), 
                                           unified_inflation[-1] if len(unified_inflation) > 0 else config.DEFAULT_INFLATION_RATE)
                        unified_inflation = np.concatenate([unified_inflation, extension])
            
            # Store results
            results[country]['inflation'] = unified_inflation
            results[country]['transition_points']['inflation_transition_idx'] = len(historical_inflation) - 1
            
        except Exception as e:
            logger.error(f"Error processing forecasts for {country}: {e}")
            logger.error(traceback.format_exc())
    
    return results

def plot_unified_forecasts_fixed(results: Dict[str, Dict[str, Any]], 
                                growth_forecast: pd.DataFrame,
                                cpi_forecast: pd.DataFrame,
                                country_code_mapping: Dict[str, str]) -> None:
    """
    FIXED VERSION: Create plots using actual data dates instead of artificial ranges
    """
    for country, forecasts in results.items():
        if forecasts['growth'] is not None and forecasts['inflation'] is not None:
            # Create figure and subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # CHANGE 4: Get actual historical data dates for this country
            country_code = country_code_mapping[country]
            growth_col = f"gdp_{country_code}"
            inflation_col = f"cpi_inf_{country_code}"
            
            # Get historical data to determine actual start dates
            historical_growth = growth_forecast[growth_col].dropna() if growth_col in growth_forecast.columns else pd.Series()
            historical_inflation = cpi_forecast[inflation_col].dropna() if inflation_col in cpi_forecast.columns else pd.Series()
            
            # Determine the actual start date from historical data
            start_dates = []
            if not historical_growth.empty:
                start_dates.append(historical_growth.index[0])
            if not historical_inflation.empty:
                start_dates.append(historical_inflation.index[0])
            
            if start_dates:
                actual_start_date = min(start_dates)
            else:
                actual_start_date = pd.Timestamp(config.HISTORICAL_START_DATE)
            
            # Create proper date range based on actual data
            
            growth_dates = pd.date_range(start=historical_growth.index[0], periods=len(forecasts['growth']), freq='ME')
            inflation_dates = pd.date_range(start=historical_inflation.index[0], periods=len(forecasts['inflation']), freq='ME')
            
            logger.info(f"\nPlotting data for {country}:")
            logger.info(f"Actual start date: {actual_start_date}")
            logger.info(f"Growth data length: {len(forecasts['growth'])}")
            logger.info(f"Inflation data length: {len(forecasts['inflation'])}")
            logger.info(f"GDP Date range: {growth_dates[0]} to {growth_dates[-1]}")
            logger.info(f"CPI Date range: {inflation_dates[0]} to {inflation_dates[-1]}")
            
            # Plot growth with actual historical data alignment
            growth_data = forecasts['growth']
            ax1.plot(growth_dates, growth_data, 'b-', linewidth=1.5)
            ax1.set_title(f"{country}: Unified GDP Growth Forecast")
            ax1.set_ylabel('GDP Growth (%)')
            ax1.set_xlabel('Date')
            ax1.grid(True)
            
            # Plot inflation with actual historical data alignment
            inflation_data = forecasts['inflation']
            ax2.plot(inflation_dates, inflation_data, 'r-', linewidth=1.5)
            ax2.set_title(f"{country}: Unified Inflation Forecast")
            ax2.set_ylabel('Inflation (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            # Add vertical line at transition point (use actual historical data length)
            if not historical_growth.empty:
                # Find the transition point based on actual historical data
                historical_end_date = historical_growth.index[-1]
                # Find the closest date in our unified forecast dates
                transition_idx = forecasts['transition_points']['growth_transition_idx']
                if transition_idx < len(growth_dates):
                    transition_date = growth_dates[transition_idx]
                    ax1.axvline(x=transition_date, color='gray', linestyle='--', alpha=0.7)
                    ax1.text(transition_date, ax1.get_ylim()[1] * 0.9, "Forecast start", 
                            rotation=90, verticalalignment='top')
            
            # Format x-axis to show years appropriately
            years_span = (growth_dates[-1] - growth_dates[0]).days / 365.25
            if years_span > 50:
                ax1.xaxis.set_major_locator(mdates.YearLocator(10))
                ax2.xaxis.set_major_locator(mdates.YearLocator(10))
            elif years_span > 20:
                ax1.xaxis.set_major_locator(mdates.YearLocator(5))
                ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            else:
                ax1.xaxis.set_major_locator(mdates.YearLocator(2))
                ax2.xaxis.set_major_locator(mdates.YearLocator(2))
            
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.tick_params(axis='x', rotation=45)
                
            plt.tight_layout()
            plt.savefig(f"unified_forecasts_{country}.png")
            plt.close(fig)
        else:
            logger.info(f"Missing data for {country}")

def calculate_cumulative_growth(series: Union[List[float], np.ndarray], periods: int) -> float:
    """
    Calculate cumulative percentage growth over a specified number of periods
    by compounding annual growth rates.
    
    Parameters:
        series: Time series data of annual growth rates (in percentage)
        periods: Number of periods (years) to look ahead
    
    Returns:
        float: Cumulative percentage growth
    """
    # Check if we have enough data
    if len(series) < periods:
        logger.debug(f"Not enough data. Series length {len(series)} < periods {periods}")
        return np.nan
    
    # Check for NaN values
    nan_count = sum(1 for rate in series[:periods] if np.isnan(rate))
    if nan_count > 0:
        logger.debug(f"Found {nan_count} NaN values in the first {periods} elements")
        return np.nan
    
    # Initialize cumulative multiplier
    cumulative_multiplier = 1.0
    
    # Compound only the relevant annual rates
    for i in range(periods):
        if i < len(series):
            # Convert annual percentage to multiplier (e.g., 2.5% → 1.025)
            rate = series[i]
            annual_multiplier = 1 + (rate / 100)
            cumulative_multiplier *= annual_multiplier
    
    # Convert back to percentage
    cumulative_growth = (cumulative_multiplier - 1) * 100
    
    logger.debug(f"Calculated cumulative growth over {periods} periods: {cumulative_growth:.2f}%")
    return cumulative_growth

def get_forecast_data_for_modelling(
    mb_client: Macrobond,
    country_list: List[str],
    country_code_mapping: Dict[str, str],
    cutoff_dates: Optional[List[pd.Timestamp]] = None,
    wind_back_years: int = config.DEFAULT_WIND_BACK_YEARS,
    target_end_date: str = config.DEFAULT_FORECAST_END_DATE,
    forecast_horizon_map: Optional[Dict[str, int]] = None,
    full_forecast_horizon: int = config.FULL_FORECAST_HORIZON,
    growth_decay_period: int = config.GROWTH_DECAY_PERIOD,
    inflation_decay_period: int = config.INFLATION_DECAY_PERIOD,
    generate_plots: bool = config.GENERATE_PLOTS,
    export_summary: bool = True,
    output_dir: str = config.OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Generate point-in-time forecasts for each country and cutoff date.
    Returns:
        {
            country: {tenor: {cutoff_date: forecast_df}},
            'historical_df': flattened DataFrame for prepare_data
        }
    """
    logger.info("Starting point-in-time forecast data generation")

    if forecast_horizon_map is None:
        forecast_horizon_map = {
            'yld_2yr': config.FORECAST_HORIZON[0],
            'yld_5yr': config.FORECAST_HORIZON[1],
            'yld_10yr': config.FORECAST_HORIZON[2],
            'yld_30yr': config.FORECAST_HORIZON[3]
        }

    if cutoff_dates is None:
        cutoff_dates = pd.date_range(
            start=config.DEFAULT_HISTORICAL_FORECAST_START,
            end=config.DEFAULT_DATE_TO,
            freq='MS'
        )

    os.makedirs(output_dir, exist_ok=True)

    def fetch_series(tickers, start_dates, mapping_key):
        data = {}
        for ticker in tickers:
            start_date = start_dates.get(ticker, config.HISTORICAL_START_DATE)
            try:
                series = mb_client.FetchSeries([ticker], start_date=start_date)
                if not series.empty:
                    mapped_name = config.COLUMN_MAPPINGS[mapping_key].get(ticker, ticker)
                    data[mapped_name] = series[ticker]
                    logger.debug(f"Fetched {ticker} as {mapped_name}")
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame(data)

    growth_forecast = fetch_series(config.gdp_tickers.keys(), config.gdp_start_dates, 'gdp')
    if isinstance(growth_forecast.index, pd.DatetimeIndex):
        growth_forecast = growth_forecast.resample('ME').interpolate('linear')
        growth_forecast = growth_forecast.pct_change(periods=12, fill_method=None) * 100

    growth_forecast_lt = mb_client.FetchSeries(list(config.growth_forecast_lt_tickers.keys()))
    growth_forecast_lt = growth_forecast_lt.rename(columns=config.COLUMN_MAPPINGS['growth_forecast_lt'])

    cpi_forecast = fetch_series(config.cpi_inf_tickers.keys(), config.cpi_inf_start_dates, 'cpi_inf')
    if isinstance(cpi_forecast.index, pd.DatetimeIndex):
        cpi_forecast = cpi_forecast.pct_change(periods=12, fill_method=None) * 100

    cpi_target = mb_client.FetchSeries(list(config.cpi_target_tickers.keys()))
    cpi_target = cpi_target.rename(columns=config.COLUMN_MAPPINGS['cpi_target'])

    for df in [growth_forecast, growth_forecast_lt, cpi_forecast, cpi_target]:
        df = apply_backfill_after_first_observation(df)

    results = {}
    summary_data = []
    flat_records = []

    for country in country_list:
        country_code = country_code_mapping[country]
        gdp_col = f"gdp_{country_code}"
        cpi_col = f"cpi_inf_{country_code}"

        gdp_start = growth_forecast[gdp_col].dropna().index.min() if gdp_col in growth_forecast else None
        cpi_start = cpi_forecast[cpi_col].dropna().index.min() if cpi_col in cpi_forecast else None

        if gdp_start and cpi_start:
            earliest_cutoff = max(gdp_start, cpi_start)
        elif gdp_start:
            earliest_cutoff = gdp_start
        elif cpi_start:
            earliest_cutoff = cpi_start
        else:
            logger.warning(f"No usable data for {country}, skipping")
            continue

        valid_cutoffs = [d for d in cutoff_dates if d >= earliest_cutoff]
        if not valid_cutoffs:
            logger.warning(f"No valid cutoff dates for {country} after {earliest_cutoff}")
            continue
        results[country] = {}

        for cutoff_date in valid_cutoffs:
            logger.info(f"Generating forecast for {country} at {cutoff_date}")
            unified = demonstrate_unified_forecasts(
                mb_client=mb_client,
                country_list=[country],
                country_code_mapping=country_code_mapping,
                growth_forecast=growth_forecast,
                growth_forecast_lt=growth_forecast_lt,
                cpi_forecast=cpi_forecast,
                cpi_target=cpi_target,
                cutoff_date=cutoff_date,
                wind_back_years=wind_back_years,
                target_end_date=target_end_date,
                forecast_horizon=max(forecast_horizon_map.values()),
                full_forecast_horizon=full_forecast_horizon,
                growth_decay_period=growth_decay_period,
                inflation_decay_period=inflation_decay_period
            )

            if country not in unified or unified[country]['growth'] is None:
                continue

            for tenor, horizon in forecast_horizon_map.items():
                forecast_start = cutoff_date + pd.DateOffset(months=1)
                forecast_index = pd.date_range(start=forecast_start, periods=horizon, freq='MS')
                growth_series = pd.Series(unified[country]['growth'][:horizon], index=forecast_index)
                inflation_series = pd.Series(unified[country]['inflation'][:horizon], index=forecast_index)

                if tenor not in results[country]:
                    results[country][tenor] = {}

                forecast_df = pd.DataFrame({
                    'growth': growth_series,
                    'inflation': inflation_series
                })

                results[country][tenor][cutoff_date] = forecast_df

                # Flattened forecast columns
                period = int(tenor.replace("yld_", "").replace("yr", ""))
                g_col = f"gdp_forecast_{period}yr_{country_code}"
                i_col = f"cpi_forecast_{period}yr_{country_code}"
                flat_df = pd.DataFrame({
                    g_col: forecast_df['growth'],
                    i_col: forecast_df['inflation']
                }, index=forecast_df.index)
                flat_records.append(flat_df)

                summary_data.append({
                    'country': country,
                    'tenor': tenor,
                    'cutoff_date': cutoff_date,
                    'forecast_start': forecast_df.index[0],
                    'forecast_end': forecast_df.index[-1],
                    'forecast_periods': len(forecast_df),
                    'avg_growth_forecast': forecast_df['growth'].mean(),
                    'avg_inflation_forecast': forecast_df['inflation'].mean(),
                    'growth_forecast_std': forecast_df['growth'].std(),
                    'inflation_forecast_std': forecast_df['inflation'].std(),
                    'first_year_growth': forecast_df['growth'].head(12).mean(),
                    'first_year_inflation': forecast_df['inflation'].head(12).mean()
                })

        if generate_plots:
            logger.info("Generating forecast plots")
            plot_unified_forecasts_fixed(unified, growth_forecast, cpi_forecast, country_code_mapping)

    if export_summary and summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "forecast_summary_report.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Forecast summary report saved to {summary_path}")

    # Combine all flattened forecast records
    historical_df = pd.concat(flat_records, axis=1)
    historical_df = historical_df.loc[:, ~historical_df.columns.duplicated()]

    logger.info("Completed point-in-time forecast generation")
    results['historical_df'] = historical_df
    return results


def validate_forecast_data(forecast_data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate the generated forecast data for completeness and quality.
    
    Parameters:
        forecast_data: Dictionary of forecast DataFrames by country
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating forecast data")
    
    validation_passed = True
    
    for country, df in forecast_data.items():
        try:
            # Check for required columns
            required_columns = ['growth', 'inflation']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"{country}: Missing required columns: {missing_columns}")
                validation_passed = False
                continue
            
            # Check for excessive NaN values
            for col in required_columns:
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio > 0.1:  # More than 10% NaN values
                    logger.warning(f"{country}: High NaN ratio in {col}: {nan_ratio:.2%}")
                
            # Check for reasonable value ranges
            growth_min, growth_max = df['growth'].min(), df['growth'].max()
            if growth_min < -50 or growth_max > 50:
                logger.warning(f"{country}: Growth values outside reasonable range: [{growth_min:.2f}, {growth_max:.2f}]")
            
            inflation_min, inflation_max = df['inflation'].min(), df['inflation'].max()
            if inflation_min < -20 or inflation_max > 100:
                logger.warning(f"{country}: Inflation values outside reasonable range: [{inflation_min:.2f}, {inflation_max:.2f}]")
            
            # Check for data continuity
            if df.index.duplicated().any():
                logger.error(f"{country}: Duplicate dates found in index")
                validation_passed = False
            
            logger.debug(f"{country}: Validation passed - {len(df)} data points")
            
        except Exception as e:
            logger.error(f"Error validating data for {country}: {e}")
            logger.exception("Validation error traceback:")
            validation_passed = False
    
    if validation_passed:
        logger.info("All forecast data validation checks passed")
    else:
        logger.error("Forecast data validation failed")
    
    return validation_passed


def export_forecast_data(forecast_data: Dict[str, pd.DataFrame], output_dir: str = "forecast_output") -> None:
    """
    Export forecast data to CSV files for external use.
    
    Parameters:
        forecast_data: Dictionary of forecast DataFrames by country
        output_dir: Directory to save the output files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Exporting forecast data to {output_dir}")
        
        for country, df in forecast_data.items():
            filename = f"{country.lower().replace(' ', '_')}_forecast.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Export with proper date formatting
            df.to_csv(filepath, date_format='%Y-%m-%d')
            logger.info(f"Exported {country} data to {filepath}")
        
        # Create a summary file
        summary_data = []
        for country, df in forecast_data.items():
            summary_data.append({
                'country': country,
                'start_date': df.index[0].strftime('%Y-%m-%d'),
                'end_date': df.index[-1].strftime('%Y-%m-%d'),
                'data_points': len(df),
                'avg_growth': df['growth'].mean(),
                'avg_inflation': df['inflation'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filepath = os.path.join(output_dir, 'forecast_summary.csv')
        summary_df.to_csv(summary_filepath, index=False)
        logger.info(f"Exported summary data to {summary_filepath}")
        
    except Exception as e:
        logger.error(f"Error exporting forecast data: {e}")
        logger.exception("Export error traceback:")
        raise


def main():
    """
    Main execution function for the forecast generator.
    """
    logger.info("Starting forecast generator main execution")
    
    try:
        # Load configuration
        country_list = config.country_list
        country_code_mapping = config.COUNTRY_CODE_MAPPING
        
        logger.info(f"Processing {len(country_list)} countries: {country_list}")
        
        # Initialize Macrobond client
        mb_client = Macrobond()
        
        # Generate forecast data
        forecast_data = get_forecast_data_for_modelling(
            mb_client=mb_client,
            country_list=country_list,
            country_code_mapping=country_code_mapping,
            cutoff_date=None,  # Use current date as cutoff
            wind_back_years=config.WIND_BACK_YEARS,
            target_end_date=config.TARGET_END_DATE,
            forecast_horizon=config.FORECAST_HORIZON[3],
            full_forecast_horizon=config.FULL_FORECAST_HORIZON,
            growth_decay_period=config.GROWTH_DECAY_PERIOD,
            inflation_decay_period=config.INFLATION_DECAY_PERIOD
        )
        
        # Validate the generated data
        if validate_forecast_data(forecast_data):
            logger.info("Forecast data validation successful")
        else:
            logger.error("Forecast data validation failed - proceeding with caution")
        
        # Export results if requested
        if config.EXPORT_RESULTS:
            export_forecast_data(forecast_data, config.OUTPUT_DIRECTORY)
        
        logger.info("Forecast generator execution completed successfully")
        
        # Return data for potential use by other modules
        return forecast_data
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.exception("Main execution error traceback:")
        raise


if __name__ == "__main__":
    main()