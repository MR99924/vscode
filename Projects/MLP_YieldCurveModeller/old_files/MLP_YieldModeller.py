import sys
import os
import pandas as pd
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import numpy as np
import math
from joblib import dump, load
import datetime as dt
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from macrobond import Macrobond
import bloomberg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import traceback
import joblib
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize API connectors
mb = Macrobond()
label_encoder = LabelEncoder()

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

def apply_backfill_after_first_observation(df):
    """
    Backfill data after the first valid observation for each column.
    
    Parameters:
        df: DataFrame - Input data
        
    Returns:
        DataFrame: Data with backfilled values after first observation
    """
    result_df = df.copy()
    for column in df.columns:
        first_valid_index = df[column].first_valid_index()
        if first_valid_index is not None:
            result_df.loc[first_valid_index:, column] = df.loc[first_valid_index:, column].bfill()
    return result_df

def calculate_term_premium(df, base_yield='yld_2yr', country_code=None):
    """
    Calculate term premiums for different tenors relative to a base yield.
    
    Parameters:
        df: DataFrame - Yield data
        base_yield: str - Base yield for term premium calculations (default: 'yld_2yr')
        country_code: str - Country code to calculate premium for
        
    Returns:
        DataFrame: Term premiums for different tenors
    """
    term_premiums = pd.DataFrame(index=df.index)
    
    if country_code is None:
        return term_premiums
    
    base_col = f'{base_yield}_{country_code}'
    if base_col not in df.columns:
        logger.warning(f"Base yield column {base_col} not found")
        return term_premiums
    
    # Calculate term premiums with better error handling
    for tenor in ['5yr', '10yr', '30yr']:
        yield_col = f'yld_{tenor}_{country_code}'
        premium_col = f'term_premium_{tenor}'
        
        if yield_col in df.columns:
            try:
                valid_mask = df[yield_col].notna() & df[base_col].notna()
                term_premiums.loc[valid_mask, premium_col] = df.loc[valid_mask, yield_col] - df.loc[valid_mask, base_col]
            except Exception as e:
                logger.error(f"Error calculating {premium_col}: {e}")
        else:
            logger.warning(f"Yield column {yield_col} not found")
    
    return term_premiums

def s_curve(x):
    """
    S-curve function for smooth transitions.
    Input t should be normalized between 0 and 1.
    Returns a value between 0 and 1.
    """
    return 1 / (1 + np.exp(-8 * (x - 0.5)))

def unify_forecasts(immediate_forecast, long_term_forecast, decay_period):
    """
    Unify short and long term forecasts with an s-curve transition.
    
    Parameters:
        immediate_forecast: pandas Series or np.array - The near-term forecast
        long_term_forecast: pandas Series or np.array - The long-term forecast
        decay_period: int - The number of periods over which to transition
        
    Returns:
        np.array - The unified forecast
    """
    logging.info("Starting unify_forecasts function")
    
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(immediate_forecast, pd.Series):
        immediate_forecast = immediate_forecast.values
    if isinstance(long_term_forecast, pd.Series):
        long_term_forecast = long_term_forecast.values
    
    # Ensure immediate_forecast and long_term_forecast are numpy arrays
    immediate_forecast = np.array(immediate_forecast)
    long_term_forecast = np.array(long_term_forecast)
    
    logging.info(f"Immediate forecast shape: {immediate_forecast.shape}")
    logging.info(f"Long-term forecast shape: {long_term_forecast.shape}")
    
    # Initialize result array
    result = np.zeros_like(long_term_forecast)
    
    # Find last valid value in immediate forecast
    last_valid_idx = -1
    for i in range(len(immediate_forecast)):
        if not np.isnan(immediate_forecast[i]) and immediate_forecast[i] is not None:
            last_valid_idx = i
    
    if last_valid_idx == -1:
        logging.warning("No valid data found in immediate forecast. Using default value of 2.")
        logging.info(f"Last valid index: {last_valid_idx}")
        # If no valid data in immediate forecast, return long-term forecast
        return long_term_forecast
    
    # Get the last valid immediate forecast value
    start_value = immediate_forecast[last_valid_idx]
    logging.info(f"Transition start value: {start_value}")
    
    # Copy all valid immediate forecast values to the result
    for i in range(last_valid_idx + 1):
        result[i] = immediate_forecast[i]
    
    # Calculate available points for transition
    available_points = len(result) - (last_valid_idx + 1)
    transition_length = min(decay_period, available_points)
    logging.info(f"Transition length: {transition_length}")
    
    if transition_length <= 0:
        logging.warning("No room for transition, returning immediate forecast")
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
        
        # Add small random noise that decreases over the transition
        noise = np.random.normal(0, 0.01) if i > 0 else 0
        
        # Calculate blended value based on position in transition
        result[pos] = start_value * (1 - weight) + current_target * weight + noise
        
        logging.info(f"Post-transition point {i}: {result[pos]} from lt_idx {current_target_idx}")
    
    # Copy remaining long-term forecast values after transition
    for i in range(last_valid_idx + 1 + transition_length, len(result)):
        # Ensure we don't exceed array bounds when accessing long-term forecast
        lt_idx = min(i, len(long_term_forecast) - 1)
        result[i] = long_term_forecast[lt_idx]
    
    return result

def demonstrate_unified_forecasts(country_list, country_code_mapping, growth_forecast, growth_forecast_lt, cpi_forecast, cpi_target,
                                 cutoff_date=None, wind_back_years=5, target_end_date='2060-12-31',
                                 forecast_horizon=60, full_forecast_horizon=432, growth_decay_period=60, inflation_decay_period=36):
    """
    Create unified forecasts for growth and inflation for each country. Now includes full historical data series.
    Added cutoff_date parameter for historical forecasting, ensuring forecasts vary based on available data at each date.
    
    Parameters:
        country_list: list - List of countries to process
        country_code_mapping: dict - Mapping from country names to country codes
        growth_forecast: DataFrame - Historical GDP growth data
        growth_forecast_lt: DataFrame - Long-term GDP growth forecast data
        cpi_forecast: DataFrame - Historical inflation data
        cpi_target: DataFrame - Inflation target data
        cutoff_date: datetime or str - Date to use as the end of historical data (optional)
        wind_back_years: int - Number of years to wind back for transition (default: 5)
        target_end_date: str - End date for forecasts (default: '2060-12-31')
        forecast_horizon: int - Near-term forecast horizon in months
        full_forecast_horizon: int - Full forecast horizon in months
        growth_decay_period: int - Period over which to transition growth forecasts
        inflation_decay_period: int - Period over which to transition inflation forecasts
        
    Returns:
        dict - Dictionary of unified forecasts by country
    """
    logging.info(f"Forecast horizon: {forecast_horizon}")
    logging.info(f"Full forecast horizon: {full_forecast_horizon}")
    logging.info(f"Growth decay period: {growth_decay_period}")
    
    if cutoff_date:
        cutoff_date = pd.Timestamp(cutoff_date)
        logging.info(f"Using cutoff date: {cutoff_date}")
        
    results = {}
    
    # Calculate the target end date
    target_end_date = pd.Timestamp(target_end_date)
    
    # Print available columns for debugging
    if logging.getLogger().level == logging.DEBUG:
        logging.debug(f"Available growth columns: {growth_forecast.columns}")
        logging.debug(f"Available CPI columns: {cpi_forecast.columns}")
    
    for country in country_list:
        logging.info(f"Processing country: {country}")
        # Check if country exists in mapping
        if country not in country_code_mapping:
            logging.warning(f"Country {country} not found in country_code_mapping")
            continue
            
        country_code = country_code_mapping[country]
        logging.info(f"Processing country: {country} (code:{country_code})")
        
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
                logging.warning(f"Column {growth_col} not found in growth_forecast")
                logging.debug(f"Available columns: {growth_forecast.columns}")
                continue
                
            # Get full historical GDP data (not just the last value)
            # Make a copy to avoid modifying the original
            historical_growth = growth_forecast[growth_col].copy().dropna()
            
            # If cutoff_date is provided, only use historical data up to that date
            if cutoff_date:
                if isinstance(historical_growth.index, pd.DatetimeIndex):
                    # This is the crucial line - we filter historical data based on cutoff date
                    historical_growth = historical_growth[historical_growth.index <= cutoff_date]
                    
                    logging.info(f"Filtered historical growth data to {len(historical_growth)} points up to {cutoff_date}")
                else:
                    logging.warning(f"Cannot apply cutoff_date to non-datetime index for {country}")
            
            # Get long-term forecast data - also filter by cutoff date
            if lt_growth_col in growth_forecast_lt.columns:
                lt_growth = growth_forecast_lt[lt_growth_col].copy().dropna()
                # Filter long-term forecasts to only include those available at cutoff_date
                if cutoff_date and isinstance(lt_growth.index, pd.DatetimeIndex):
                    lt_growth = lt_growth[lt_growth.index <= cutoff_date]
                    # If we have no long-term forecast available at this date, create a default one
                    if lt_growth.empty:
                        logging.info(f"No long-term growth data available at {cutoff_date}, creating default forecast")
                        # Use the last available historical growth rate as a starting point
                        if not historical_growth.empty:
                            default_growth_rate = historical_growth.iloc[-1]
                        else:
                            default_growth_rate = 2.0  # Default if no historical data
                        lt_growth = pd.Series([default_growth_rate] * full_forecast_horizon)
            else:
                lt_growth = None
                logging.warning(f"Long-term growth column {lt_growth_col} not found for {country}")
            
            if historical_growth.empty:
                logging.warning(f"No historical growth data for {country}")
                continue
                
            if lt_growth is None or lt_growth.empty:
                logging.warning(f"No long-term growth data for {country}")
                # Create a default constant forecast based on recent historical data
                if not historical_growth.empty:
                    # Use average of last 3 years or whatever is available
                    recent_years = min(36, len(historical_growth))
                    default_growth_rate = historical_growth.iloc[-recent_years:].mean()
                    logging.info(f"Using average of last {recent_years} months: {default_growth_rate} as default")
                else:
                    default_growth_rate = 2.0
                    
                lt_growth = pd.Series([default_growth_rate] * full_forecast_horizon)
                
            # Print diagnostics
            if logging.getLogger().level <= logging.INFO:
                logging.info(f"\n{country} - Growth Data:")
                logging.info(f"Historical data points: {len(historical_growth)}")
                logging.info(f"Historical data sample: {historical_growth.head()}")
                logging.info(f"Historical data last point: {historical_growth.iloc[-1] if not historical_growth.empty else None}")
                logging.info(f"LT forecast points: {len(lt_growth)}")
                logging.info(f"LT growth sample: {lt_growth.head()}")
            
            # Wind back years for transition (to avoid data issues)
            # Only wind back if we have enough historical data
            if wind_back_years > 0 and len(historical_growth) > wind_back_years * 12:
                historical_end_idx = len(historical_growth) - (wind_back_years * 12)
                transition_historical = historical_growth.iloc[:historical_end_idx]
                logging.info(f"Using data up to {transition_historical.index[-1] if not transition_historical.empty else None} for transition")
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
                    complete_date_range = pd.date_range(start=start_date, end=forecast_end_date, freq='MS')
                else:
                    complete_date_range = pd.date_range(start=start_date, end=target_end_date, freq='MS')
                
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
                transition_idx = len(historical_growth)
                
                # Prepare the immediate forecast for transition
                immediate_horizon = min(forecast_horizon, len(transition_historical_array))
                if immediate_horizon > 0:
                    immediate_forecast = transition_historical_array[-immediate_horizon:]
                else:
                    # If we don't have enough transition data, use what we have
                    immediate_forecast = np.array([historical_growth.iloc[-1]] * forecast_horizon if not historical_growth.empty else [2.0] * forecast_horizon)
                
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
                unified_growth = np.concatenate([
                    historical_growth_array,
                    unify_forecasts(
                        transition_historical_array[-forecast_horizon:] if len(transition_historical_array) >= forecast_horizon
                        else np.full(forecast_horizon, transition_historical_array[-1] if len(transition_historical_array) > 0 else 2.0),
                        lt_growth_array,
                        decay_period=growth_decay_period
                    )
                ])
                
                # Trim or extend to required length if needed
                if required_length > 0:
                    if len(unified_growth) > required_length:
                        unified_growth = unified_growth[:required_length]
                    elif len(unified_growth) < required_length:
                        extension = np.full(required_length - len(unified_growth), unified_growth[-1] if len(unified_growth) > 0 else 2.0)
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
                logging.warning(f"Column {inflation_col} not found in cpi_forecast")
                continue
                
            # Get full historical inflation data
            historical_inflation = cpi_forecast[inflation_col].copy().dropna()
            
            # If cutoff_date is provided, only use historical data up to that date
            if cutoff_date:
                if isinstance(historical_inflation.index, pd.DatetimeIndex):
                    historical_inflation = historical_inflation[historical_inflation.index <= cutoff_date]
                    logging.info(f"Filtered historical inflation data to {len(historical_inflation)} points up to {cutoff_date}")
                else:
                    logging.warning(f"Cannot apply cutoff_date to non-datetime index for {country}")
            
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
                        logging.info(f"No inflation target available at {cutoff_date}, creating default target")
                        if not historical_inflation.empty:
                            # Use average of last 3 years or whatever is available
                            recent_years = min(36, len(historical_inflation))
                            default_target = historical_inflation.iloc[-recent_years:].mean()
                        else:
                            default_target = 2.0
                        target_inflation = pd.Series([default_target] * full_forecast_horizon)
            else:
                target_inflation = None
                logging.warning(f"Inflation target column {target_inflation_col} not found for {country}")
            
            if historical_inflation.empty:
                logging.warning(f"No historical inflation data for {country}")
                continue
                
            if target_inflation is None or target_inflation.empty:
                logging.warning(f"No inflation target data for {country}")
                # Create a default target based on recent historical data
                if not historical_inflation.empty:
                    # Use average of last 3 years or whatever is available
                    recent_years = min(36, len(historical_inflation))
                    default_target = historical_inflation.iloc[-recent_years:].mean()
                    logging.info(f"Using average of last {recent_years} months: {default_target} as default target")
                else:
                    default_target = 2.0
                    
                target_inflation = pd.Series([default_target] * full_forecast_horizon)
                
            # Print diagnostics for inflation
            if logging.getLogger().level <= logging.INFO:
                logging.info(f"\n{country} - Inflation Data:")
                logging.info(f"Historical inflation points: {len(historical_inflation)}")
                logging.info(f"Historical inflation sample: {historical_inflation.head()}")
                logging.info(f"Historical inflation last point: {historical_inflation.iloc[-1] if not historical_inflation.empty else None}")
                logging.info(f"Target inflation points: {len(target_inflation)}")
                logging.info(f"Target inflation sample: {target_inflation.head()}")
            
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
                    complete_date_range = pd.date_range(start=start_date, end=forecast_end_date, freq='MS')
                else:
                    complete_date_range = pd.date_range(start=start_date, end=target_end_date, freq='MS')
                
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
                    immediate_inflation = np.array([historical_inflation.iloc[-1]] * forecast_horizon if not historical_inflation.empty else [2.0] * forecast_horizon)
                
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
                unified_inflation = np.concatenate([
                    historical_inflation_array,
                    unify_forecasts(
                        transition_inflation_array[-forecast_horizon:] if len(transition_inflation_array) >= forecast_horizon
                        else np.full(forecast_horizon, transition_inflation_array[-1] if len(transition_inflation_array) > 0 else 2.0),
                        target_inflation_array,
                        decay_period=inflation_decay_period
                    )
                ])
                
                # Trim or extend to match required length
                if required_length > 0:
                    if len(unified_inflation) > required_length:
                        unified_inflation = unified_inflation[:required_length]
                    elif len(unified_inflation) < required_length:
                        extension = np.full(required_length - len(unified_inflation), unified_inflation[-1] if len(unified_inflation) > 0 else 2.0)
                        unified_inflation = np.concatenate([unified_inflation, extension])
            
            # Store results
            results[country]['inflation'] = unified_inflation
            results[country]['transition_points']['inflation_transition_idx'] = len(historical_inflation) - 1
            
        except Exception as e:
            logging.error(f"Error processing forecasts for {country}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def plot_unified_forecasts(results):
    """
    Create plots of the unified forecasts including full historical data.
    
    Parameters:
        results: dict - Output from demonstrate_unified_forecasts
    """
    for country, forecasts in results.items():
        if forecasts['growth'] is not None and forecasts['inflation'] is not None:
            try:
                # Create figure and subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Generate appropriate date range starting from historical data
                total_periods = max(len(forecasts['growth']), len(forecasts['inflation']))
                dates = pd.date_range(start='1947-01-01', periods=total_periods, freq='MS')
                
                # Print diagnostics
                logger.info(f"\nPlotting data for {country}:")
                logger.info(f"Growth data length: {len(forecasts['growth'])}")
                logger.info(f"Inflation data length: {len(forecasts['inflation'])}")
                logger.info(f"Date range: {dates[0]} to {dates[-1]}")
                
                # Plot growth with full historical data
                growth_data = forecasts['growth']
                growth_dates = dates[:len(growth_data)]
                ax1.plot(growth_dates, growth_data, 'b-', linewidth=1.5)
                ax1.set_title(f"{country}: Unified GDP Growth Forecast")
                ax1.set_ylabel('GDP Growth (%)')
                ax1.set_xlabel('Date')
                ax1.grid(True)
                
                # Plot inflation with full historical data
                inflation_data = forecasts['inflation']
                inflation_dates = dates[:len(inflation_data)]
                ax2.plot(inflation_dates, inflation_data, 'r-', linewidth=1.5)
                ax2.set_title(f"{country}: Unified Inflation Forecast")
                ax2.set_ylabel('Inflation (%)')
                ax2.set_xlabel('Date')
                ax2.grid(True)
                
                # Format x-axis to show years
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_locator(mdates.YearLocator(10))  # Show every 10 years
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(f"unified_forecasts_{country}.png")
                plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                logger.error(f"Error plotting forecasts for {country}: {e}")
        else:
            logger.warning(f"Missing data for {country}")

def calculate_cumulative_growth(series, periods):
    """
    Calculate cumulative percentage growth over a specified number of periods
    by compounding annual growth rates.
    
    Parameters:
        series: array-like - Time series data of annual growth rates (in percentage)
        periods: int - Number of periods (years) to look ahead
        
    Returns:
        float - Cumulative percentage growth
    """
    # Debug information
    logger.debug(f"Calculate_cumulative_growth called with series of length {len(series)}, periods={periods}")
    if len(series) > 0:
        logger.debug(f"First few values in series: {series[:min(5, len(series))]}")
    
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
            logger.debug(f"i={i}, rate={rate}, multiplier={annual_multiplier}")
            cumulative_multiplier *= annual_multiplier
    
    # Convert back to percentage
    cumulative_growth = (cumulative_multiplier - 1) * 100
    logger.debug(f"Final cumulative_multiplier={cumulative_multiplier}, cumulative_growth={cumulative_growth}%")
    
    return cumulative_growth

def save_historical_forecasts_to_csv(historical_forecasts, forecast_horizons):
    """
    Save historical forecasts to CSV files.
    
    Parameters:
        historical_forecasts: dict - Output from generate_historical_forecasts
        forecast_horizons: list - List of forecast horizons in months
    """
    dates = historical_forecasts['dates']
    
    horizon_labels = {
        24: "2yr",
        60: "5yr",
        120: "10yr",
        360: "30yr"
    }
    
    for country, data in historical_forecasts['countries'].items():
        growth_df = pd.DataFrame(index=dates)
        inflation_df = pd.DataFrame(index=dates)
        
        for horizon in forecast_horizons:
            horizon_str = str(horizon)
            horizon_label = horizon_labels.get(horizon, f"{horizon//12}yr")
            
            if horizon_str in data['growth'] and len(data['growth'][horizon_str]) == len(dates):
                growth_df[f"Growth_{horizon_label}"] = data['growth'][horizon_str]
            
            if horizon_str in data['inflation'] and len(data['inflation'][horizon_str]) == len(dates):
                inflation_df[f"Inflation_{horizon_label}"] = data['inflation'][horizon_str]
        
        # Save growth data
        growth_filename = f"{country}_historical_growth_forecasts.csv"
        growth_df.to_csv(growth_filename)
        logger.info(f"Saved historical growth forecasts for {country} to {growth_filename}")
        
        # Save inflation data
        inflation_filename = f"{country}_historical_inflation_forecasts.csv"
        inflation_df.to_csv(inflation_filename)
        logger.info(f"Saved historical inflation forecasts for {country} to {inflation_filename}")
        
        # Create combined file with both growth and inflation
        combined_df = pd.concat([growth_df, inflation_df], axis=1)
        combined_filename = f"{country}_historical_forecasts.csv"
        combined_df.to_csv(combined_filename)
        logger.info(f"Saved combined historical forecasts for {country} to {combined_filename}")

def generate_historical_forecasts(country_list, country_code_mapping, growth_forecast, growth_forecast_lt,
                                 cpi_forecast, cpi_target, start_date='1990-01-01',
                                 forecast_horizons=[24, 60, 120, 360], wind_back_years=5):
    """
    Generate forecasts from each historical month and calculate cumulative growth/inflation.
    Ensures every month from start_date has a forecast for each horizon.
    
    Parameters:
        country_list: list - List of countries to process
        country_code_mapping: dict - Mapping from country names to country codes
        growth_forecast: DataFrame - Historical GDP growth data
        growth_forecast_lt: DataFrame - Long-term GDP growth forecast data
        cpi_forecast: DataFrame - Historical inflation data
        cpi_target: DataFrame - Inflation target data
        start_date: str - Start date for historical forecasts (default: '1990-01-01')
        forecast_horizons: list - List of periods (in months) to calculate cumulative growth for
        wind_back_years: int - Number of years to wind back for transition (default: 5)
        
    Returns:
        dict - Dictionary of historical forecasts by country and horizon
    """
    # Convert start_date to datetime
    start_date = pd.Timestamp(start_date)
    
    # Get the current date (end of historical data)
    if isinstance(growth_forecast.index, pd.DatetimeIndex):
        end_date = growth_forecast.index.max()
    else:
        end_date = pd.Timestamp.now()
    
    # Generate list of monthly dates from start_date to end_date
    historical_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Initialize results dictionary
    historical_forecasts = {
        'dates': historical_dates,
        'countries': {}
    }
    
    for country in country_list:
        historical_forecasts['countries'][country] = {
            'growth': {str(h): [] for h in forecast_horizons},
            'inflation': {str(h): [] for h in forecast_horizons}
        }
    
    # Set up progress bar
    total_steps = len(historical_dates) * len(country_list)
    progress_bar = tqdm(total=total_steps, desc="Generating historical forecasts")
    
    # For each historical date
    for cutoff_date in historical_dates:
        # Generate forecasts as if we were at this historical point
        results = demonstrate_unified_forecasts(
            country_list=country_list,
            country_code_mapping=country_code_mapping,
            growth_forecast=growth_forecast.copy(),
            growth_forecast_lt=growth_forecast_lt.copy(),
            cpi_forecast=cpi_forecast.copy(),
            cpi_target=cpi_target.copy(),
            cutoff_date=cutoff_date,
            wind_back_years=wind_back_years
        )
        
        # For each country
        for country in country_list:
            if country not in results:
                # Country missing from results - add NaN values for all horizons
                for horizon in forecast_horizons:
                    horizon_str = str(horizon)
                    historical_forecasts['countries'][country]['growth'][horizon_str].append(np.nan)
                    historical_forecasts['countries'][country]['inflation'][horizon_str].append(np.nan)
                progress_bar.update(1)
                continue
            
            # Check if we have growth and inflation data
            has_growth_data = results[country]['growth'] is not None and len(results[country]['growth']) > 0
            has_inflation_data = results[country]['inflation'] is not None and len(results[country]['inflation']) > 0
            
            # For each forecast horizon
            for horizon in forecast_horizons:
                horizon_years = horizon // 12  # Convert months to years
                
                # GROWTH FORECAST
                if has_growth_data:
                    growth_data = results[country]['growth']
                    
                    # Get the transition point (where forecast starts)
                    if ('transition_points' in results[country] and
                        'growth_transition_idx' in results[country]['transition_points']):
                        transition_idx = results[country]['transition_points']['growth_transition_idx']
                    else:
                        # Default to first position if transition point not available
                        transition_idx = 0
                    
                    # Make sure we have a valid transition index
                    if transition_idx < 0:
                        transition_idx = 0
                    
                    # Extract forecast rates, ensuring we always have enough data
                    annual_growth_rates = []
                    
                    for year in range(horizon_years):
                        # Calculate index for each year's forecast
                        forecast_idx = transition_idx + 1 + (year * 12)
                        
                        # If we have data at this index, use it
                        if forecast_idx < len(growth_data):
                            rate = growth_data[forecast_idx]
                        # Otherwise use the last available value or a default
                        elif len(growth_data) > 0:
                            rate = growth_data[-1]  # Use last available value
                        else:
                            rate = 2.0  # Default fallback value
                        
                        annual_growth_rates.append(rate)
                    
                    # Calculate with the required number of years
                    cumulative_growth = calculate_cumulative_growth(annual_growth_rates, horizon_years)
                    historical_forecasts['countries'][country]['growth'][str(horizon)].append(cumulative_growth)
                else:
                    # No growth data available - use NaN
                    historical_forecasts['countries'][country]['growth'][str(horizon)].append(np.nan)
                
                # INFLATION FORECAST - Similar approach
                if has_inflation_data:
                    inflation_data = results[country]['inflation']
                    
                    # Get the transition point for inflation
                    if ('transition_points' in results[country] and
                        'inflation_transition_idx' in results[country]['transition_points']):
                        transition_idx_inf = results[country]['transition_points']['inflation_transition_idx']
                    else:
                        # Default to first position if transition point not available
                        transition_idx_inf = 0
                    
                    # Make sure we have a valid transition index
                    if transition_idx_inf < 0:
                        transition_idx_inf = 0
                    
                    # Extract forecast rates, ensuring we always have enough data
                    annual_inflation_rates = []
                    
                    for year in range(horizon_years):
                        # Calculate index for each year's forecast
                        forecast_idx = transition_idx_inf + 1 + (year * 12)
                        
                        # If we have data at this index, use it
                        if forecast_idx < len(inflation_data):
                            rate = inflation_data[forecast_idx]
                        # Otherwise use the last available value or a default
                        elif len(inflation_data) > 0:
                            rate = inflation_data[-1]  # Use last available value
                        else:
                            rate = 2.0  # Default fallback value
                        
                        annual_inflation_rates.append(rate)
                    
                    # Calculate with the required number of years
                    cumulative_inflation = calculate_cumulative_growth(annual_inflation_rates, horizon_years)
                    historical_forecasts['countries'][country]['inflation'][str(horizon)].append(cumulative_inflation)
                else:
                    # No inflation data available - use NaN
                    historical_forecasts['countries'][country]['inflation'][str(horizon)].append(np.nan)
            
            progress_bar.update(1)
    
    # Verify that each country and horizon has the correct number of forecast values
    for country in country_list:
        if country in historical_forecasts['countries']:
            for horizon in forecast_horizons:
                horizon_str = str(horizon)
                
                # Check growth forecasts
                if horizon_str in historical_forecasts['countries'][country]['growth']:
                    growth_values = historical_forecasts['countries'][country]['growth'][horizon_str]
                    if len(growth_values) != len(historical_dates):
                        logger.warning(f"Country {country}, Growth Horizon {horizon}: {len(growth_values)} values "
                                     f"(expected {len(historical_dates)})")
                        
                        # Pad with NaN if needed to ensure correct length
                        while len(growth_values) < len(historical_dates):
                            growth_values.append(np.nan)
                        
                        # Trim if too long (should not happen)
                        if len(growth_values) > len(historical_dates):
                            growth_values = growth_values[:len(historical_dates)]
                
                # Check inflation forecasts
                if horizon_str in historical_forecasts['countries'][country]['inflation']:
                    inflation_values = historical_forecasts['countries'][country]['inflation'][horizon_str]
                    if len(inflation_values) != len(historical_dates):
                        logger.warning(f"Country {country}, Inflation Horizon {horizon}: {len(inflation_values)} values "
                                     f"(expected {len(historical_dates)})")
                        
                        # Pad with NaN if needed to ensure correct length
                        while len(inflation_values) < len(historical_dates):
                            inflation_values.append(np.nan)
                        
                        # Trim if too long (should not happen)
                        if len(inflation_values) > len(historical_dates):
                            inflation_values = inflation_values[:len(historical_dates)]
    
    progress_bar.close()
    return historical_forecasts

def prepare_data(country, tenor_name, country_code_mapping, tenor, pol_rat, cpi_inf, act_track, risk_rating, tenor_forecasts=None):
    """
    Prepare data for MLP modeling by selecting and combining relevant features.
    Enhanced to include tenor-matched forecast features.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        country_code_mapping: dict - Mapping from country names to country codes
        tenor: DataFrame - Yield data for the specified tenor
        pol_rat: DataFrame - Policy rate data
        cpi_inf: DataFrame - Inflation data
        act_track: DataFrame - Economic activity tracker data
        risk_rating: DataFrame - Risk rating data
        tenor_forecasts: dict - Tenor-matched forecast features (optional)
        
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
        'feature_counts': {}
    }
    
    print(f"\n--- DATA PREPARATION DIAGNOSTICS: {country} - {tenor_name} ---")
    
    # Check if country is in the mapping
    if country not in country_code_mapping:
        print(f"ERROR: Country {country} not found in country_code_mapping")
        return pd.DataFrame(), pd.Series(), feature_details
    
    country_code = country_code_mapping[country]
    print(f"Country code: {country_code}")
    
    # Get target variable (yield for the specified tenor and country)
    yield_col = f"{tenor_name}_{country_code}"
    if yield_col not in tenor.columns:
        print(f"ERROR: Yield column {yield_col} not found in tenor data")
        return pd.DataFrame(), pd.Series(), feature_details
    
    y = tenor[yield_col].dropna()
    feature_details['target_column'] = yield_col
    
    if not y.empty:
        print(f"Target yield data: {yield_col}")
        print(f"  Date range: {y.index.min().strftime('%Y-%m-%d')} to {y.index.max().strftime('%Y-%m-%d')}")
        print(f"  Number of observations: {len(y)}")
    else:
        print(f"ERROR: No valid data found for {yield_col}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    feature_details['date_ranges']['target'] = {
        'start': y.index.min().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'end': y.index.max().strftime('%Y-%m-%d') if len(y) > 0 else 'N/A',
        'count': len(y)
    }
    
    # Extract tenor period from tenor_name
    tenor_period = tenor_name.replace('yld_', '')
    
    # Select features based on tenor
    feature_dfs = []
    source_dfs = {
        'policy_rates': pol_rat,
        'inflation': cpi_inf,
        'activity': act_track,
        'risk_rating': risk_rating
    }
    
    # Add tenor-matched forecasts if available
    if tenor_forecasts is not None and country in tenor_forecasts:
        # Add growth forecasts
        if 'growth' in tenor_forecasts[country] and tenor_period in tenor_forecasts[country]['growth']:
            growth_forecast = tenor_forecasts[country]['growth'][tenor_period]
            if not growth_forecast.empty:
                source_dfs['growth_forecast'] = pd.DataFrame({f'growth_forecast_{tenor_period}_{country_code}': growth_forecast})
        
        # Add inflation forecasts
        if 'inflation' in tenor_forecasts[country] and tenor_period in tenor_forecasts[country]['inflation']:
            inflation_forecast = tenor_forecasts[country]['inflation'][tenor_period]
            if not inflation_forecast.empty:
                source_dfs['inflation_forecast'] = pd.DataFrame({f'inflation_forecast_{tenor_period}_{country_code}': inflation_forecast})
    
    # Define which sources to use based on tenor
    if tenor_name == 'yld_2yr':
        # For 2-year yields, use policy rates, inflation, activity, and short-term forecasts
        sources = ['policy_rates', 'inflation', 'activity']
        # Add forecast sources if available
        if 'growth_forecast' in source_dfs:
            sources.append('growth_forecast')
        if 'inflation_forecast' in source_dfs:
            sources.append('inflation_forecast')
        print("Model type: 2-year yield - Using policy rates, inflation, economic activity, and short-term forecasts")
    elif tenor_name == 'yld_5yr':
        # For 5-year yields, use policy rates, inflation, risk ratings, and medium-term forecasts
        sources = ['policy_rates', 'inflation', 'risk_rating']
        # Add forecast sources if available
        if 'growth_forecast' in source_dfs:
            sources.append('growth_forecast')
        if 'inflation_forecast' in source_dfs:
            sources.append('inflation_forecast')
        print("Model type: 5-year yield - Using policy rates, inflation, risk ratings, and medium-term forecasts")
    else:  # 10yr, 30yr
        # For longer-term yields, use all available features, with emphasis on long-term forecasts
        sources = ['policy_rates', 'inflation', 'activity', 'risk_rating']
        # Add forecast sources if available
        if 'growth_forecast' in source_dfs:
            sources.append('growth_forecast')
        if 'inflation_forecast' in source_dfs:
            sources.append('inflation_forecast')
        print(f"Model type: {tenor_name} - Using all available features including long-term forecasts")
    
    feature_details['feature_sources'] = sources
    print("\nFeature sources:")
    
    # For each source, extract the relevant columns and add to features
    all_columns = []
    for source_name in sources:
        print(f"\nSource: {source_name}")
        source_df = source_dfs[source_name]
        
        if source_df is None or source_df.empty:
            print(f"  WARNING: Source {source_name} is empty or None")
            continue
            
        # For tenor-matched forecasts, we already have filtered columns
        if source_name in ['growth_forecast', 'inflation_forecast']:
            country_cols = source_df.columns
        else:
            country_cols = [col for col in source_df.columns if col.endswith(f"_{country_code}")]
        
        print(f"  Columns for {country}: {country_cols}")
        
        if country_cols:
            source_data = source_df[country_cols].copy()
            all_columns.extend(country_cols)
            
            if not source_data.empty:
                print(f"  Date range: {source_data.index.min().strftime('%Y-%m-%d')} to {source_data.index.max().strftime('%Y-%m-%d')}")
                print(f"  Number of observations before NaN removal: {len(source_data)}")
                print(f"  NaN percentage: {source_data.isna().mean().mean() * 100:.2f}%")
            else:
                print("  WARNING: No data after column filtering")
            
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
            print(f"  WARNING: No columns found for {country} in {source_name}")
    
    if not feature_dfs:
        print(f"ERROR: No features found for {country} - {tenor_name}")
        return pd.DataFrame(), pd.Series(), feature_details
    
    # Combine selected features
    x = pd.concat(feature_dfs, axis=1)
    print(f"\nCombined features:")
    print(f"  Total columns: {x.shape[1]}")
    print(f"  All columns: {list(x.columns)}")
    print(f"  Date range: {x.index.min().strftime('%Y-%m-%d')} to {x.index.max().strftime('%Y-%m-%d')}")
    print(f"  Number of rows before NaN removal: {len(x)}")
    
    feature_details['total_features'] = x.shape[1]
    feature_details['feature_columns'] = list(x.columns)
    feature_details['combined_date_range'] = {
        'start': x.index.min().strftime('%Y-%m-%d') if not x.empty else 'N/A',
        'end': x.index.max().strftime('%Y-%m-%d') if not x.empty else 'N/A',
        'count': len(x)
    }
    
    # Drop rows with NaN values
    x_clean = x.dropna()
    print(f"  Number of rows after NaN removal: {len(x_clean)}")
    print(f"  Rows removed due to NaN: {len(x) - len(x_clean)} ({((len(x) - len(x_clean)) / len(x) * 100) if len(x) > 0 else 0:.2f}%)")
    
    feature_details['clean_date_range'] = {
        'start': x_clean.index.min().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'end': x_clean.index.max().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A',
        'count': len(x_clean)
    }
    
    # Ensure indices have same format
    if not x_clean.empty and len(y) > 0:
        # Standardize index formatting to ensure proper overlap
        x_clean.index = x_clean.index.to_period('M').to_timestamp('M')
        y.index = y.index.to_period('M').to_timestamp('M')
        
        # Force string conversion to ensure exact comparison
        x_clean.index = pd.DatetimeIndex(x_clean.index.strftime('%Y-%m-%d'))
        y.index = pd.DatetimeIndex(y.index.strftime('%Y-%m-%d'))
    
    # Ensure x and y have overlapping indices
    common_indices = x_clean.index.intersection(y.index)
    print(f"\nData overlap:")
    print(f"  Target data range: {y.index.min().strftime('%Y-%m-%d')} to {y.index.max().strftime('%Y-%m-%d')} ({len(y)} points)")
    print(f"  Clean feature data range: {x_clean.index.min().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A'} to {x_clean.index.max().strftime('%Y-%m-%d') if not x_clean.empty else 'N/A'} ({len(x_clean)} points)")
    
    if len(common_indices) == 0:
        print(f"ERROR: No overlapping data found between features and target")
        
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
        
        # Try to diagnose the reason for no overlap
        if not x_clean.empty and len(y) > 0:
            target_start = y.index.min()
            target_end = y.index.max()
            feature_start = x_clean.index.min()
            feature_end = x_clean.index.max()
            
            if feature_end < target_start:
                print(f"  DIAGNOSIS: Feature data ends ({feature_end.strftime('%Y-%m-%d')}) before target data begins ({target_start.strftime('%Y-%m-%d')})")
                feature_details['overlap_diagnosis']['issue'] = 'Feature data ends before target data begins'
            elif feature_start > target_end:
                print(f"  DIAGNOSIS: Feature data begins ({feature_start.strftime('%Y-%m-%d')}) after target data ends ({target_end.strftime('%Y-%m-%d')})")
                feature_details['overlap_diagnosis']['issue'] = 'Feature data begins after target data ends'
            else:
                print(f"  DIAGNOSIS: Date ranges overlap but no common dates found. This may be due to different frequencies or sparse data.")
                feature_details['overlap_diagnosis']['issue'] = 'Date ranges overlap but no common dates'
                
                # Print some sample dates for debugging
                print(f"  Sample target dates: {y.index[:5]}")
                print(f"  Sample feature dates: {x_clean.index[:5]}")
        
        return pd.DataFrame(), pd.Series(), feature_details
    
    x_final = x_clean.loc[common_indices]
    y_final = y.loc[common_indices]
    
    print(f"  Overlapping date range: {common_indices.min().strftime('%Y-%m-%d')} to {common_indices.max().strftime('%Y-%m-%d')}")
    print(f"  Number of overlapping points: {len(common_indices)}")
    print(f"  Overlap percentage of target: {len(common_indices) / len(y) * 100:.2f}%")
    print(f"  Overlap percentage of clean features: {len(common_indices) / len(x_clean) * 100:.2f}%")
    
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
    
    print(f"\nFinal dataset:")
    print(f"  Shape: {x_final.shape[0]} rows × {x_final.shape[1]} columns")
    print(f"  Date range: {x_final.index.min().strftime('%Y-%m-%d')} to {x_final.index.max().strftime('%Y-%m-%d')}")
    print(f"  Feature correlation with target:")
    
    # Calculate and print correlations
    if not x_final.empty and len(y_final) > 0:
        correlations = {}
        for col in x_final.columns:
            correlation = x_final[col].corr(y_final)
            correlations[col] = correlation
            print(f"    {col}: {correlation:.4f}")
        
        # Sort features by absolute correlation
        sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_details['feature_correlations'] = {k: round(v, 4) for k, v in sorted_features}
        
        print("\n  Top correlated features:")
        for feature, corr in sorted_features[:min(5, len(sorted_features))]:
            print(f"    {feature}: {corr:.4f}")
            
        # Highlight any tenor-matched forecast features
        forecast_features = [f for f, _ in sorted_features if 'forecast' in f]
        if forecast_features:
            print("\n  Tenor-matched forecast features:")
            for feature in forecast_features:
                print(f"    {feature}: {correlations[feature]:.4f}")
    
    print("--- END OF DATA PREPARATION DIAGNOSTICS ---\n")
    
    return x_final, y_final, feature_details

def train_evaluate_mlp(country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, act_track, risk_rating, tenor_forecasts=None):
    """
    Train and evaluate an MLP model for a specific country and yield tenor.
    Enhanced with comprehensive diagnostics, visualizations, feature importance analysis
    and now incorporating tenor-matched forecasts.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        country_code_mapping: dict - Mapping from country names to country codes
        tenor_data: DataFrame - Yield data for the specified tenor
        pol_rat: DataFrame - Policy rate data
        cpi_inf: DataFrame - Inflation data
        act_track: DataFrame - Economic activity tracker data
        risk_rating: DataFrame - Risk rating data
        tenor_forecasts: dict - Tenor-matched forecast features (optional)
        
    Returns:
        dict: Model evaluation results and diagnostics
    """
    print(f"\n=== TRAINING MLP MODEL FOR {country} - {tenor_name} ===")
    
    # Initialize results dictionary
    results = {
        'country': country,
        'tenor': tenor_name,
        'status': 'Not Started',
        'metrics': {},
        'feature_importance': None,
        'data_summary': None,
        'error': None,
        'includes_forecasts': tenor_forecasts is not None
    }
    
    try:
        # Prepare data using our enhanced prepare_data function
        x, y, feature_details = prepare_data(
            country=country,
            tenor_name=tenor_name,
            country_code_mapping=country_code_mapping,
            tenor=tenor_data,
            pol_rat=pol_rat,
            cpi_inf=cpi_inf,
            act_track=act_track,
            risk_rating=risk_rating,
            tenor_forecasts=tenor_forecasts
        )
        
        # Store data summary in results
        results['data_summary'] = feature_details
        
        # Check if we have enough data
        if x.empty or len(y) == 0:
            print(f"Insufficient data to train model for {country} - {tenor_name}")
            results['status'] = 'Failed - Insufficient Data'
            results['error'] = 'No overlapping data between features and target'
            return results
            
        if len(x) < 30:  # Arbitrary threshold for minimum data points
            print(f"Not enough data points ({len(x)}) to train a reliable model")
            results['status'] = 'Failed - Too Few Data Points'
            results['error'] = f'Only {len(x)} data points available (minimum 30 required)'
            return results
        
        # Check if tenor-matched forecasts are being used
        forecast_columns = [col for col in x.columns if 'forecast' in col]
        results['forecast_columns'] = forecast_columns
        
        if forecast_columns:
            print(f"Using {len(forecast_columns)} tenor-matched forecast features:")
            for col in forecast_columns:
                print(f"  - {col}")
        
        print(f"Training model with {len(x)} data points and {x.shape[1]} features")
        
        # Update status
        results['status'] = 'Data Prepared'
        
        # Split data into training and test sets (80% train, 20% test)
        # For forecasting, we should use a time-based split rather than random
        train_size = int(len(x) * 0.8)
        
        x_train = x.iloc[:train_size]
        y_train = y.iloc[:train_size]
        x_test = x.iloc[train_size:]
        y_test = y.iloc[train_size:]
        
        print(f"Training set: {len(x_train)} samples")
        print(f"Test set: {len(x_test)} samples")
        
        # Scale the data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # Train two model variants: one with all features, one with only forecast features (if available)
        model_variants = []
        
        # 1. Model with all features
        model_variants.append(('full_model', x_train, x_test, x_train_scaled, x_test_scaled))
        
        # 2. Model with only forecast features (if available and sufficient)
        if len(forecast_columns) >= 2:  # At least need growth and inflation forecasts
            x_train_forecast = x_train[forecast_columns]
            x_test_forecast = x_test[forecast_columns]
            
            # Scale forecast features
            forecast_scaler = StandardScaler()
            x_train_forecast_scaled = forecast_scaler.fit_transform(x_train_forecast)
            x_test_forecast_scaled = forecast_scaler.transform(x_test_forecast)
            
            model_variants.append(('forecast_only', x_train_forecast, x_test_forecast, 
                                 x_train_forecast_scaled, x_test_forecast_scaled))
            
            print(f"Added forecast-only model variant with {len(forecast_columns)} features")
        
        # Train different MLP configurations for each variant
        best_models = {}
        
        for variant_name, variant_x_train, variant_x_test, variant_x_train_scaled, variant_x_test_scaled in model_variants:
            print(f"\nTraining {variant_name} model variant")
            
            mlp_configs = [
                {'hidden_layer_sizes': (10,), 'max_iter': 1000, 'random_state': 42},
                {'hidden_layer_sizes': (20, 10), 'max_iter': 1000, 'random_state': 42},
                {'hidden_layer_sizes': (50, 25), 'max_iter': 1000, 'random_state': 42},
                {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42}
            ]
            
            best_mlp = None
            best_score = float('inf')  # Use MSE as metric, so lower is better
            best_config = None
            
            for config in mlp_configs:
                print(f"  Trying MLP configuration: {config}")
                
                try:
                    # Train model
                    mlp = MLPRegressor(**config)
                    mlp.fit(variant_x_train_scaled, y_train)
                    
                    # Evaluate on test set
                    y_pred = mlp.predict(variant_x_test_scaled)
                    test_mse = mean_squared_error(y_test, y_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(y_test, y_pred)
                    
                    print(f"    Test RMSE: {test_rmse:.4f}")
                    print(f"    Test R²: {test_r2:.4f}")
                    
                    # Check if this is the best model so far
                    if test_mse < best_score:
                        best_score = test_mse
                        best_mlp = mlp
                        best_config = config
                        print(f"    New best model!")
                
                except Exception as e:
                    print(f"    Error training model with this configuration: {e}")
            
            if best_mlp is None:
                print(f"  All model configurations failed for {variant_name}")
            else:
                best_models[variant_name] = {
                    'model': best_mlp,
                    'config': best_config,
                    'x_train': variant_x_train,
                    'x_test': variant_x_test,
                    'x_train_scaled': variant_x_train_scaled,
                    'x_test_scaled': variant_x_test_scaled,
                    'scaler': scaler if variant_name == 'full_model' else forecast_scaler
                }
                print(f"  Selected best {variant_name} model configuration: {best_config}")
        
        # Check if any model variant succeeded
        if not best_models:
            print(f"All model variants failed")
            results['status'] = 'Failed - Training Error'
            results['error'] = 'All model configurations failed to train properly'
            return results
        
        # Use the best full model for main evaluation if available, otherwise use forecast model
        variant_name = 'full_model' if 'full_model' in best_models else 'forecast_only'
        best_variant = best_models[variant_name]
        
        mlp = best_variant['model']
        variant_x_train = best_variant['x_train']
        variant_x_test = best_variant['x_test']
        variant_x_train_scaled = best_variant['x_train_scaled']
        variant_x_test_scaled = best_variant['x_test_scaled']
        variant_scaler = best_variant['scaler']
        
        print(f"\nSelected {variant_name} as primary model")
        
        # Evaluate on training set
        y_train_pred = mlp.predict(variant_x_train_scaled)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"Training metrics:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²: {train_r2:.4f}")
        
        # Evaluate on test set again with the best model
        y_test_pred = mlp.predict(variant_x_test_scaled)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Test metrics:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²: {test_r2:.4f}")
        
        # Store metrics in results
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
            },
            'model_variant': variant_name
        }
        
        # Compare with other model variants if available
        if len(best_models) > 1:
            print("\nComparing model variants:")
            variant_metrics = {}
            
            for var_name, var_data in best_models.items():
                if var_name == variant_name:
                    continue  # Skip primary model already evaluated
                
                var_model = var_data['model']
                var_x_test_scaled = var_data['x_test_scaled']
                
                # Evaluate variant model
                var_y_test_pred = var_model.predict(var_x_test_scaled)
                var_test_mse = mean_squared_error(y_test, var_y_test_pred)
                var_test_rmse = np.sqrt(var_test_mse)
                var_test_r2 = r2_score(y_test, var_y_test_pred)
                
                print(f"  {var_name} test metrics:")
                print(f"    RMSE: {var_test_rmse:.4f}")
                print(f"    R²: {var_test_r2:.4f}")
                
                variant_metrics[var_name] = {
                    'test_mse': var_test_mse,
                    'test_rmse': var_test_rmse,
                    'test_r2': var_test_r2
                }
            
            results['variant_metrics'] = variant_metrics
        
        # Find future data for prediction (beyond the training and test sets)
        country_code = country_code_mapping[country]
        yield_col = f"{tenor_name}_{country_code}"
        all_yield_data = tenor_data[yield_col].dropna()
        
        # Identify future data (data after the last test point)
        if not x_test.empty:
            last_test_date = x_test.index[-1]
            future_yield_data = all_yield_data[all_yield_data.index > last_test_date]
        else:
            last_train_date = x_train.index[-1]
            future_yield_data = all_yield_data[all_yield_data.index > last_train_date]
        
        # Prepare future data if available
        if not future_yield_data.empty:
            print(f"\nMaking predictions on {len(future_yield_data)} future points")
            
            # Get feature data for future dates
            future_x = None
            try:
                # Re-call prepare_data but restrict to future dates
                min_future_date = future_yield_data.index.min()
                max_future_date = future_yield_data.index.max()
                
                # Extract feature data for future dates
                future_x_list = []
                for col in x.columns:
                    source_name = col.split('_')[0]  # Extract source name from column
                    
                    # Find the source dataframe
                    if source_name == 'pol':
                        source_df = pol_rat
                    elif source_name == 'cpi':
                        source_df = cpi_inf
                    elif source_name == 'act':
                        source_df = act_track
                    elif source_name in ['m', 'f', 's', 'rating']:
                        source_df = risk_rating
                    elif source_name in ['growth', 'inflation'] and 'forecast' in col:
                        # For forecast columns, find in tenor_forecasts
                        if tenor_forecasts and country in tenor_forecasts:
                            forecast_type = 'growth' if 'growth' in col else 'inflation'
                            tenor_period = tenor_name.replace('yld_', '')
                            
                            if (forecast_type in tenor_forecasts[country] and 
                                tenor_period in tenor_forecasts[country][forecast_type]):
                                
                                forecast_df = pd.DataFrame({
                                    col: tenor_forecasts[country][forecast_type][tenor_period]
                                })
                                source_df = forecast_df
                            else:
                                source_df = None
                        else:
                            source_df = None
                    else:
                        continue
                    
                    if source_df is not None and col in source_df.columns:
                        future_values = source_df.loc[
                            (source_df.index >= min_future_date) & 
                            (source_df.index <= max_future_date),
                            col
                        ]
                        if not future_values.empty:
                            future_x_list.append(future_values)
                
                if future_x_list:
                    future_x = pd.concat(future_x_list, axis=1)
                    future_x.columns = x.columns
                    
                    # Make sure all required columns are present
                    missing_cols = set(variant_x_train.columns) - set(future_x.columns)
                    if missing_cols:
                        print(f"Warning: Missing {len(missing_cols)} columns in future data")
                
                # Make predictions if we have future data
                if future_x is not None and not future_x.empty:
                    # Scale the future data using the same scaler used for training
                    future_x_scaled = variant_scaler.transform(future_x[variant_x_train.columns])
                    
                    # Make predictions
                    future_predictions = mlp.predict(future_x_scaled)
                    
                    # Calculate metrics if we have actual future values
                    future_metrics = {}
                    if len(future_predictions) == len(future_yield_data):
                        future_mse = mean_squared_error(future_yield_data, future_predictions)
                        future_rmse = np.sqrt(future_mse)
                        future_r2 = r2_score(future_yield_data, future_predictions)
                        
                        print(f"Future prediction metrics:")
                        print(f"  RMSE: {future_rmse:.4f}")
                        print(f"  R²: {future_r2:.4f}")
                        
                        future_metrics = {
                            'mse': future_mse,
                            'rmse': future_rmse,
                            'r2': future_r2
                        }
                        
                        results['metrics']['future'] = future_metrics
            except Exception as e:
                print(f"Error preparing future data: {e}")
        
        # Update status to successful
        results['status'] = 'Success'
        
        # Visualize model predictions
        try:
            # Create future data DataFrame for visualization
            future_data = None
            if future_x is not None and not future_x.empty:
                future_data = (future_x, future_yield_data)
            
            visualize_mlp_model_predictions(
                mlp_model=mlp,
                x_train=variant_x_train,
                y_train=y_train,
                x_test=variant_x_test,
                y_test=y_test,
                x_future=future_x if 'future_x' in locals() and future_x is not None else pd.DataFrame(),
                y_future=future_yield_data if 'future_yield_data' in locals() and not future_yield_data.empty else pd.Series(),
                scaler=variant_scaler,
                country=country,
                tenor_name=tenor_name
            )
            
            # If we have multiple model variants, create comparison visualization
            if len(best_models) > 1 and 'forecast_only' in best_models and 'full_model' in best_models:
                try:
                    # Get both models
                    full_model = best_models['full_model']['model']
                    forecast_model = best_models['forecast_only']['model']
                    
                    # Get scaled inputs
                    full_x_test_scaled = best_models['full_model']['x_test_scaled']
                    forecast_x_test_scaled = best_models['forecast_only']['x_test_scaled']
                    
                    # Make predictions
                    full_preds = full_model.predict(full_x_test_scaled)
                    forecast_preds = forecast_model.predict(forecast_x_test_scaled)
                    
                    # Create comparison plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(y_test.index, y_test, 'b-', label='Actual', linewidth=2)
                    plt.plot(y_test.index, full_preds, 'r--', label='Full Model Predictions', linewidth=2)
                    plt.plot(y_test.index, forecast_preds, 'g--', label='Forecast-Only Predictions', linewidth=2)
                    
                    plt.title(f"{country} - {tenor_name}: Model Comparison", fontsize=14)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Yield (%)', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True)
                    
                    # Format x-axis to show years
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
                    plt.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    plt.savefig(f"{country}_{tenor_name}_model_comparison.png")
                    plt.close()
                    
                    print(f"Model comparison visualization saved to {country}_{tenor_name}_model_comparison.png")
                    
                    # Calculate and display the difference in predictive power
                    full_rmse = np.sqrt(mean_squared_error(y_test, full_preds))
                    forecast_rmse = np.sqrt(mean_squared_error(y_test, forecast_preds))
                    rmse_diff = full_rmse - forecast_rmse
                    rmse_pct = (rmse_diff / full_rmse) * 100
                    
                    print(f"\nModel comparison analysis:")
                    if rmse_diff < 0:
                        print(f"  Full model outperforms forecast-only model by {abs(rmse_pct):.2f}% (RMSE)")
                    else:
                        print(f"  Forecast-only model outperforms full model by {rmse_pct:.2f}% (RMSE)")
                        
                    # Calculate contribution of forecasts to predictive power
                    forecast_columns_corr = 0
                    for col in forecast_columns:
                        if col in variant_x_test.columns:
                            corr = variant_x_test[col].corr(y_test)
                            forecast_columns_corr += abs(corr)
                    
                    total_corr = 0
                    for col in variant_x_test.columns:
                        corr = variant_x_test[col].corr(y_test)
                        total_corr += abs(corr)
                    
                    if total_corr > 0:
                        forecast_contribution = (forecast_columns_corr / total_corr) * 100
                        print(f"  Forecast features contribute approximately {forecast_contribution:.2f}% of total predictive power")
                    
                except Exception as e:
                    print(f"Error creating model comparison: {e}")
                
        except Exception as e:
            print(f"Error visualizing model predictions: {e}")
        
        # Generate feature importance report
        try:
            importance_results = generate_feature_importance_report(
                mlp_model=mlp,
                x_data=variant_x_train,
                y_data=y_train,
                scaler=variant_scaler,
                country=country,
                tenor_name=tenor_name
            )
            
            results['feature_importance'] = importance_results
            
            # Analyze the importance of forecast features specifically
            if forecast_columns:
                forecast_importance = {}
                for feature in forecast_columns:
                    if feature in importance_results.get('consensus_ranking', {}):
                        forecast_importance[feature] = importance_results['consensus_ranking'][feature]
                
                results['forecast_importance'] = forecast_importance
                
                print("\nForecast feature importance:")
                for feature, importance in sorted(forecast_importance.items(), 
                                               key=lambda x: abs(x[1]), reverse=True):
                    print(f"  {feature}: {importance:.4f}")
        except Exception as e:
            print(f"Error generating feature importance report: {e}")
        
        # Save trained model for future use
        try:
            model_filename = f"{country}_{tenor_name}_mlp_model.pkl"
            with open(model_filename, 'wb') as f:
                joblib.dump((mlp, variant_scaler, feature_details), f)
            print(f"Model saved to {model_filename}")
            
            # If we have multiple model variants, save them too
            if len(best_models) > 1:
                for var_name, var_data in best_models.items():
                    if var_name == variant_name:
                        continue  # Skip primary model already saved
                    
                    var_model_filename = f"{country}_{tenor_name}_{var_name}_model.pkl"
                    with open(var_model_filename, 'wb') as f:
                        joblib.dump((var_data['model'], var_data['scaler'], feature_details), f)
                    print(f"{var_name} model saved to {var_model_filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
    except Exception as e:
        print(f"Error training and evaluating model: {e}")
        results['status'] = 'Failed - Exception'
        results['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    print(f"=== END OF MODEL TRAINING FOR {country} - {tenor_name} ===\n")
    
    return results

def create_tenor_matched_forecasts(country_list, country_code_mapping, growth_forecast, growth_forecast_lt, cpi_forecast, cpi_target, cutoff_date=None):
    """
    Create tenor-matched forecast features by calculating average expected growth and inflation
    over each yield tenor's maturity period (2yr, 5yr, 10yr, 30yr).
    
    This function combines short-term and long-term forecasts to create consistent
    forward-looking time series that align with bond maturities.
    
    Parameters:
        country_list (list): List of countries to process
        country_code_mapping (dict): Mapping from country names to country codes
        growth_forecast (DataFrame): Short-term GDP growth forecast data
        growth_forecast_lt (DataFrame): Long-term GDP growth forecast data
        cpi_forecast (DataFrame): Historical inflation data
        cpi_target (DataFrame): Long-term inflation target data
        cutoff_date (datetime or str, optional): Date to use as the end of historical data
            
    Returns:
        dict: Dictionary containing tenor-matched forecasts with structure:
            {
                'country_name': {
                    'growth': {
                        '2yr': DataFrame, # 2-year growth expectations
                        '5yr': DataFrame, # 5-year growth expectations
                        '10yr': DataFrame, # 10-year growth expectations
                        '30yr': DataFrame  # 30-year growth expectations
                    },
                    'inflation': {
                        '2yr': DataFrame, # 2-year inflation expectations
                        '5yr': DataFrame, # 5-year inflation expectations
                        '10yr': DataFrame, # 10-year inflation expectations
                        '30yr': DataFrame  # 30-year inflation expectations
                    }
                },
                ...
            }
    """
    # Initialize results dictionary
    tenor_forecasts = {}
    
    # Define tenor horizons in years
    tenor_horizons = {
        '2yr': 2,
        '5yr': 5,
        '10yr': 10,
        '30yr': 30
    }
    
    # Convert cutoff_date to datetime if provided
    if cutoff_date is not None:
        cutoff_date = pd.Timestamp(cutoff_date)
    
    logger.info("Creating tenor-matched forecast features...")
    
    # Process each country
    for country in country_list:
        if country not in country_code_mapping:
            continue
            
        country_code = country_code_mapping[country]
        logger.info(f"Processing {country} (code: {country_code}) forecasts...")
        
        tenor_forecasts[country] = {
            'growth': {},
            'inflation': {}
        }
        
        # 1. Process growth forecasts for each tenor
        for tenor_name, horizon_years in tenor_horizons.items():
            # Horizon in months
            horizon_months = horizon_years * 12
            
            # Get relevant forecast columns
            growth_col = f"gdp_{country_code}"
            lt_growth_col = f"gdp_lt_{country_code}"
            
            # Extract and process data with proper error handling
            try:
                # Get historical growth data
                if growth_col in growth_forecast.columns:
                    historical_growth = growth_forecast[growth_col].copy().dropna()
                    
                    # Apply cutoff date if provided
                    if cutoff_date is not None and isinstance(historical_growth.index, pd.DatetimeIndex):
                        historical_growth = historical_growth[historical_growth.index <= cutoff_date]
                else:
                    historical_growth = pd.Series(dtype=float)
                
                # Get long-term growth forecast
                if lt_growth_col in growth_forecast_lt.columns:
                    lt_growth = growth_forecast_lt[lt_growth_col].copy().dropna()
                    
                    # Apply cutoff date if provided
                    if cutoff_date is not None and isinstance(lt_growth.index, pd.DatetimeIndex):
                        lt_growth = lt_growth[lt_growth.index <= cutoff_date]
                else:
                    lt_growth = pd.Series(dtype=float)
                
                # Check if we have enough data
                if historical_growth.empty:
                    logger.warning(f"No historical growth data for {country}")
                    tenor_forecasts[country]['growth'][tenor_name] = pd.Series(dtype=float)
                    continue
                
                # If we have limited or no long-term forecast, create a default one
                if lt_growth.empty or len(lt_growth) < horizon_months:
                    logger.warning(f"Limited or no long-term growth data for {country} - creating default")
                    # Use average of last 3 years or whatever is available
                    default_rate = historical_growth.iloc[-min(36, len(historical_growth)):].mean()
                    lt_growth = pd.Series([default_rate] * horizon_months, 
                                         index=pd.date_range(start=historical_growth.index[-1] + pd.DateOffset(months=1), 
                                                           periods=horizon_months, freq='MS'))
                
                # Create unified forecast
                # We'll use historical data for the start, then blend into long-term forecast
                dates = pd.date_range(start=historical_growth.index[0], periods=len(historical_growth) + horizon_months, freq='MS')
                unified_forecast = pd.Series(index=dates, dtype=float)
                
                # Fill historical part
                unified_forecast.loc[historical_growth.index] = historical_growth
                
                # Get future dates
                future_dates = dates[~dates.isin(historical_growth.index)]
                
                # Fill future part with appropriate long-term forecast values
                for i, date in enumerate(future_dates):
                    if i < len(lt_growth):
                        unified_forecast.loc[date] = lt_growth.iloc[i]
                    else:
                        # If we run out of long-term forecast, use the last available value
                        unified_forecast.loc[date] = lt_growth.iloc[-1] if not lt_growth.empty else default_rate
                
                # Calculate rolling average forecast over the tenor horizon
                # This gives us, for each date, the average expected growth over the next X years
                rolling_forecast = unified_forecast.rolling(window=horizon_months, min_periods=1).mean()
                
                # Store the result
                tenor_forecasts[country]['growth'][tenor_name] = rolling_forecast
                logger.info(f"  Created {tenor_name} growth forecast for {country} with {len(rolling_forecast)} points")
                
            except Exception as e:
                logger.error(f"Error creating {tenor_name} growth forecast for {country}: {e}")
                tenor_forecasts[country]['growth'][tenor_name] = pd.Series(dtype=float)
        
        # 2. Process inflation forecasts for each tenor
        for tenor_name, horizon_years in tenor_horizons.items():
            # Horizon in months
            horizon_months = horizon_years * 12
            
            # Get relevant forecast columns
            inflation_col = f"cpi_inf_{country_code}"
            target_inflation_col = f"cpi_target_{country_code}"
            
            # Extract and process data with proper error handling
            try:
                # Get historical inflation data
                if inflation_col in cpi_forecast.columns:
                    historical_inflation = cpi_forecast[inflation_col].copy().dropna()
                    
                    # Apply cutoff date if provided
                    if cutoff_date is not None and isinstance(historical_inflation.index, pd.DatetimeIndex):
                        historical_inflation = historical_inflation[historical_inflation.index <= cutoff_date]
                else:
                    historical_inflation = pd.Series(dtype=float)
                
                # Get inflation target
                if target_inflation_col in cpi_target.columns:
                    target_inflation = cpi_target[target_inflation_col].copy().dropna()
                    
                    # Apply cutoff date if provided
                    if cutoff_date is not None and isinstance(target_inflation.index, pd.DatetimeIndex):
                        target_inflation = target_inflation[target_inflation.index <= cutoff_date]
                else:
                    target_inflation = pd.Series(dtype=float)
                
                # Check if we have enough data
                if historical_inflation.empty:
                    logger.warning(f"No historical inflation data for {country}")
                    tenor_forecasts[country]['inflation'][tenor_name] = pd.Series(dtype=float)
                    continue
                
                # If we have limited or no inflation target, create a default one
                if target_inflation.empty or len(target_inflation) < horizon_months:
                    logger.warning(f"Limited or no inflation target data for {country} - creating default")
                    # Use average of last 3 years or whatever is available
                    default_target = historical_inflation.iloc[-min(36, len(historical_inflation)):].mean()
                    target_inflation = pd.Series([default_target] * horizon_months, 
                                               index=pd.date_range(start=historical_inflation.index[-1] + pd.DateOffset(months=1), 
                                                                 periods=horizon_months, freq='MS'))
                
                # Create unified forecast
                # We'll use historical data for the start, then blend into target inflation
                dates = pd.date_range(start=historical_inflation.index[0], periods=len(historical_inflation) + horizon_months, freq='MS')
                unified_forecast = pd.Series(index=dates, dtype=float)
                
                # Fill historical part
                unified_forecast.loc[historical_inflation.index] = historical_inflation
                
                # Get future dates
                future_dates = dates[~dates.isin(historical_inflation.index)]
                
                # Fill future part with appropriate target values
                for i, date in enumerate(future_dates):
                    if i < len(target_inflation):
                        unified_forecast.loc[date] = target_inflation.iloc[i]
                    else:
                        # If we run out of target data, use the last available value
                        unified_forecast.loc[date] = target_inflation.iloc[-1] if not target_inflation.empty else default_target
                
                # Calculate rolling average forecast over the tenor horizon
                # This gives us, for each date, the average expected inflation over the next X years
                rolling_forecast = unified_forecast.rolling(window=horizon_months, min_periods=1).mean()
                
                # Store the result
                tenor_forecasts[country]['inflation'][tenor_name] = rolling_forecast
                logger.info(f"  Created {tenor_name} inflation forecast for {country} with {len(rolling_forecast)} points")
                
            except Exception as e:
                logger.error(f"Error creating {tenor_name} inflation forecast for {country}: {e}")
                tenor_forecasts[country]['inflation'][tenor_name] = pd.Series(dtype=float)
    
    # Create a consolidated DataFrame for easier usage in models
    consolidated_forecasts = {
        'growth': {},
        'inflation': {}
    }
    
    # Consolidate growth forecasts
    for tenor_name in tenor_horizons.keys():
        # Create empty DataFrames to hold all countries
        growth_df = pd.DataFrame()
        inflation_df = pd.DataFrame()
        
        for country in country_list:
            if country not in tenor_forecasts:
                continue
                
            country_code = country_code_mapping[country]
            
            # Add growth forecast if available
            if tenor_name in tenor_forecasts[country]['growth']:
                growth_series = tenor_forecasts[country]['growth'][tenor_name]
                if not growth_series.empty:
                    growth_df[f'growth_forecast_{tenor_name}_{country_code}'] = growth_series
            
            # Add inflation forecast if available
            if tenor_name in tenor_forecasts[country]['inflation']:
                inflation_series = tenor_forecasts[country]['inflation'][tenor_name]
                if not inflation_series.empty:
                    inflation_df[f'inflation_forecast_{tenor_name}_{country_code}'] = inflation_series
        
        # Store consolidated forecasts
        consolidated_forecasts['growth'][tenor_name] = growth_df
        consolidated_forecasts['inflation'][tenor_name] = inflation_df
    
    # Apply date alignment to ensure compatibility with other data
    for forecast_type in consolidated_forecasts:
        for tenor_name in consolidated_forecasts[forecast_type]:
            df = consolidated_forecasts[forecast_type][tenor_name]
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                # Convert to month start dates to match other data
                df.index = df.index.to_period('M').to_timestamp('M')
                
                # Ensure same date format as other dataframes
                df.index = pd.DatetimeIndex(df.index.strftime('%Y-%m-%d'))
    
    logger.info("Tenor-matched forecast features created successfully")
    
    return tenor_forecasts, consolidated_forecasts

def visualize_mlp_model_predictions(mlp_model, x_train, y_train, x_test, y_test, x_future, y_future, scaler, country, tenor_name):
    """
    Create visualizations of MLP model performance including:
    1. Training data fit
    2. Test data predictions
    3. Future predictions
    
    Parameters:
        mlp_model: Trained MLPRegressor model
        x_train, y_train: Training data
        x_test, y_test: Test data
        x_future, y_future: Future data for prediction
        scaler: The StandardScaler used to scale the input data
        country: Country name
        tenor_name: Yield tenor name
    """
    print(f"\n--- VISUALIZING MLP MODEL FOR {country} - {tenor_name} ---")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # 1. Training data fit
    if not x_train.empty and len(y_train) > 0:
        print("Generating training data fit visualization...")
        x_train_scaled = scaler.transform(x_train)
        y_train_pred = mlp_model.predict(x_train_scaled)
        
        axes[0].plot(x_train.index, y_train, 'b-', label='Actual', linewidth=2)
        axes[0].plot(x_train.index, y_train_pred, 'r--', label='Model fit', linewidth=2)
        axes[0].set_title(f"{country} - {tenor_name}: Training Data Fit (1990-2014)", fontsize=14)
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Yield (%)', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True)
        
        # Calculate and display RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
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
    else:
        print("No training data available for visualization")
        axes[0].text(0.5, 0.5, "No training data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axes[0].transAxes, fontsize=14)
    
    # 2. Test data predictions
    if not x_test.empty and len(y_test) > 0:
        print("Generating test data predictions visualization...")
        x_test_scaled = scaler.transform(x_test)
        y_test_pred = mlp_model.predict(x_test_scaled)
        
        axes[1].plot(x_test.index, y_test, 'b-', label='Actual', linewidth=2)
        axes[1].plot(x_test.index, y_test_pred, 'r--', label='Predicted', linewidth=2)
        axes[1].set_title(f"{country} - {tenor_name}: Test Data Predictions", fontsize=14)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Yield (%)', fontsize=12)
        axes[1].legend(fontsize=12)
        axes[1].grid(True)
        
        # Calculate and display RMSE
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
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
    else:
        print("No test data available for visualization")
        axes[1].text(0.5, 0.5, "No test data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axes[1].transAxes, fontsize=14)
    
    # 3. Future predictions (2015-present)
    if not x_future.empty:
        print("Generating future predictions visualization...")
        x_future_scaled = scaler.transform(x_future)
        y_future_pred = mlp_model.predict(x_future_scaled)
        
        axes[2].plot(x_future.index, y_future_pred, 'r--', label='Predicted', linewidth=2)
        
        # If actual future data is available, plot it
        if not y_future.empty:
            axes[2].plot(y_future.index, y_future, 'b-', label='Actual', linewidth=2)
            
            # Calculate and display RMSE if we have actual future data
            if len(y_future) == len(y_future_pred):
                future_rmse = np.sqrt(mean_squared_error(y_future, y_future_pred))
                future_r2 = r2_score(y_future, y_future_pred)
                print(f"  Future RMSE: {future_rmse:.4f}")
                print(f"  Future R²: {future_r2:.4f}")
                
                axes[2].annotate(f"RMSE: {future_rmse:.4f}\nR²: {future_r2:.4f}", 
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               verticalalignment='top', horizontalalignment='left',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                               fontsize=12)
        
        axes[2].set_title(f"{country} - {tenor_name}: Future Predictions (2015-Present)", fontsize=14)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_ylabel('Yield (%)', fontsize=12)
        axes[2].legend(fontsize=12)
        axes[2].grid(True)
        
        # Format x-axis to show years
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[2].xaxis.set_major_locator(mdates.YearLocator(1))
        axes[2].tick_params(axis='x', rotation=45)
    else:
        print("No future data available for visualization")
        axes[2].text(0.5, 0.5, "No future data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axes[2].transAxes, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{country}_{tenor_name}_mlp_model_analysis.png")
    plt.close(fig)
    
    print(f"Model visualization saved to {country}_{tenor_name}_mlp_model_analysis.png")
    print("--- END OF MODEL VISUALIZATION ---\n")


def analyze_data_overlap_issues(country_list, country_code_mapping, yield_list, yield_names, pol_rat, cpi_inf, act_track, risk_rating):
    """
    Analyze data overlap issues to diagnose why some models have no overlapping data.
    Generates a comprehensive report with date ranges for each data source.
    
    Parameters:
        country_list: List of countries
        country_code_mapping: Mapping from country names to codes
        yield_list: List of yield DataFrames
        yield_names: List of yield tenor names
        pol_rat, cpi_inf, act_track, risk_rating: Feature DataFrames
    
    Returns:
        dict: Detailed analysis results for each country-tenor combination
    """
    print("\n=== DATA OVERLAP ANALYSIS ===")
    
    # Create results container
    overlap_results = {}
    
    # Organize feature sources
    feature_sources = {
        'policy_rates': pol_rat,
        'inflation': cpi_inf,
        'activity': act_track,
        'risk_rating': risk_rating
    }
    
    # Define which features to use for each tenor
    tenor_features = {
        'yld_2yr': ['policy_rates', 'inflation', 'activity'],
        'yld_5yr': ['policy_rates', 'inflation', 'risk_rating'],
        'yld_10yr': ['policy_rates', 'inflation', 'activity', 'risk_rating'],
        'yld_30yr': ['policy_rates', 'inflation', 'activity', 'risk_rating']
    }
    
    for country in country_list:
        if country not in country_code_mapping:
            continue
            
        country_code = country_code_mapping[country]
        print(f"\nAnalyzing data overlap for {country} (code: {country_code})")
        
        overlap_results[country] = {}
        
        for tenor_name, tenor_data in zip(yield_names, yield_list):
            print(f"\n  {tenor_name}:")
            
            country_tenor_result = {
                'status': 'Unknown',
                'issue': None,
                'yield_data': {},
                'features': {},
                'overlap_analysis': {}
            }
            
            # Get yield data date range
            yield_col = f"{tenor_name}_{country_code}"
            
            if yield_col not in tenor_data.columns:
                print(f"    No {tenor_name} data found for {country}")
                country_tenor_result['status'] = 'Missing Yield Data'
                country_tenor_result['issue'] = f"Column {yield_col} not found in yield data"
                overlap_results[country][tenor_name] = country_tenor_result
                continue
                
            yield_series = tenor_data[yield_col].dropna()
            
            if yield_series.empty:
                print(f"    {tenor_name} data exists but is empty for {country}")
                country_tenor_result['status'] = 'Empty Yield Data'
                country_tenor_result['issue'] = f"No valid values in {yield_col}"
                overlap_results[country][tenor_name] = country_tenor_result
                continue
                
            # Print yield data info
            yield_start = yield_series.index.min().strftime('%Y-%m-%d')
            yield_end = yield_series.index.max().strftime('%Y-%m-%d')
            yield_count = len(yield_series)
            
            print(f"    Target yield data: {yield_start} to {yield_end} ({yield_count} points)")
            
            country_tenor_result['yield_data'] = {
                'column': yield_col,
                'start_date': yield_start,
                'end_date': yield_end,
                'count': yield_count
            }
            
            # Collect feature sets that will be used for this tenor
            required_features = tenor_features.get(tenor_name, [])
            all_feature_data = {}
            feature_stats = {}
            
            for source_name in required_features:
                source_df = feature_sources[source_name]
                
                if source_df is None or source_df.empty:
                    print(f"    Feature source {source_name} is empty or None")
                    feature_stats[source_name] = {
                        'status': 'Missing Feature Source',
                        'issue': f"{source_name} data is empty or None"
                    }
                    continue
                    
                # Get columns for this country
                country_cols = [col for col in source_df.columns if col.endswith(f"_{country_code}")]
                
                if not country_cols:
                    print(f"    No {source_name} columns found for {country}")
                    feature_stats[source_name] = {
                        'status': 'No Country Columns',
                        'issue': f"No columns ending with _{country_code} in {source_name}"
                    }
                    continue
                    
                # Get data for these columns
                source_data = source_df[country_cols].copy()
                
                if source_data.empty:
                    print(f"    {source_name} data exists but is empty for {country}")
                    feature_stats[source_name] = {
                        'status': 'Empty Feature Data',
                        'issue': f"No data in {source_name} for {country}"
                    }
                    continue
                    
                # Print source data info
                source_start = source_data.index.min().strftime('%Y-%m-%d')
                source_end = source_data.index.max().strftime('%Y-%m-%d')
                source_count = len(source_data)
                nan_percentage = source_data.isna().mean().mean() * 100
                
                print(f"    {source_name}: {source_start} to {source_end} ({source_count} points, {nan_percentage:.2f}% NaN)")
                
                # Store data for overlap calculation
                all_feature_data[source_name] = source_data
                
                feature_stats[source_name] = {
                    'start_date': source_start,
                    'end_date': source_end,
                    'count': source_count,
                    'nan_pct': nan_percentage,
                    'columns': country_cols
                }
            
            country_tenor_result['features'] = feature_stats
            
            # If any required feature source is missing, record issue and continue
            missing_sources = [s for s in required_features if s not in all_feature_data]
            if missing_sources:
                issue_msg = f"Missing feature sources: {', '.join(missing_sources)}"
                print(f"    {issue_msg}")
                country_tenor_result['status'] = 'Missing Feature Data'
                country_tenor_result['issue'] = issue_msg
                overlap_results[country][tenor_name] = country_tenor_result
                continue
            
            # Combine all feature data
            combined_features = pd.concat(all_feature_data.values(), axis=1)
            
            # Clean NaN values
            clean_features = combined_features.dropna()
            
            if clean_features.empty:
                print(f"    All feature data contains NaN values, no clean rows available")
                country_tenor_result['status'] = 'No Clean Feature Data'
                country_tenor_result['issue'] = "All feature data contains NaN values"
                overlap_results[country][tenor_name] = country_tenor_result
                continue
                
            print(f"    Combined features: {combined_features.shape[1]} columns, {combined_features.shape[0]} rows")
            print(f"    Clean features (no NaN): {clean_features.shape[0]} rows ({clean_features.shape[0]/combined_features.shape[0]*100:.2f}%)")
            
            # Check for date overlap
            common_indices = clean_features.index.intersection(yield_series.index)
            overlap_count = len(common_indices)
            
            if overlap_count == 0:
                yield_min = yield_series.index.min()
                yield_max = yield_series.index.max()
                features_min = clean_features.index.min()
                features_max = clean_features.index.max()
                
                print(f"    NO OVERLAP found between features and target!")
                print(f"    Yield data range: {yield_start} to {yield_end}")
                print(f"    Clean features range: {features_min.strftime('%Y-%m-%d')} to {features_max.strftime('%Y-%m-%d')}")
                
                if features_max < yield_min:
                    issue = "Feature data ends before yield data begins"
                    print(f"    DIAGNOSIS: {issue}")
                elif features_min > yield_max:
                    issue = "Feature data begins after yield data ends"
                    print(f"    DIAGNOSIS: {issue}")
                else:
                    issue = "Date ranges overlap but no common dates"
                    print(f"    DIAGNOSIS: {issue} - This may be due to different frequencies or missing data points")
                
                country_tenor_result['status'] = 'No Overlap'
                country_tenor_result['issue'] = issue
                country_tenor_result['overlap_analysis'] = {
                    'yield_range': [yield_start, yield_end],
                    'feature_range': [features_min.strftime('%Y-%m-%d'), features_max.strftime('%Y-%m-%d')],
                    'common_count': 0,
                    'gap_size': None
                }
                
                # Calculate gap size if applicable
                if features_max < yield_min:
                    gap_days = (yield_min - features_max).days
                    country_tenor_result['overlap_analysis']['gap_size'] = f"{gap_days} days"
                    print(f"    Gap size: {gap_days} days")
                elif features_min > yield_max:
                    gap_days = (features_min - yield_max).days
                    country_tenor_result['overlap_analysis']['gap_size'] = f"{gap_days} days"
                    print(f"    Gap size: {gap_days} days")
            else:
                overlap_start = common_indices.min().strftime('%Y-%m-%d')
                overlap_end = common_indices.max().strftime('%Y-%m-%d')
                overlap_pct_yield = overlap_count / yield_count * 100
                overlap_pct_features = overlap_count / len(clean_features) * 100
                
                print(f"    OVERLAP found: {overlap_count} points ({overlap_pct_yield:.2f}% of yield data, {overlap_pct_features:.2f}% of clean features)")
                print(f"    Overlap date range: {overlap_start} to {overlap_end}")
                
                country_tenor_result['status'] = 'Has Overlap'
                country_tenor_result['overlap_analysis'] = {
                    'common_count': overlap_count,
                    'common_range': [overlap_start, overlap_end],
                    'yield_coverage_pct': overlap_pct_yield,
                    'feature_coverage_pct': overlap_pct_features
                }
                
                # Check if there's sufficient overlap for modeling
                if overlap_count < 30:  # Arbitrary threshold for minimum data points
                    print(f"    WARNING: Overlap may be too small for reliable modeling ({overlap_count} points)")
                    country_tenor_result['issue'] = f"Limited overlap ({overlap_count} points)"
            
            overlap_results[country][tenor_name] = country_tenor_result
    
    # Generate summary statistics
    overlap_summary = {
        'total_combinations': 0,
        'successful': 0,
        'failed': 0,
        'by_status': {},
        'by_country': {},
        'by_tenor': {}
    }
    
    for country in overlap_results:
        overlap_summary['by_country'][country] = {'total': 0, 'successful': 0}
        
        for tenor in overlap_results[country]:
            overlap_summary['total_combinations'] += 1
            overlap_summary['by_country'][country]['total'] += 1
            
            # Initialize tenor counters if needed
            if tenor not in overlap_summary['by_tenor']:
                overlap_summary['by_tenor'][tenor] = {'total': 0, 'successful': 0}
            overlap_summary['by_tenor'][tenor]['total'] += 1
            
            # Count by status
            status = overlap_results[country][tenor]['status']
            if status not in overlap_summary['by_status']:
                overlap_summary['by_status'][status] = 0
            overlap_summary['by_status'][status] += 1
            
            # Check if successful (has overlap)
            if status == 'Has Overlap':
                overlap_summary['successful'] += 1
                overlap_summary['by_country'][country]['successful'] += 1
                overlap_summary['by_tenor'][tenor]['successful'] += 1
            else:
                overlap_summary['failed'] += 1
    
    print("\n=== DATA OVERLAP SUMMARY ===")
    print(f"Total country-tenor combinations: {overlap_summary['total_combinations']}")
    print(f"Successful combinations (has overlap): {overlap_summary['successful']} ({overlap_summary['successful']/overlap_summary['total_combinations']*100:.2f}%)")
    print(f"Failed combinations: {overlap_summary['failed']} ({overlap_summary['failed']/overlap_summary['total_combinations']*100:.2f}%)")
    
    print("\nStatus breakdown:")
    for status, count in overlap_summary['by_status'].items():
        print(f"  {status}: {count} ({count/overlap_summary['total_combinations']*100:.2f}%)")
    
    print("\nBy country:")
    for country, stats in overlap_summary['by_country'].items():
        pct = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {country}: {stats['successful']}/{stats['total']} successful ({pct:.2f}%)")
    
    print("\nBy tenor:")
    for tenor, stats in overlap_summary['by_tenor'].items():
        pct = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {tenor}: {stats['successful']}/{stats['total']} successful ({pct:.2f}%)")
    
    # Save detailed results to CSV
    results_df = []
    for country in overlap_results:
        for tenor in overlap_results[country]:
            result = overlap_results[country][tenor]
            row = {
                'Country': country,
                'Tenor': tenor,
                'Status': result['status'],
                'Issue': result.get('issue', 'None')
            }
            
            # Add yield data details
            if 'yield_data' in result:
                row.update({
                    'Yield_Start': result['yield_data'].get('start_date'),
                    'Yield_End': result['yield_data'].get('end_date'),
                    'Yield_Count': result['yield_data'].get('count')
                })
            
            # Add overlap analysis
            if 'overlap_analysis' in result and result['overlap_analysis']:
                row.update({
                    'Overlap_Count': result['overlap_analysis'].get('common_count', 0),
                    'Overlap_Start': result['overlap_analysis'].get('common_range', ['N/A', 'N/A'])[0] if result['status'] == 'Has Overlap' else 'N/A',
                    'Overlap_End': result['overlap_analysis'].get('common_range', ['N/A', 'N/A'])[1] if result['status'] == 'Has Overlap' else 'N/A',
                    'Yield_Coverage': result['overlap_analysis'].get('yield_coverage_pct', 0) if result['status'] == 'Has Overlap' else 0,
                    'Feature_Coverage': result['overlap_analysis'].get('feature_coverage_pct', 0) if result['status'] == 'Has Overlap' else 0
                })
            
            results_df.append(row)
    
    # Convert to DataFrame and save
    if results_df:
        results_summary = pd.DataFrame(results_df)
        results_summary.to_csv('data_overlap_analysis.csv', index=False)
        print("\nDetailed results saved to 'data_overlap_analysis.csv'")
    
    print("=== END OF DATA OVERLAP ANALYSIS ===\n")
    
    # Create visualization of overlap analysis
    try:
        plt.figure(figsize=(15, 8))
        
        # Count combinations by status
        status_counts = {}
        for country in overlap_results:
            for tenor in overlap_results[country]:
                status = overlap_results[country][tenor]['status']
                status_counts[status] = status_counts.get(status, 0) + 1
        
        # Create bar chart
        statuses = list(status_counts.keys())
        counts = [status_counts[s] for s in statuses]
        
        colors = {
            'Has Overlap': 'green',
            'No Overlap': 'red',
            'Missing Yield Data': 'gray',
            'Empty Yield Data': 'lightgray',
            'Missing Feature Data': 'orange',
            'No Clean Feature Data': 'yellow'
        }
        
        bar_colors = [colors.get(s, 'blue') for s in statuses]
        
        plt.bar(statuses, counts, color=bar_colors)
        plt.title('Data Overlap Analysis Results', fontsize=16)
        plt.xlabel('Status', fontsize=14)
        plt.ylabel('Number of Country-Tenor Combinations', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('data_overlap_analysis.png')
        plt.close()
        
        print("Visualization saved to 'data_overlap_analysis.png'")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    return overlap_results

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

def create_data_availability_summary(country_list, country_code_mapping, yield_list, yield_names, pol_rat, cpi_inf, act_track, risk_rating):
    """
    Create a comprehensive summary of data availability across all data sources,
    countries, and tenors to identify gaps and coverage.
    
    Parameters:
        country_list: List of countries
        country_code_mapping: Mapping from country names to codes
        yield_list: List of yield DataFrames
        yield_names: List of yield tenor names
        pol_rat, cpi_inf, act_track, risk_rating: Feature DataFrames
        
    Returns:
        dict: Comprehensive data availability statistics
    """
    print("\n=== DATA AVAILABILITY SUMMARY ===")
    
    # Define data sources
    data_sources = {
        'policy_rates': pol_rat,
        'inflation': cpi_inf,
        'activity': act_track,
        'risk_rating': risk_rating
    }
    
    # Initialize results dictionary
    availability = {
        'by_country': {},
        'by_tenor': {},
        'by_source': {},
        'by_country_tenor': {},
        'global_stats': {
            'start_date': None,
            'end_date': None,
            'total_observations': 0,
            'missing_percentage': {}
        }
    }
    
    # Define a function to calculate date range statistics
    def get_date_stats(df, country_code=None):
        if df is None or df.empty:
            return {
                'start_date': None,
                'end_date': None,
                'count': 0,
                'nan_pct': 100.0
            }
        
        # If country_code is provided, filter columns
        if country_code:
            country_cols = [col for col in df.columns if col.endswith(f"_{country_code}")]
            if country_cols:
                df_subset = df[country_cols]
            else:
                return {
                    'start_date': None,
                    'end_date': None,
                    'count': 0,
                    'nan_pct': 100.0
                }
        else:
            df_subset = df
        
        # Get date range
        if len(df_subset) > 0:
            start_date = df_subset.index.min()
            end_date = df_subset.index.max()
            count = len(df_subset)
            nan_pct = df_subset.isna().mean().mean() * 100
            
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'count': count,
                'nan_pct': nan_pct
            }
        else:
            return {
                'start_date': None,
                'end_date': None,
                'count': 0,
                'nan_pct': 100.0
            }
    
    # 1. Global date range for all data
    all_dfs = list(data_sources.values()) + yield_list
    all_dfs = [df for df in all_dfs if df is not None and not df.empty]
    
    if all_dfs:
        all_start_dates = [df.index.min() for df in all_dfs if len(df) > 0]
        all_end_dates = [df.index.max() for df in all_dfs if len(df) > 0]
        
        if all_start_dates and all_end_dates:
            global_start = min(all_start_dates)
            global_end = max(all_end_dates)
            
            availability['global_stats']['start_date'] = global_start.strftime('%Y-%m-%d')
            availability['global_stats']['end_date'] = global_end.strftime('%Y-%m-%d')
            
            print(f"Global date range: {global_start.strftime('%Y-%m-%d')} to {global_end.strftime('%Y-%m-%d')}")
    
    # 2. Data availability by source
    print("\nData availability by source:")
    
    for source_name, source_df in data_sources.items():
        stats = get_date_stats(source_df)
        availability['by_source'][source_name] = stats
        
        print(f"  {source_name}:")
        print(f"    Date range: {stats['start_date'] or 'N/A'} to {stats['end_date'] or 'N/A'}")
        print(f"    Observations: {stats['count']}")
        print(f"    Missing data: {stats['nan_pct']:.2f}%")
    
    # 3. Data availability by tenor
    print("\nData availability by yield tenor:")
    
    for tenor_name, tenor_df in zip(yield_names, yield_list):
        stats = get_date_stats(tenor_df)
        availability['by_tenor'][tenor_name] = stats
        
        print(f"  {tenor_name}:")
        print(f"    Date range: {stats['start_date'] or 'N/A'} to {stats['end_date'] or 'N/A'}")
        print(f"    Observations: {stats['count']}")
        print(f"    Missing data: {stats['nan_pct']:.2f}%")
    
    # 4. Data availability by country
    print("\nData availability by country:")
    
    for country in country_list:
        if country not in country_code_mapping:
            continue
            
        country_code = country_code_mapping[country]
        availability['by_country'][country] = {
            'sources': {},
            'tenors': {}
        }
        
        print(f"  {country} (code: {country_code}):")
        
        # Check sources
        print(f"    Data sources:")
        for source_name, source_df in data_sources.items():
            stats = get_date_stats(source_df, country_code)
            availability['by_country'][country]['sources'][source_name] = stats
            
            date_range = f"{stats['start_date'] or 'N/A'} to {stats['end_date'] or 'N/A'}"
            print(f"      {source_name}: {date_range} ({stats['count']} obs, {stats['nan_pct']:.2f}% NaN)")
        
        # Check tenors
        print(f"    Yield tenors:")
        for tenor_name, tenor_df in zip(yield_names, yield_list):
            yield_col = f"{tenor_name}_{country_code}"
            
            if yield_col in tenor_df.columns:
                yield_data = tenor_df[yield_col].dropna()
                
                if not yield_data.empty:
                    start_date = yield_data.index.min().strftime('%Y-%m-%d')
                    end_date = yield_data.index.max().strftime('%Y-%m-%d')
                    count = len(yield_data)
                    
                    stats = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'count': count,
                        'nan_pct': 0.0  # We already dropped NaN values
                    }
                else:
                    stats = {
                        'start_date': None,
                        'end_date': None,
                        'count': 0,
                        'nan_pct': 100.0
                    }
            else:
                stats = {
                    'start_date': None,
                    'end_date': None,
                    'count': 0,
                    'nan_pct': 100.0
                }
            
            availability['by_country'][country]['tenors'][tenor_name] = stats
            
            date_range = f"{stats['start_date'] or 'N/A'} to {stats['end_date'] or 'N/A'}"
            print(f"      {tenor_name}: {date_range} ({stats['count']} obs)")
    
    # 5. Detailed data availability by country and tenor
    for country in country_list:
        if country not in country_code_mapping:
            continue
            
        country_code = country_code_mapping[country]
        
        for tenor_name in yield_names:
            # Define which features we need for this tenor
            if tenor_name == 'yld_2yr':
                required_sources = ['policy_rates', 'inflation', 'activity']
            elif tenor_name == 'yld_5yr':
                required_sources = ['policy_rates', 'inflation', 'risk_rating']
            else:  # 10yr, 30yr
                required_sources = ['policy_rates', 'inflation', 'activity', 'risk_rating']
            
            yield_col = f"{tenor_name}_{country_code}"
            
            # Check if yield data exists
            yield_exists = False
            yield_count = 0
            yield_date_range = (None, None)
            
            for tenor_df in yield_list:
                if yield_col in tenor_df.columns:
                    yield_data = tenor_df[yield_col].dropna()
                    if not yield_data.empty:
                        yield_exists = True
                        yield_count = len(yield_data)
                        yield_date_range = (
                            yield_data.index.min().strftime('%Y-%m-%d'),
                            yield_data.index.max().strftime('%Y-%m-%d')
                        )
                    break
            
            # Check each required source
            sources_available = {}
            for source_name in required_sources:
                source_df = data_sources[source_name]
                
                if source_df is None or source_df.empty:
                    sources_available[source_name] = False
                    continue
                
                country_cols = [col for col in source_df.columns if col.endswith(f"_{country_code}")]
                
                if not country_cols:
                    sources_available[source_name] = False
                    continue
                
                source_data = source_df[country_cols]
                if source_data.empty:
                    sources_available[source_name] = False
                else:
                    sources_available[source_name] = True
            
            # Determine if we have all required data
            all_sources_available = all(sources_available.values())
            
            # Store results
            key = f"{country}_{tenor_name}"
            availability['by_country_tenor'][key] = {
                'yield_exists': yield_exists,
                'yield_count': yield_count,
                'yield_date_range': yield_date_range,
                'sources_available': sources_available,
                'all_sources_available': all_sources_available
            }
    
    # 6. Generate summary statistics
    total_combinations = len(country_list) * len(yield_names)
    complete_combinations = sum(1 for data in availability['by_country_tenor'].values() 
                              if data['yield_exists'] and data['all_sources_available'])
    
    availability['global_stats']['total_combinations'] = total_combinations
    availability['global_stats']['complete_combinations'] = complete_combinations
    availability['global_stats']['complete_percentage'] = (complete_combinations / total_combinations * 100) if total_combinations > 0 else 0
    
    print(f"\nSummary Statistics:")
    print(f"  Total country-tenor combinations: {total_combinations}")
    print(f"  Complete combinations (yield + all features): {complete_combinations} ({availability['global_stats']['complete_percentage']:.2f}%)")
    
    # 7. Create visualizations
    try:
        # Heatmap of data availability by country and tenor
        data = []
        
        for country in country_list:
            if country not in country_code_mapping:
                continue
                
            row = {'Country': country}
            
            for tenor_name in yield_names:
                key = f"{country}_{tenor_name}"
                if key in availability['by_country_tenor']:
                    info = availability['by_country_tenor'][key]
                    
                    if info['yield_exists'] and info['all_sources_available']:
                        row[tenor_name] = 2  # Complete data
                    elif info['yield_exists']:
                        row[tenor_name] = 1  # Only yield data
                    else:
                        row[tenor_name] = 0  # No data
                else:
                    row[tenor_name] = 0  # No data
            
            data.append(row)
        
        if data:
            availability_df = pd.DataFrame(data)
            availability_df.set_index('Country', inplace=True)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(availability_df, cmap=['white', 'yellow', 'green'], 
                      annot=True, fmt='d', cbar_kws={'label': 'Data Availability'})
            
            plt.title('Data Availability by Country and Tenor', fontsize=16)
            plt.tight_layout()
            plt.savefig('data_availability_heatmap.png')
            plt.close()
            
            print("\nData availability heatmap saved to 'data_availability_heatmap.png'")
            
            # Save to CSV
            availability_df.to_csv('data_availability_summary.csv')
            print("Data availability summary saved to 'data_availability_summary.csv'")
        
        # Bar chart of data completeness by country
        country_completeness = []
        
        for country in country_list:
            if country not in country_code_mapping:
                continue
                
            complete_count = 0
            partial_count = 0
            
            for tenor_name in yield_names:
                key = f"{country}_{tenor_name}"
                if key in availability['by_country_tenor']:
                    info = availability['by_country_tenor'][key]
                    
                    if info['yield_exists'] and info['all_sources_available']:
                        complete_count += 1
                    elif info['yield_exists']:
                        partial_count += 1
            
            country_completeness.append({
                'Country': country,
                'Complete': complete_count,
                'Partial': partial_count,
                'Missing': len(yield_names) - complete_count - partial_count
            })
        
        if country_completeness:
            completeness_df = pd.DataFrame(country_completeness)
            
            plt.figure(figsize=(12, 6))
            completeness_df.set_index('Country').plot(kind='bar', stacked=True, 
                                                    color=['green', 'yellow', 'red'])
            
            plt.title('Data Completeness by Country', fontsize=16)
            plt.xlabel('Country', fontsize=14)
            plt.ylabel('Number of Tenors', fontsize=14)
            plt.legend(['Complete', 'Partial', 'Missing'])
            plt.tight_layout()
            plt.savefig('data_completeness_by_country.png')
            plt.close()
            
            print("Data completeness chart saved to 'data_completeness_by_country.png'")
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print("=== END OF DATA AVAILABILITY SUMMARY ===\n")
    
    return availability

def visualize_a_model_prediction(country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, act_track, risk_rating):
    """
    Create a single visualization of what a model predicts for a specific country and tenor,
    based on available feature data, even if there's not enough data to train a proper model.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        country_code_mapping: dict - Mapping from country names to country codes
        tenor_data: DataFrame - Yield data for the specified tenor
        pol_rat, cpi_inf, act_track, risk_rating: Feature DataFrames
    """
    print(f"\n--- VISUALIZING MODEL PREDICTION FOR {country} - {tenor_name} ---")
    
    if country not in country_code_mapping:
        print(f"Error: Country {country} not found in country_code_mapping")
        return None
    
    country_code = country_code_mapping[country]
    
    # Get target data
    yield_col = f"{tenor_name}_{country_code}"
    if yield_col not in tenor_data.columns:
        print(f"Error: Yield column {yield_col} not found in tenor data")
        return None
    
    yield_data = tenor_data[yield_col].dropna()
    if yield_data.empty:
        print(f"Error: No yield data available for {country} - {tenor_name}")
        return None
    
    print(f"Target yield data: {yield_data.index.min().strftime('%Y-%m-%d')} to {yield_data.index.max().strftime('%Y-%m-%d')} ({len(yield_data)} points)")
    
    # Prepare data using our enhanced prepare_data function
    x, y, feature_details = prepare_data(
        country=country,
        tenor_name=tenor_name,
        country_code_mapping=country_code_mapping,
        tenor=tenor_data,
        pol_rat=pol_rat,
        cpi_inf=cpi_inf,
        act_track=act_track,
        risk_rating=risk_rating
    )
    
    if x.empty or len(y) == 0:
        print(f"Error: Insufficient data to create model")
        
        # Even though we can't train a model, let's still visualize what data we have
        plt.figure(figsize=(12, 6))
        plt.plot(yield_data.index, yield_data, 'b-', label=f'Actual {tenor_name}', linewidth=2)
        
        plt.title(f"{country} - {tenor_name}: Available Data (No Model Possible)", fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Yield (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Format x-axis to show years
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
        plt.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{country}_{tenor_name}_available_data.png")
        plt.close()
        
        print(f"Available data visualization saved to {country}_{tenor_name}_available_data.png")
        return None
    
    # We have overlapping data, so we can create a model
    print(f"Creating model with {len(x)} data points")
    
    # Split data into train (90%) and validation (10%) sets
    train_size = int(len(x) * 0.9)
    x_train = x.iloc[:train_size]
    y_train = y.iloc[:train_size]
    x_val = x.iloc[train_size:]
    y_val = y.iloc[train_size:]
    
    # Scale the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val) if not x_val.empty else None
    
    # Train a simple MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
    mlp.fit(x_train_scaled, y_train)
    
    # Make predictions on training data
    y_train_pred = mlp.predict(x_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Model metrics:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Training R²: {train_r2:.4f}")
    
    # If we have validation data, evaluate on it
    if not x_val.empty and len(y_val) > 0:
        y_val_pred = mlp.predict(x_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        print(f"  Validation RMSE: {val_rmse:.4f}")
        print(f"  Validation R²: {val_r2:.4f}")
    
    # Create plot of actual vs. predicted values
    plt.figure(figsize=(12, 6))
    
    # Plot actual data
    plt.plot(y.index, y, 'b-', label='Actual', linewidth=2)
    
    # Plot training predictions
    train_indices = y_train.index
    plt.plot(train_indices, y_train_pred, 'r--', label='Model Predictions (Training)', linewidth=2)
    
    # Plot validation predictions if available
    if not x_val.empty and len(y_val) > 0:
        val_indices = y_val.index
        plt.plot(val_indices, y_val_pred, 'g--', label='Model Predictions (Validation)', linewidth=2)
    
    # Add annotations for model metrics
    metrics_text = f"Training RMSE: {train_rmse:.4f}\nTraining R²: {train_r2:.4f}"
    if not x_val.empty and len(y_val) > 0:
        metrics_text += f"\nValidation RMSE: {val_rmse:.4f}\nValidation R²: {val_r2:.4f}"
    
    plt.annotate(metrics_text, 
               xy=(0.05, 0.95), xycoords='axes fraction',
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
               fontsize=10)
    
    # Calculate feature importance via permutation
    if len(x) >= 30:  # Only do this if we have enough data
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(mlp, x_train_scaled, y_train, n_repeats=10, random_state=42)
            importance = result.importances_mean
            
            # Create a sorted list of feature importances
            feature_importances = [(x.columns[i], importance[i]) for i in range(len(importance))]
            feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Add top 5 features to plot
            top_features = feature_importances[:min(5, len(feature_importances))]
            feature_text = "Top Features:\n" + "\n".join([f"{name}: {imp:.4f}" for name, imp in top_features])
            
            plt.annotate(feature_text, 
                       xy=(0.05, 0.70), xycoords='axes fraction',
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                       fontsize=9)
        except Exception as e:
            print(f"Could not calculate feature importance: {e}")
    
    plt.title(f"{country} - {tenor_name}: Model Prediction", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Yield (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Format x-axis to show years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{country}_{tenor_name}_model_prediction.png")
    plt.close()
    
    print(f"Model prediction visualization saved to {country}_{tenor_name}_model_prediction.png")
    
    # Create a complete visualization with feature data if there are at least 3 features
    if x.shape[1] >= 3:
        try:
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(x.shape[1] + 1, 1, figsize=(12, 3 * (x.shape[1] + 1)), sharex=True)
            
            # Plot yield data in the first subplot
            axes[0].plot(y.index, y, 'b-', label='Actual Yield', linewidth=2)
            axes[0].plot(train_indices, y_train_pred, 'r--', label='Model Predictions', linewidth=2)
            if not x_val.empty and len(y_val) > 0:
                axes[0].plot(val_indices, y_val_pred, 'g--', label='Validation Predictions', linewidth=2)
            
            axes[0].set_title(f"{country} - {tenor_name}: Yield and Features", fontsize=14)
            axes[0].set_ylabel('Yield (%)', fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].grid(True)
            
            # Plot each feature in its own subplot
            for i, column in enumerate(x.columns):
                ax = axes[i+1]
                ax.plot(x.index, x[column], 'k-', linewidth=1.5)
                ax.set_ylabel(column, fontsize=10)
                ax.grid(True)
                
                # Add correlation value
                correlation = x[column].corr(y)
                ax.annotate(f"Corr: {correlation:.4f}", 
                           xy=(0.05, 0.85), xycoords='axes fraction',
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                           fontsize=9)
            
            # Format x-axis on the bottom subplot
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
            axes[-1].tick_params(axis='x', rotation=45)
            axes[-1].set_xlabel('Date', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{country}_{tenor_name}_with_features.png")
            plt.close(fig)
            
            print(f"Feature visualization saved to {country}_{tenor_name}_with_features.png")
        except Exception as e:
            print(f"Error creating feature visualization: {e}")
    
    print("--- END OF MODEL PREDICTION VISUALIZATION ---\n")
    
    return mlp, scaler


def main():
    """
    Main function to execute the yield curve modeling and forecasting pipeline.
    Enhanced with comprehensive diagnostics, visualizations, and improved error handling.
    """
    # Define list of countries and their codes
    country_list = [
        'United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Poland',
        'Hungary', 'Czechia', 'South Africa', 'Canada', 'Australia', 'South Korea'
    ]
    
    country_code_mapping = {
        'United States': 'us',
        'United Kingdom': 'gb',
        'France': 'fr',
        'Germany': 'de',
        'Italy': 'it',
        'Poland': 'pl',
        'Hungary': 'hu',
        'Czechia': 'cz',
        'South Africa': 'za',
        'Canada': 'ca',
        'Australia': 'au',
        'South Korea': 'kr'
    }
    
    forecast_horizons = [24, 60, 120, 360]
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("yield_curve_model.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting yield curve modeling and forecasting pipeline")
    
    # Create directories for outputs if they don't exist
    os.makedirs("model_outputs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("data_analysis", exist_ok=True)
    
    # Initialize results tracking
    results_summary = {
        'data_overlap': None,
        'data_availability': None,
        'model_results': {},
        'success_count': 0,
        'failure_count': 0
    }
    
    try:
        # Fetch bond yield data for different tenors
        logger.info("Fetching bond yield data...")
        
        # 2-year bond yields
        bondyield_2yr_tickers = [
            "USGG2YR Index", "GTGBP2YR Corp", "GTFRF2YR Corp", "GTDEM2YR Corp",
            "GTITL2YR Corp", "GTPLN2YR Corp", "GTHUF3YR Corp", "GTCZK2YR Corp",
            "GTZAR2YR Corp", "GTCAD2YR Corp", "GTAUD2YR Corp", "GTKRW2YR Corp"
        ]
        
        dt_from = dt.date(1990, 1, 1)
        dt_to = dt.date.today()
        
        yld_2yr = get_bloomberg_date(bondyield_2yr_tickers, dt_from, dt_to, periodicity='DAILY')
        
        yld_2yr = yld_2yr.rename(columns={
            "USGG2YR Index": "yld_2yr_us",
            "GTGBP2YR Corp": "yld_2yr_gb",
            "GTFRF2YR Corp": "yld_2yr_fr",
            "GTDEM2YR Corp": "yld_2yr_de",
            "GTITL2YR Corp": "yld_2yr_it",
            "GTPLN2YR Corp": "yld_2yr_pl",
            "GTHUF3YR Corp": "yld_2yr_hu",  # Note: Using 3yr for Hungary as specified
            "GTCZK2YR Corp": "yld_2yr_cz",
            "GTZAR2YR Corp": "yld_2yr_za",
            "GTCAD2YR Corp": "yld_2yr_ca",
            "GTAUD2YR Corp": "yld_2yr_au",
            "GTKRW2YR Corp": "yld_2yr_kr"
        })
        
        yld_2yr_ann = yld_2yr.resample('M').mean()
        logger.info("2-year yield data retrieved and processed")
        
        # 5-year bond yields
        bondyield_5yr_tickers = [
            "USGG5YR Index", "GTGBP5YR Corp", "GTFRF5YR Corp", "GTDEM5YR Corp",
            "GTITL5YR Corp", "GTPLN5YR Corp", "GTHUF5YR Corp", "GTCZK5YR Corp",
            "GTZAR5YR Corp", "GTCAD5YR Corp", "GTAUD5YR Corp", "GTKRW5YR Corp"
        ]
        
        yld_5yr = get_bloomberg_date(bondyield_5yr_tickers, dt_from, dt_to, periodicity='DAILY')
        
        yld_5yr = yld_5yr.rename(columns={
            "USGG5YR Index": "yld_5yr_us",
            "GTGBP5YR Corp": "yld_5yr_gb",
            "GTFRF5YR Corp": "yld_5yr_fr",
            "GTDEM5YR Corp": "yld_5yr_de",
            "GTITL5YR Corp": "yld_5yr_it",
            "GTPLN5YR Corp": "yld_5yr_pl",
            "GTHUF5YR Corp": "yld_5yr_hu",
            "GTCZK5YR Corp": "yld_5yr_cz",
            "GTZAR5YR Corp": "yld_5yr_za",
            "GTCAD5YR Corp": "yld_5yr_ca",
            "GTAUD5YR Corp": "yld_5yr_au",
            "GTKRW5YR Corp": "yld_5yr_kr"
        })
        
        yld_5yr_ann = yld_5yr.resample('M').mean()
        logger.info("5-year yield data retrieved and processed")
        
        # 10-year bond yields
        bondyield_10yr_tickers = [
            "USGG10YR Index", "GTGBP10YR Corp", "GTFRF10YR Corp", "GTDEM10YR Corp",
            "GTITL10YR Corp", "GTPLN10YR Corp", "GTHUF10YR Corp", "GTCZK10YR Corp",
            "GTZAR10YR Corp", "GTCAD10YR Corp", "GTAUD10YR Corp", "GTKRW10YR Corp"
        ]
        
        yld_10yr = get_bloomberg_date(bondyield_10yr_tickers, dt_from, dt_to, periodicity='DAILY')
        
        yld_10yr = yld_10yr.rename(columns={
            "USGG10YR Index": "yld_10yr_us",
            "GTGBP10YR Corp": "yld_10yr_gb",
            "GTFRF10YR Corp": "yld_10yr_fr",
            "GTDEM10YR Corp": "yld_10yr_de",
            "GTITL10YR Corp": "yld_10yr_it",
            "GTPLN10YR Corp": "yld_10yr_pl",
            "GTHUF10YR Corp": "yld_10yr_hu",
            "GTCZK10YR Corp": "yld_10yr_cz",
            "GTZAR10YR Corp": "yld_10yr_za",
            "GTCAD10YR Corp": "yld_10yr_ca",
            "GTAUD10YR Corp": "yld_10yr_au",
            "GTKRW10YR Corp": "yld_10yr_kr"
        })
        
        yld_10yr_ann = yld_10yr.resample('M').mean()
        logger.info("10-year yield data retrieved and processed")
        
        # 30-year bond yields (note: some countries don't have 30yr bonds)
        bondyield_30yr_tickers = [
            "USGG30YR Index", "GTGBP30YR Corp", "GTFRF30YR Corp", "GTDEM30YR Corp",
            "GTITL30YR Corp", "GTCAD30YR Corp", "GTAUD30YR Corp", "GTKRW30YR Corp"
        ]
        
        yld_30yr = get_bloomberg_date(bondyield_30yr_tickers, dt_from, dt_to, periodicity='DAILY')
        
        yld_30yr = yld_30yr.rename(columns={
            "USGG30YR Index": "yld_30yr_us",
            "GTGBP30YR Corp": "yld_30yr_gb",
            "GTFRF30YR Corp": "yld_30yr_fr",
            "GTDEM30YR Corp": "yld_30yr_de",
            "GTITL30YR Corp": "yld_30yr_it",
            "GTCAD30YR Corp": "yld_30yr_ca",
            "GTAUD30YR Corp": "yld_30yr_au",
            "GTKRW30YR Corp": "yld_30yr_kr"
        })
        
        yld_30yr_ann = yld_30yr.resample('M').mean()
        logger.info("30-year yield data retrieved and processed")
        
        # Fetch policy rate data
        logger.info("Fetching policy rate data...")
        policy_rate_tickers = [
            "BISPDHUS Index", "BISPDHUK Index", "BISPDHEA Index", 
            "EURR002W Index", "EUORDEPO Index", "BISPDHPO Index", 
            "BISPDHHU Index", "BISPDHCZ Index", "BISPDHSA Index", 
            "BISPDHCA Index", "BISPDHAU Index", "BISPDHSK Index"
        ]
        
        pol_rat = get_bloomberg_date(policy_rate_tickers, dt_from, dt_to, periodicity='DAILY')
        
        # Use distinct column names for eurozone countries
        pol_rat = pol_rat.rename(columns={
            "BISPDHUS Index": "pol_rat_us",
            "BISPDHUK Index": "pol_rat_gb",
            "BISPDHEA Index": "pol_rat_fr",  # First eurozone country
            "EURR002W Index": "pol_rat_de",  # Second eurozone country
            "EUORDEPO Index": "pol_rat_it",  # Third eurozone country
            "BISPDHPO Index": "pol_rat_pl",
            "BISPDHHU Index": "pol_rat_hu",
            "BISPDHCZ Index": "pol_rat_cz",
            "BISPDHSA Index": "pol_rat_za",
            "BISPDHCA Index": "pol_rat_ca",
            "BISPDHAU Index": "pol_rat_au",
            "BISPDHSK Index": "pol_rat_kr"
        })
        
        pol_rat = pol_rat.resample('M').mean()
        logger.info("Policy rate data retrieved and processed")
        
        # Fetch economic activity tracker data
        logger.info("Fetching economic activity tracker data...")
        activity_tickers = [
            "GSUSCAI Index", "GSGBCAI Index", "GSFRCAI Index", "GSDECAI Index",
            "GSITCAI Index", "GSPLCAI Index", "GSHUCAI Index", "GSCZCAI Index",
            "GSZACAI Index", "GSCACAI Index", "GSAUCAI Index", "GSKRCAI Index"
        ]
        
        act_track = get_bloomberg_date(activity_tickers, dt_from, dt_to, periodicity='MONTHLY')
        
        act_track = act_track.rename(columns={
            "GSUSCAI Index": "act_track_us",
            "GSGBCAI Index": "act_track_gb",
            "GSFRCAI Index": "act_track_fr",
            "GSDECAI Index": "act_track_de",
            "GSITCAI Index": "act_track_it",
            "GSPLCAI Index": "act_track_pl",
            "GSHUCAI Index": "act_track_hu",
            "GSCZCAI Index": "act_track_cz",
            "GSZACAI Index": "act_track_za",
            "GSCACAI Index": "act_track_ca",
            "GSAUCAI Index": "act_track_au",
            "GSKRCAI Index": "act_track_kr"
        })

        if not act_track.empty:
            act_track.index = act_track.index.to_period("M").to_timestamp("M")
            act_track = act_track.resample('M').first().ffill()
            act_track.index = pd.DatetimeIndex(act_track.index.strftime('%Y-%m-%d'))
            
        print(act_track)
        
        logger.info("Economic activity tracker data retrieved and processed")
        
        # Fetch inflation data
        logger.info("Fetching inflation data...")

            # GDP data for various countries
        # cpi_inf_tickers = ["ih:mb:com:us_cpi_exp_consen:us_cpi_exp_consen",
        #                "ih:mb:com:gb_cpi_exp_consen:gb_cpi_exp_consen",
        #                "ih:mb:com:fr_cpi_exp_consen:fr_cpi_exp_consen",
        #                "ih:mb:com:de_cpi_exp_consen:de_cpi_exp_consen",
        #                "ih:mb:com:it_cpi_exp_consen:it_cpi_exp_consen",
        #                "ih:mb:com:pl_cpi_exp_consen:pl_cpi_exp_consen",
        #                "ih:mb:com:hu_cpi_exp_consen:hu_cpi_exp_consen",
        #                "ih:mb:com:cz_cpi_exp_consen:cz_cpi_exp_consen",
        #                "ih:mb:com:za_cpi_exp_consen:za_cpi_exp_consen",
        #                "ih:mb:com:ca_cpi_exp_consen:ca_cpi_exp_consen",
        #                "ih:mb:com:au_cpi_exp_consen:au_cpi_exp_consen",
        #                "ih:mb:com:kr_cpi_exp_consen:kr_cpi_exp_consen"]

        # # Fetch GDP data
        # cpi_inf = mb.FetchSeries(gdp_tickers)

        # # Create mapping between original column names and more readable names
        # new_column_names = {"ih:mb:com:us_cpi_exp_consen:us_cpi_exp_consen" : "gdp_us",
        #                "ih:mb:com:gb_cpi_exp_consen:gb_cpi_exp_consen" : "gdp_gb",
        #                "ih:mb:com:fr_cpi_exp_consen:fr_cpi_exp_consen" : "gdp_fr",
        #                "ih:mb:com:de_cpi_exp_consen:de_cpi_exp_consen" : "gdp_de",
        #                "ih:mb:com:it_cpi_exp_consen:it_cpi_exp_consen" : "gdp_it",
        #                "ih:mb:com:pl_cpi_exp_consen:pl_cpi_exp_consen" : "gdp_pl",
        #                "ih:mb:com:hu_cpi_exp_consen:hu_cpi_exp_consen" : "gdp_hu",
        #                "ih:mb:com:cz_cpi_exp_consen:cz_cpi_exp_consen" : "gdp_cz",
        #                "ih:mb:com:za_cpi_exp_consen:za_cpi_exp_consen" : "gdp_za",
        #                "ih:mb:com:ca_cpi_exp_consen:ca_cpi_exp_consen" : "gdp_ca",
        #                "ih:mb:com:au_cpi_exp_consen:au_cpi_exp_consen" : "gdp_au",
        #                "ih:mb:com:kr_cpi_exp_consen:kr_cpi_exp_consen" : "gdp_kr"
        # }

        # cpi_inf = cpi_inf.rename(columns=new_column_names)
        # cpi_inf = cpi_inf.resample('M').mean()
        # cpi_inf = cpi_inf.ffill()  # Forward fill missing values
        # cpi_inf = cpi_inf.bfill()

        cpi_inf_tickers = [
            "uscpi", "gbcpi", "frcpi", "decpi", "itcpi", "plcpi",
            "hucpi", "czcpi", "zacpi", "cacpi", "aucpi", "krcpi"
        ]
        
        try:
            cpi_inf = mb.FetchSeries(cpi_inf_tickers)
            
            new_column_names = {
                "uscpi": "cpi_inf_us",
                "gbcpi": "cpi_inf_gb",
                "frcpi": "cpi_inf_fr",
                "decpi": "cpi_inf_de",
                "itcpi": "cpi_inf_it",
                "plcpi": "cpi_inf_pl",
                "hucpi": "cpi_inf_hu",
                "czcpi": "cpi_inf_cz",
                "zacpi": "cpi_inf_za",
                "cacpi": "cpi_inf_ca",
                "aucpi": "cpi_inf_au",
                "krcpi": "cpi_inf_kr"
            }
            
            cpi_inf = cpi_inf.rename(columns=new_column_names)
            cpi_inf = cpi_inf.resample('M').mean()
            cpi_inf = cpi_inf.pct_change(periods=12) * 100  # Convert to year-over-year percentage change
            print(cpi_inf)
            logger.info("Inflation data retrieved and processed")
        except Exception as e:
            logger.error(f"Error fetching inflation data: {e}")
            cpi_inf = pd.DataFrame()

            # GDP data for various countries
        # gdp_tickers = ["ih:mb:com:us_gdp_exp_consen:us_gdp_exp_consen",
        #                "ih:mb:com:gb_gdp_exp_consen:gb_gdp_exp_consen",
        #                "ih:mb:com:fr_gdp_exp_consen:fr_gdp_exp_consen",
        #                "ih:mb:com:de_gdp_exp_consen:de_gdp_exp_consen",
        #                "ih:mb:com:it_gdp_exp_consen:it_gdp_exp_consen",
        #                "ih:mb:com:pl_gdp_exp_consen:pl_gdp_exp_consen",
        #                "ih:mb:com:hu_gdp_exp_consen:hu_gdp_exp_consen",
        #                "ih:mb:com:cz_gdp_exp_consen:cz_gdp_exp_consen",
        #                "ih:mb:com:za_gdp_exp_consen:za_gdp_exp_consen",
        #                "ih:mb:com:ca_gdp_exp_consen:ca_gdp_exp_consen",
        #                "ih:mb:com:au_gdp_exp_consen:au_gdp_exp_consen",
        #                "ih:mb:com:kr_gdp_exp_consen:kr_gdp_exp_consen"]

        # # Fetch GDP data
        # gdp = mb.FetchSeries(gdp_tickers)

        # # Create mapping between original column names and more readable names
        # new_column_names = {"ih:mb:com:us_gdp_exp_consen:us_gdp_exp_consen" : "gdp_us",
        #                "ih:mb:com:gb_gdp_exp_consen:gb_gdp_exp_consen" : "gdp_gb",
        #                "ih:mb:com:fr_gdp_exp_consen:fr_gdp_exp_consen" : "gdp_fr",
        #                "ih:mb:com:de_gdp_exp_consen:de_gdp_exp_consen" : "gdp_de",
        #                "ih:mb:com:it_gdp_exp_consen:it_gdp_exp_consen" : "gdp_it",
        #                "ih:mb:com:pl_gdp_exp_consen:pl_gdp_exp_consen" : "gdp_pl",
        #                "ih:mb:com:hu_gdp_exp_consen:hu_gdp_exp_consen" : "gdp_hu",
        #                "ih:mb:com:cz_gdp_exp_consen:cz_gdp_exp_consen" : "gdp_cz",
        #                "ih:mb:com:za_gdp_exp_consen:za_gdp_exp_consen" : "gdp_za",
        #                "ih:mb:com:ca_gdp_exp_consen:ca_gdp_exp_consen" : "gdp_ca",
        #                "ih:mb:com:au_gdp_exp_consen:au_gdp_exp_consen" : "gdp_au",
        #                "ih:mb:com:kr_gdp_exp_consen:kr_gdp_exp_consen" : "gdp_kr"
        # }

        # gdp = gdp.rename(columns=new_column_names)
        # gdp = gdp.resample('M').mean()
        # gdp = gdp.ffill()  # Forward fill missing values
        # gdp = gdp.bfill()


        gdp_tickers = ["usgdp", "gbgdp", "frgdp", "degdp", "oecd_qna_00011061", "plnaac0197", "hugdp", "czgdp", "zagdp", "cagdp", "augdp", "krgdp"]

        # Fetch GDP data
        gdp = mb.FetchSeries(gdp_tickers)

        # Create mapping between original column names and more readable names
        new_column_names = {"usgdp": "gdp_us",
        "gbgdp": "gdp_gb",
        "frgdp": "gdp_fr",
        "degdp": "gdp_de",
        "oecd_qna_00011061": "gdp_it",
        "plnaac0197": "gdp_pl",
        "hugdp": "gdp_hu",
        "czgdp": "gdp_cz",
        "zagdp": "gdp_za",
        "cagdp": "gdp_ca",
        "augdp": "gdp_au",
        "krgdp": "gdp_kr"
        }

        # Rename columns and resample to monthly frequency
        gdp = gdp.rename(columns=new_column_names)
        gdp = gdp.pct_change(periods=4) * 100  # Convert to year-over-year percentage change
        gdp = gdp.resample('M').mean()
        gdp = gdp.ffill()  # Forward fill missing values
        gdp = gdp.bfill()
        
        # Growth forecast data (long-term)
        growth_forecast_lt_tickers = [
            "ih:mb:com:gdp_lt_us:gdp_lt_us", "ih:mb:com:gdp_lt_gb:gdp_lt_gb", "ih:mb:com:gdp_lt_fr:gdp_lt_fr",
            "ih:mb:com:gdp_lt_de:gdp_lt_de", "ih:mb:com:gdp_lt_it:gdp_lt_it", "ih:mb:com:gdp_lt_pl:gdp_lt_pl", "ih:mb:com:gdp_lt_hu:gdp_lt_hu",
            "ih:mb:com:gdp_lt_za:gdp_lt_za", "ih:mb:com:gdp_lt_ca:gdp_lt_ca", "ih:mb:com:gdp_lt_au:gdp_lt_au", "ih:mb:com:gdp_lt_kr:gdp_lt_kr"
        ]

        # Fetch long-term growth forecast
        growth_forecast_lt = mb.FetchSeries(growth_forecast_lt_tickers)
        growth_forecast_lt.index.name = "Date"

        # Create mapping for growth forecast columns
        new_column_names = {
            "ih:mb:com:gdp_lt_us:gdp_lt_us": "gdp_lt_us",
            "ih:mb:com:gdp_lt_gb:gdp_lt_gb": "gdp_lt_gb",
            "ih:mb:com:gdp_lt_fr:gdp_lt_fr": "gdp_lt_fr",
            "ih:mb:com:gdp_lt_de:gdp_lt_de": "gdp_lt_de",
            "ih:mb:com:gdp_lt_it:gdp_lt_it": "gdp_lt_it",
            "ih:mb:com:gdp_lt_pl:gdp_lt_pl": "gdp_lt_pl",
            "ih:mb:com:gdp_lt_hu:gdp_lt_hu": "gdp_lt_hu",
            "ih:mb:com:gdp_lt_cz:gdp_lt_cz": "gdp_lt_cz",
            "ih:mb:com:gdp_lt_za:gdp_lt_za": "gdp_lt_za",
            "ih:mb:com:gdp_lt_ca:gdp_lt_ca": "gdp_lt_ca",
            "ih:mb:com:gdp_lt_au:gdp_lt_au": "gdp_lt_au",
            "ih:mb:com:gdp_lt_kr:gdp_lt_kr": "gdp_lt_kr"
        }

            # Rename columns
        growth_forecast_lt = growth_forecast_lt.rename(columns=new_column_names)
        growth_forecast_lt = growth_forecast_lt.pct_change(periods=1) * 100  # Convert to percentage change
        growth_forecast_lt = growth_forecast_lt.resample('M')  # Resample to monthly
        growth_forecast_lt = growth_forecast_lt.ffill()  # Forward fill missing values

        # Create a date range from March 1947 to the start of gdp_lt
        start_date = '1947-03-31'
        end_date = growth_forecast_lt.index[0] - pd.DateOffset(months=1)
        backfill_dates = pd.date_range(start=start_date, end=end_date, freq='M')

        # Create a DataFrame with the backfill dates and fill with a default value (e.g., 2.0)
        backfill_data = pd.DataFrame(index=backfill_dates, columns=growth_forecast_lt.columns)
        backfill_data = backfill_data.fillna(2.0)  # Default value for backfill

        # Concatenate the backfill data with the original gdp_lt data
        growth_forecast_lt = growth_forecast_lt.fillna(2.0)
        growth_forecast_lt = pd.concat([backfill_data, growth_forecast_lt])

            # CPI target data (long-term inflation targets)
        cpi_target_tickers = [
            "usrate1950", "gbrate0237", "eurate0022", "eurate0022",  # Same as FR since they share ECB
            "eurate0022",  # Same as IT since they share ECB
            "carate0093", "aurate0097", "plrate0043", "zapric0688",
            "hurate0005", "krrate0161", "czrate0064"]
            

        # Fetch CPI target data
        cpi_target = mb.FetchSeries(cpi_target_tickers)
        cpi_target.index.name = "Date"

        # Create mapping for CPI target columns
        new_column_names = {
            "usrate1950": "cpi_target_us",
            "gbrate0237": "cpi_target_gb",
            "eurate0022_3": "cpi_target_fr",
            "eurate0022_4": "cpi_target_de",
            "eurate0022_5": "cpi_target_it",
            "carate0093": "cpi_target_ca",
            "aurate0097": "cpi_target_au",
            "plrate0043": "cpi_target_pl",
            "zapric0688": "cpi_target_za",
            "hurate0005": "cpi_target_hu",
            "krrate0161": "cpi_target_kr",
            "czrate0064" : "cpi_target_cz"
        }

        # Fix column names for ECB countries (they share same target)
        cpi_target.columns = [f"eurate0022_{i+1}" if col == "eurate0022" and i > 0 else col for i, col in enumerate(cpi_target.columns)]

        # Rename and process CPI target data
        cpi_target = cpi_target.rename(columns=new_column_names)
        cpi_target = cpi_target.resample('M')
        cpi_target = cpi_target.ffill()  # Forward fill missing values
        cpi_target = cpi_target.bfill()

        # Create a date range from March 1947 to the start of gdp_lt
        start_date = '1947-03-31'
        end_date = cpi_target.index[0] - pd.DateOffset(months=1)
        backfill_dates = pd.date_range(start=start_date, end=end_date, freq='M')

        # Create a DataFrame with the backfill dates and fill with a default value (e.g., 2.0)
        backfill_data = pd.DataFrame(index=backfill_dates, columns=cpi_target.columns)
        backfill_data = backfill_data.fillna(2.0)  # Default value for backfill

        # Concatenate the backfill data with the original gdp_lt data
        cpi_target = cpi_target.fillna(2.0)
        cpi_target = pd.concat([backfill_data, cpi_target])
        
        # Fetch credit rating data and calculate consolidated risk rating
        logger.info("Fetching credit rating data...")
        
        # Fetch Moody's, Fitch, and S&P ratings
        m_rating = pd.DataFrame()
        f_rating = pd.DataFrame()
        s_rating = pd.DataFrame()
        
        try:
            # Moody's ratings
            moodys_credit_tickers = [
                "ih:mb:com:m_rating_usa", "ih:mb:com:m_rating_gbr", "ih:mb:com:m_rating_fra",
                "ih:mb:com:m_rating_deu", "ih:mb:com:m_rating_ita", "ih:mb:com:m_rating_pol",
                "ih:mb:com:m_rating_hun", "ih:mb:com:m_rating_cze", "ih:mb:com:m_rating_zaf",
                "ih:mb:com:m_rating_can", "ih:mb:com:m_rating_aus", "ih:mb:com:m_rating_kor"
            ]
            
            m_rating = mb.FetchSeries(moodys_credit_tickers)
            m_rating.index.name = "Date"
            
            new_column_names = {
                "ih:mb:com:m_rating_usa": "m_rating_us",
                "ih:mb:com:m_rating_gbr": "m_rating_gb",
                "ih:mb:com:m_rating_fra": "m_rating_fr",
                "ih:mb:com:m_rating_deu": "m_rating_de",
                "ih:mb:com:m_rating_ita": "m_rating_it",
                "ih:mb:com:m_rating_pol": "m_rating_pl",
                "ih:mb:com:m_rating_hun": "m_rating_hu",
                "ih:mb:com:m_rating_cze": "m_rating_cz",
                "ih:mb:com:m_rating_zaf": "m_rating_za",
                "ih:mb:com:m_rating_can": "m_rating_ca",
                "ih:mb:com:m_rating_aus": "m_rating_au",
                "ih:mb:com:m_rating_kor": "m_rating_kr"
            }
            
            m_rating = m_rating.rename(columns=new_column_names)
            m_rating = m_rating.resample('M').mean()
            
            # Fitch ratings
            fitch_credit_tickers = [
                "ih:mb:com:f_rating_usa", "ih:mb:com:f_rating_gbr", "ih:mb:com:f_rating_fra",
                "ih:mb:com:f_rating_deu", "ih:mb:com:f_rating_ita", "ih:mb:com:f_rating_pol",
                "ih:mb:com:f_rating_hun", "ih:mb:com:f_rating_cze", "ih:mb:com:f_rating_zaf",
                "ih:mb:com:f_rating_can", "ih:mb:com:f_rating_aus", "ih:mb:com:f_rating_kor"
            ]
            
            f_rating = mb.FetchSeries(fitch_credit_tickers)
            f_rating.index.name = "Date"
            
            new_column_names = {
                "ih:mb:com:f_rating_usa": "f_rating_us",
                "ih:mb:com:f_rating_gbr": "f_rating_gb",
                "ih:mb:com:f_rating_fra": "f_rating_fr",
                "ih:mb:com:f_rating_deu": "f_rating_de",
                "ih:mb:com:f_rating_ita": "f_rating_it",
                "ih:mb:com:f_rating_pol": "f_rating_pl",
                "ih:mb:com:f_rating_hun": "f_rating_hu",
                "ih:mb:com:f_rating_cze": "f_rating_cz",
                "ih:mb:com:f_rating_zaf": "f_rating_za",
                "ih:mb:com:f_rating_can": "f_rating_ca",
                "ih:mb:com:f_rating_aus": "f_rating_au",
                "ih:mb:com:f_rating_kor": "f_rating_kr"
            }
            
            f_rating = f_rating.rename(columns=new_column_names)
            f_rating = f_rating.resample('M').mean()
            
            # S&P ratings
            s_credit_tickers = [
                "ih:mb:com:s_rating_usa", "ih:mb:com:s_rating_gbr", "ih:mb:com:s_rating_fra",
                "ih:mb:com:s_rating_deu", "ih:mb:com:s_rating_ita", "ih:mb:com:s_rating_pol",
                "ih:mb:com:s_rating_hun", "ih:mb:com:s_rating_cze", "ih:mb:com:s_rating_zaf",
                "ih:mb:com:s_rating_can", "ih:mb:com:s_rating_aus", "ih:mb:com:s_rating_kor"
            ]
            
            s_rating = mb.FetchSeries(s_credit_tickers)
            s_rating.index.name = "Date"
            
            new_column_names = {
                "ih:mb:com:s_rating_usa": "s_rating_us",
                "ih:mb:com:s_rating_gbr": "s_rating_gb",
                "ih:mb:com:s_rating_fra": "s_rating_fr",
                "ih:mb:com:s_rating_deu": "s_rating_de",
                "ih:mb:com:s_rating_ita": "s_rating_it",
                "ih:mb:com:s_rating_pol": "s_rating_pl",
                "ih:mb:com:s_rating_hun": "s_rating_hu",
                "ih:mb:com:s_rating_cze": "s_rating_cz",
                "ih:mb:com:s_rating_zaf": "s_rating_za",
                "ih:mb:com:s_rating_can": "s_rating_ca",
                "ih:mb:com:s_rating_aus": "s_rating_au",
                "ih:mb:com:s_rating_kor": "s_rating_kr"
            }
            
            s_rating = s_rating.rename(columns=new_column_names)
            s_rating = s_rating.resample('M').mean()
            
            logger.info("Credit rating data retrieved and processed")
        except Exception as e:
            logger.error(f"Error fetching credit rating data: {e}")
        
        # Calculate consolidated risk rating
        risk_rating = pd.DataFrame()
        
        if not m_rating.empty and not s_rating.empty and not f_rating.empty:
            country_rating = pd.concat([m_rating, s_rating, f_rating], axis=1)
            
            countries = ['us', 'gb', 'fr', 'de', 'it', 'pl', 'hu', 'cz', 'za', 'ca', 'au', 'kr']
            averages = {}
            
            for country_code in countries:
                country_columns = [col for col in country_rating.columns if col.endswith(f'_{country_code}')]
                if country_columns:  # Only process countries with data
                    averages[country_code] = country_rating[country_columns].mean(axis=1)
            
            risk_rating = pd.DataFrame(averages)
            
            new_column_names = {
                "us": "rating_us",
                "gb": "rating_gb",
                "fr": "rating_fr",
                "de": "rating_de",
                "it": "rating_it",
                "pl": "rating_pl",
                "hu": "rating_hu",
                "cz": "rating_cz",
                "za": "rating_za",
                "ca": "rating_ca",
                "au": "rating_au",
                "kr": "rating_kr"
            }
            
            risk_rating = risk_rating.rename(columns=new_column_names)
            logger.info("Consolidated risk ratings calculated")
        else:
            logger.warning("Cannot calculate risk rating: one or more rating agencies' data is missing")
        
        # Store yield data in a list for easier iteration
        yield_list = [yld_2yr_ann, yld_5yr_ann, yld_10yr_ann, yld_30yr_ann]
        yield_names = ['yld_2yr', 'yld_5yr', 'yld_10yr', 'yld_30yr']

        # Create tenor-matched forecasts
        logger.info("Creating tenor-matched forecasts...")
        try:
            tenor_forecasts, consolidated_forecasts = create_tenor_matched_forecasts(
                country_list=country_list,
                country_code_mapping=country_code_mapping,
                growth_forecast=gdp,
                growth_forecast_lt=growth_forecast_lt,
                cpi_forecast=cpi_inf,
                cpi_target=cpi_target,
                cutoff_date=None  # Set to a date if you want to use historical cutoff
            )
            
            # Save consolidated forecasts for future use
            for forecast_type in consolidated_forecasts:
                for tenor_name in consolidated_forecasts[forecast_type]:
                    df = consolidated_forecasts[forecast_type][tenor_name]
                    if not df.empty:
                        filename = f"{forecast_type}_{tenor_name}_forecasts.csv"
                        df.to_csv(filename)
                        logger.info(f"Saved {forecast_type} {tenor_name} forecasts to {filename}")
            
            logger.info("Tenor-matched forecasts created successfully")
        except Exception as e:
            logger.error(f"Error creating tenor-matched forecasts: {e}")
            tenor_forecasts = None
            consolidated_forecasts = None

        # Add a section to analyze the forecast data
        if tenor_forecasts is not None:
            logger.info("Analyzing forecast data...")
            forecast_data_summary = {
                'country_coverage': {},
                'tenor_coverage': {
                    'growth': {'2yr': 0, '5yr': 0, '10yr': 0, '30yr': 0},
                    'inflation': {'2yr': 0, '5yr': 0, '10yr': 0, '30yr': 0}
                }
            }
            
            # Count available forecasts by country and tenor
            for country in tenor_forecasts:
                growth_count = sum(1 for tenor in tenor_forecasts[country]['growth'] 
                                if not tenor_forecasts[country]['growth'][tenor].empty)
                inflation_count = sum(1 for tenor in tenor_forecasts[country]['inflation'] 
                                    if not tenor_forecasts[country]['inflation'][tenor].empty)
                
                forecast_data_summary['country_coverage'][country] = {
                    'growth': growth_count,
                    'inflation': inflation_count,
                    'total': growth_count + inflation_count
                }
                
                # Count by tenor
                for tenor in ['2yr', '5yr', '10yr', '30yr']:
                    if (tenor in tenor_forecasts[country]['growth'] and 
                        not tenor_forecasts[country]['growth'][tenor].empty):
                        forecast_data_summary['tenor_coverage']['growth'][tenor] += 1
                    
                    if (tenor in tenor_forecasts[country]['inflation'] and 
                        not tenor_forecasts[country]['inflation'][tenor].empty):
                        forecast_data_summary['tenor_coverage']['inflation'][tenor] += 1
            
            # Display summary
            logger.info("Forecast data coverage:")
            logger.info(f"  Total countries with forecasts: {len(forecast_data_summary['country_coverage'])}")
            
            for forecast_type in ['growth', 'inflation']:
                logger.info(f"  {forecast_type.capitalize()} forecast coverage by tenor:")
                for tenor, count in forecast_data_summary['tenor_coverage'][forecast_type].items():
                    logger.info(f"    {tenor}: {count}/{len(country_list)} countries")
            
            # Store in results summary
            results_summary['forecast_data_summary'] = forecast_data_summary
        
        # Run data diagnostics before modeling
        logger.info("Running data diagnostics...")
        
        # 1. Create data availability summary
        data_availability = create_data_availability_summary(
            country_list=country_list,
            country_code_mapping=country_code_mapping,
            yield_list=yield_list,
            yield_names=yield_names,
            pol_rat=pol_rat,
            cpi_inf=cpi_inf,
            act_track=act_track,
            risk_rating=risk_rating,
            tenor_forecasts=tenor_forecasts

        )
        
        results_summary['data_availability'] = data_availability
        
        # 2. Analyze data overlap issues
        data_overlap = analyze_data_overlap_issues(
            country_list=country_list,
            country_code_mapping=country_code_mapping,
            yield_list=yield_list,
            yield_names=yield_names,
            pol_rat=pol_rat,
            cpi_inf=cpi_inf,
            act_track=act_track,
            risk_rating=risk_rating,
            tenor_forecasts=tenor_forecasts
        )
        
        results_summary['data_overlap'] = data_overlap
        
        # Train and evaluate models
        logger.info("Training and evaluating MLP models...")
        
        # Initialize a DataFrame to store results
        models_df = []
        
        # Iterate through countries and tenors
        for country in country_list:
            results_summary['model_results'][country] = {}
            
            for tenor_name, tenor_data in zip(yield_names, yield_list):
                logger.info(f"Processing {country} - {tenor_name}")
                
                # Train and evaluate model
                model_results = train_evaluate_mlp(
                                country=country,
                                tenor_name=tenor_name,
                                country_code_mapping=country_code_mapping,
                                tenor_data=tenor_data,
                                pol_rat=pol_rat,
                                cpi_inf=cpi_inf,
                                act_track=act_track,
                                risk_rating=risk_rating,
                                tenor_forecasts=tenor_forecasts
                            )
                
                # Store results
                results_summary['model_results'][country][tenor_name] = model_results
                
                # Update success/failure counts
                if model_results['status'] == 'Success':
                    results_summary['success_count'] += 1
                else:
                    results_summary['failure_count'] += 1
                
                # Add to results DataFrame
                models_df.append({
                    'Country': country,
                    'Tenor': tenor_name,
                    'Status': model_results['status'],
                    'Train_RMSE': model_results.get('metrics', {}).get('train', {}).get('rmse'),
                    'Train_R2': model_results.get('metrics', {}).get('train', {}).get('r2'),
                    'Test_RMSE': model_results.get('metrics', {}).get('test', {}).get('rmse'),
                    'Test_R2': model_results.get('metrics', {}).get('test', {}).get('r2'),
                    'Future_RMSE': model_results.get('metrics', {}).get('future', {}).get('rmse'),
                    'Future_R2': model_results.get('metrics', {}).get('future', {}).get('r2'),
                    'Error': model_results.get('error')
                })
        
        # Convert results to DataFrame and save
        if models_df:
            models_results_df = pd.DataFrame(models_df)
            models_results_df.to_csv('model_results_summary.csv', index=False)
            logger.info("Model results summary saved to 'model_results_summary.csv'")
        
        # Create summary visualization
        try:
            plt.figure(figsize=(12, 8))
            
            # Count models by status
            status_counts = {}
            for country in results_summary['model_results']:
                for tenor in results_summary['model_results'][country]:
                    status = results_summary['model_results'][country][tenor]['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
            
            # Create bar chart
            statuses = list(status_counts.keys())
            counts = [status_counts[s] for s in statuses]
            
            colors = {
                'Success': 'green',
                'Failed - Insufficient Data': 'red',
                'Failed - Too Few Data Points': 'orange',
                'Failed - Training Error': 'darkred',
                'Failed - Exception': 'black'
            }
            
            bar_colors = [colors.get(s, 'blue') for s in statuses]
            
            plt.bar(statuses, counts, color=bar_colors)
            plt.title('Model Training Results', fontsize=16)
            plt.xlabel('Status', fontsize=14)
            plt.ylabel('Number of Models', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('model_training_results.png')
            plt.close()
            
            logger.info("Model training results visualization saved to 'model_training_results.png'")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
        
        # Generate unified forecasts for selected models
        logger.info("Generating unified forecasts...")
        
        # Get list of successfully trained models
        successful_models = []
        for country in results_summary['model_results']:
            for tenor in results_summary['model_results'][country]:
                if results_summary['model_results'][country][tenor]['status'] == 'Success':
                    successful_models.append((country, tenor))
        
        # Add a section to the end of main() to create a forecast impact summary
        if tenor_forecasts is not None:
            logger.info("Creating forecast impact summary...")
            
            # Count models with and without forecasts
            forecast_impact = {
                'with_forecasts': {
                    'success_count': 0,
                    'failure_count': 0,
                    'rmse_improvement': []
                },
                'without_forecasts': {
                    'success_count': 0,
                    'failure_count': 0
                }
            }
            
            for country in results_summary['model_results']:
                for tenor in results_summary['model_results'][country]:
                    model_result = results_summary['model_results'][country][tenor]
                    
                    # Check if model used forecasts
                    has_forecasts = model_result.get('includes_forecasts', False)
                    
                    if model_result['status'] == 'Success':
                        if has_forecasts:
                            forecast_impact['with_forecasts']['success_count'] += 1
                            
                            # If we have variant metrics, calculate improvement
                            if 'variant_metrics' in model_result and 'forecast_only' in model_result['variant_metrics']:
                                full_rmse = model_result['metrics']['test']['rmse']
                                forecast_rmse = model_result['variant_metrics']['forecast_only']['test_rmse']
                                
                                improvement = (full_rmse - forecast_rmse) / full_rmse * 100
                                forecast_impact['with_forecasts']['rmse_improvement'].append(improvement)
                        else:
                            forecast_impact['without_forecasts']['success_count'] += 1
        
        # If we have at least one successful model, try to generate a unified forecast
        if successful_models:
            logger.info(f"Found {len(successful_models)} successful models for unified forecasting")
            
            # Select example model (first successful one) to demonstrate unified forecasting
            for country, tenor in successful_models:

                logger.info(f"Creating model prediction visualization for {country} - {tenor} for  unified forecast")
            
                try:
                    visualize_a_model_prediction(
                        country=country,
                        tenor_name=tenor,
                        country_code_mapping=country_code_mapping,
                        tenor_data=yield_list[yield_names.index(tenor)],
                        pol_rat=pol_rat,
                        cpi_inf=cpi_inf,
                        act_track=act_track,
                        risk_rating=risk_rating
                    )
                    logger.info(f"Example model prediction visualization created for {country} - {tenor}")
                except Exception as e:
                    logger.error(f"Error creating example model prediction: {e}")
        else:
            logger.warning("No successful models found for unified forecasting")
        
        logger.info("Yield curve modeling and forecasting pipeline completed")
        
    except Exception as e:
        logger.error(f"Error in yield curve modeling pipeline: {e}")
        import traceback
        traceback.print_exc()

    
    
    return results_summary

if __name__ == "__main__":
    main()