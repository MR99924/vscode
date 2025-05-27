import sys
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import os
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from macrobond import Macrobond
import bloomberg
import logging
from tqdm import tqdm
from data_worker import forward_fill_to_current_date

# Initialize the logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Initialize Macrobond
mb = Macrobond()
label_encoder = LabelEncoder()

def get_bloomberg_data(tickers, date_from, date_to, field="PX_LAST", periodicity="DAILY"):
    """ Fetch data from Bloomberg for the given tickers and date range. """
    bbg = bloomberg.Bloomberg()
    df = bbg.historicalRequest(
        tickers, field, date_from, date_to, periodicity=periodicity,
        nonTradingDayFillOption="ALL_CALENDAR_DAYS", nonTradingDayFillMethod="PREVIOUS_VALUE",
    )
    # Pivot the dataframe for easier manipulation
    df = pd.pivot_table(df, values='bbergvalue', index='bbergdate', columns='bbergsymbol')
    # Return only the tickers columns
    df = df[tickers]
    return df

def apply_backfill_after_first_observation(df):
    """ Apply backfill but only after the first valid observation for each column. This preserves the integrity of the historical data. """
    for column in df.columns:
        first_valid_index = df[column].first_valid_index()
        if first_valid_index is not None:
            df.loc[first_valid_index:, column] = df.loc[first_valid_index:, column].bfill()
    return df

def s_curve(x):
    """
    S-curve function for smooth transitions. Input t should be normalized between 0 and 1. Returns a value between 0 and 1.
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
    

    result[0] = start_value
    last_valid_idx = 0
    
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
        
        # Calculate blended value based on position in transition
        result[pos] = start_value * (1 - weight) + current_target * weight
        
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
        print(f"Available growth columns: {growth_forecast.columns}")
        print(f"Available CPI columns: {cpi_forecast.columns}")
    
    for country in country_list:
        logging.info(f"Processing country: {country}")
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
                print(f"Available columns: {growth_forecast.columns}")
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
                print(f"\n{country} - Growth Data:")
                print(f"Historical data points: {len(historical_growth)}")
                print(f"Historical data sample: {historical_growth.head()}")
                print(f"Historical data last point: {historical_growth.iloc[-1] if not historical_growth.empty else None}")
                print(f"LT forecast points: {len(lt_growth)}")
                print(f"LT growth sample: {lt_growth.head()}")
            
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
                    complete_date_range = pd.date_range(start=start_date, end=forecast_end_date, freq='M')
                else:
                    complete_date_range = pd.date_range(start=start_date, end=target_end_date, freq='M')
                
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
                print(f"\n{country} - Inflation Data:")
                print(f"Historical inflation points: {len(historical_inflation)}")
                print(f"Historical inflation sample: {historical_inflation.head()}")
                print(f"Historical inflation last point: {historical_inflation.iloc[-1] if not historical_inflation.empty else None}")
                print(f"Target inflation points: {len(target_inflation)}")
                print(f"Target inflation sample: {target_inflation.head()}")
            
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
                    complete_date_range = pd.date_range(start=start_date, end=forecast_end_date, freq='M')
                else:
                    complete_date_range = pd.date_range(start=start_date, end=target_end_date, freq='M')
                
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
    """ Create plots of the unified forecasts including full historical data. """
    for country, forecasts in results.items():
        if forecasts['growth'] is not None and forecasts['inflation'] is not None:
            # Create figure and subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            # Generate appropriate date range starting from historical data
            # Assume our data starts from 1947-01-01 (based on your terminal output)
            # and extends to forecast horizon
            
            total_periods = max(len(forecasts['growth']), len(forecasts['inflation']))
            dates = pd.date_range(start='1947-01-01', periods=total_periods, freq='M')
            # Print diagnostics
            print(f"\nPlotting data for {country}:")
            print(f"Growth data length: {len(forecasts['growth'])}")
            print(f"Inflation data length: {len(forecasts['inflation'])}")
            print(f"Date range: {dates[0]} to {dates[-1]}")
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
            transition_date = dates[forecasts['transition_points']['growth_transition_idx']]
            ax1.axvline(x=transition_date, color='gray',linestyle='--', alpha=0.7 )
            ax1.text(transition_date, ax1.get_ylim()[1] * 0.9, "Forecast start", rotation=90, verticalalignment='top')
            # Format x-axis to show years
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_locator(mdates.YearLocator(10))  # Show every 10 years
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(f"unified_forecasts_{country}.png")
            plt.show()
        else:
            print(f"Missing data for {country}")

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
    print(f"DEBUG: calculate_cumulative_growth called with series of length {len(series)}, periods={periods}")
    if len(series) > 0:
        print(f"DEBUG: First few values in series: {series[:min(5, len(series))]}")
    
    # Check if we have enough data
    if len(series) < periods:
        print(f"DEBUG: Not enough data. Series length {len(series)} < periods {periods}")
        return np.nan
    
    # Check for NaN values
    nan_count = sum(1 for rate in series[:periods] if np.isnan(rate))
    if nan_count > 0:
        print(f"DEBUG: Found {nan_count} NaN values in the first {periods} elements")
        return np.nan
    
    # Initialize cumulative multiplier
    cumulative_multiplier = 1.0
    
    # Compound only the relevant annual rates
    for i in range(periods):
        if i < len(series):
            # Convert annual percentage to multiplier (e.g., 2.5% → 1.025)
            rate = series[i]
            annual_multiplier = 1 + (rate / 100)
            print(f"DEBUG: i={i}, rate={rate}, multiplier={annual_multiplier}")
            cumulative_multiplier *= annual_multiplier
    
    # Convert back to percentage
    cumulative_growth = (cumulative_multiplier - 1) * 100
    print(f"DEBUG: Final cumulative_multiplier={cumulative_multiplier}, cumulative_growth={cumulative_growth}%")
    
    return cumulative_growth
    

def save_historical_forecasts_to_csv(historical_forecasts, forecast_horizons):
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
        #growth_df.to_csv(growth_filename)
        logging.info(f"Saved historical growth forecasts for {country} to {growth_filename}")

        # Save inflation data
        inflation_filename = f"{country}_historical_inflation_forecasts.csv"
        #inflation_df.to_csv(inflation_filename)
        logging.info(f"Saved historical growth forecasts for {country} to {inflation_filename}")

        # Create combined file with both growth and inflation
        combined_df = pd.concat([growth_df, inflation_df], axis=1)
        combined_filename = f"{country}_historical_forecasts.csv"
        combined_df.to_csv(combined_filename)
        logging.info(f"Saved combined historical forecasts for {country} to {combined_filename}")


def save_results_to_csv(results, growth_forecast=None, cpi_forecast=None, 
                        growth_forecast_lt=None, cpi_target=None, 
                        country_code_mapping=None, start_date='1947-01-01'):
    """
    Save the unified forecasts to a CSV file with fixed indexing.
    Now includes historical vs. forecast comparison data and all original data sources.
    
    Parameters:
        results: dict - Dictionary of unified forecasts
        growth_forecast: DataFrame - Original historical GDP growth data (optional)
        cpi_forecast: DataFrame - Original historical inflation data (optional)
        growth_forecast_lt: DataFrame - Original long-term GDP forecast data (optional)
        cpi_target: DataFrame - Original inflation target data (optional)
        country_code_mapping: dict - Mapping from country names to country codes (optional)
        start_date: str - Start date for the date index (default: '1947-01-01')
    """
    logging.info("Starting save_results_to_csv function")
    all_data = []
    
    for country, data in results.items():
        if data['growth'] is not None and data['inflation'] is not None:
            logging.info(f"Processing CSV data for {country}")
            
            # Get lengths of data arrays
            growth_len = len(data['growth'])
            inflation_len = len(data['inflation'])
            
            # Use the minimum length to avoid index errors
            valid_length = min(growth_len, inflation_len)
            logging.info(f"Growth data length: {growth_len}, Inflation data length: {inflation_len}")
            logging.info(f"Using valid length: {valid_length}")
            
            # Create a date range based on the valid length
            dates = pd.date_range(start=start_date, periods=valid_length, freq='M')
            
            # Create data for each date, only up to the valid length
            for i in range(valid_length):
                try:
                    all_data.append({
                        "Country": country,
                        "Date": dates[i],
                        "GDP Growth (%)": data['growth'][i] if i < growth_len else None,
                        "Inflation (%)": data['inflation'][i] if i < inflation_len else None
                    })
                except IndexError as e:
                    logging.error(f"Index error at position {i}: {e}")
                    continue
    
    # Check if we have data to save
    if not all_data:
        logging.warning("No data to save to CSV")
        return
    
    # Create DataFrame and save to CSV
    df_all_data = pd.DataFrame(all_data)
    df_all_data.to_csv("unified_forecasts.csv", index=False)
    logging.info("Results saved to unified_forecasts.csv")
    
    # If additional parameters are provided, create the comparison CSVs with all source data
    if all([growth_forecast is not None, cpi_forecast is not None, 
            growth_forecast_lt is not None, cpi_target is not None, country_code_mapping is not None]):
        # Process by country for the comparison CSVs
        for country, data in results.items():
            if data['growth'] is None or data['inflation'] is None:
                logging.info(f"Missing data for {country}, skipping comparison CSV")
                continue
            
            # Get country code
            country_code = country_code_mapping.get(country)
            if not country_code:
                logging.warning(f"No country code found for {country}, skipping comparison CSV")
                continue
            
            # Get transition points (end of historical data)
            growth_transition_idx = data['transition_points'].get('growth_transition_idx', 0)
            inflation_transition_idx = data['transition_points'].get('inflation_transition_idx', 0)
            
            # Create date ranges for the full series
            growth_dates = pd.date_range(start=start_date, periods=len(data['growth']), freq='M')
            inflation_dates = pd.date_range(start=start_date, periods=len(data['inflation']), freq='M')
            
            # Create DataFrames for growth and inflation
            growth_df = pd.DataFrame({
                'Date': growth_dates,
                'Unified_GDP_Growth': data['growth']
            })
            
            inflation_df = pd.DataFrame({
                'Date': inflation_dates,
                'Unified_Inflation': data['inflation']
            })
            
            # Set Date as index for easier merging
            growth_df.set_index('Date', inplace=True)
            inflation_df.set_index('Date', inplace=True)
            
            # Add a column to indicate historical vs forecast data
            growth_df['GDP_Data_Type'] = 'Historical'
            growth_df.loc[growth_dates[growth_transition_idx+1:], 'GDP_Data_Type'] = 'Forecast'
            
            inflation_df['Inflation_Data_Type'] = 'Historical'
            inflation_df.loc[inflation_dates[inflation_transition_idx+1:], 'Inflation_Data_Type'] = 'Forecast'
            
            # Get original historical and long-term data for comparison
            historical_growth_col = f"gdp_{country_code}"
            historical_inflation_col = f"cpi_inf_{country_code}"
            lt_growth_col = f"gdp_lt_{country_code}"
            inflation_target_col = f"cpi_target_{country_code}"
            
            # Extract historical GDP data and align with date index
            if historical_growth_col in growth_forecast.columns:
                historical_growth = growth_forecast[historical_growth_col].copy()
                # Ensure index is datetime if it's not already
                if not isinstance(historical_growth.index, pd.DatetimeIndex):
                    historical_growth.index = pd.date_range(start=start_date, periods=len(historical_growth), freq='M')
                # Add to the growth DataFrame
                growth_df['Historical_GDP_Growth'] = historical_growth
            
            # Extract long-term GDP forecast data
            if lt_growth_col in growth_forecast_lt.columns:
                lt_growth = growth_forecast_lt[lt_growth_col].copy()
                if not isinstance(lt_growth.index, pd.DatetimeIndex):
                    lt_growth.index = pd.date_range(start=lt_growth.index[0] if hasattr(lt_growth.index, '__getitem__') else start_date, 
                                                    periods=len(lt_growth), freq='M')
                # Add to the growth DataFrame, aligned by date
                growth_df['LT_GDP_Growth'] = lt_growth
            
            # Extract historical inflation data
            if historical_inflation_col in cpi_forecast.columns:
                historical_inflation = cpi_forecast[historical_inflation_col].copy()
                if not isinstance(historical_inflation.index, pd.DatetimeIndex):
                    historical_inflation.index = pd.date_range(start=start_date, periods=len(historical_inflation), freq='M')
                # Add to the inflation DataFrame
                inflation_df['Historical_Inflation'] = historical_inflation
            
            # Extract inflation target data
            if inflation_target_col in cpi_target.columns:
                inflation_target = cpi_target[inflation_target_col].copy()
                if not isinstance(inflation_target.index, pd.DatetimeIndex):
                    inflation_target.index = pd.date_range(start=inflation_target.index[0] if hasattr(inflation_target.index, '__getitem__') else start_date, 
                                                          periods=len(inflation_target), freq='M')
                # Add to the inflation DataFrame
                inflation_df['Inflation_Target'] = inflation_target
            
            # Merge growth and inflation data
            combined_df = pd.concat([growth_df, inflation_df], axis=1)
            
            # Fill NaN values with empty string for clearer distinction
            combined_df.fillna('', inplace=True)
            
            # Save country-specific comparison CSV with all source data
            csv_filename = f"{country}_complete_data.csv"
            combined_df.to_csv(csv_filename)
            logging.info(f"Saved complete data for {country} to {csv_filename}")
            
            # Create a version focusing just on the transition period
            transition_date_growth = growth_dates[growth_transition_idx]
            start_focus = transition_date_growth - pd.DateOffset(years=5)
            end_focus = transition_date_growth + pd.DateOffset(years=10)
            
            # Handle case where dates might be outside the range
            start_focus = max(start_focus, combined_df.index.min())
            end_focus = min(end_focus, combined_df.index.max())
            
            transition_focus_df = combined_df.loc[start_focus:end_focus].copy()
            transition_focus_df.to_csv(f"{country}_transition_focus.csv")
            logging.info(f"Saved transition focus for {country} to {country}_transition_focus.csv")
    else:
        logging.info("Additional parameters not provided, skipping detailed comparison CSVs")

def get_forecast_data(cutoff_date=None, wind_back_years=5):
    """
    Generate forecast data for use in the yield modeller.
    
    Parameters:
        cutoff_date: datetime or str - Date to use as the end of historical data
        wind_back_years: int - Number of years to wind back for transition
        
    Returns:
        dict - Dictionary containing all forecast data
    """
    # Fetch required data (similar to main())
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
    
    # Fetch data using Macrobond
    mb = Macrobond()
    
    # GDP data
    gdp_tickers = ["usgdp", "gbgdp", "frgdp", "degdp", "oecd_qna_00011061", "plnaac0197", 
                  "hugdp", "czgdp", "zagdp", "cagdp", "augdp", "krgdp"]
    growth_forecast = mb.FetchSeries(gdp_tickers)
    # Process growth_forecast data...
    
    # Create mapping between original column names and more readable names
    new_column_names = {
        "usgdp": "gdp_us",
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
    growth_forecast = growth_forecast.rename(columns=new_column_names)
    growth_forecast = growth_forecast.pct_change(periods=4) * 100  # Convert to year-over-year percentage change
    growth_forecast = growth_forecast.resample('M').mean()
    growth_forecast = growth_forecast.ffill()  # Forward fill missing values
    growth_forecast = growth_forecast.bfill()
    
    # CPI data
    cpi_inf_tickers = ["uscpi", "gbcpi", "frcpi", "decpi", "itcpi", "plcpi", 
                      "hucpi", "czcpi", "zapric0000", "cacpi", "aucpi", "krcpi"]
    cpi_forecast = mb.FetchSeries(cpi_inf_tickers)
    
    # Create mapping for CPI columns
    new_column_names = {
        "uscpi": "cpi_inf_us",
        "gbcpi": "cpi_inf_gb",
        "frcpi": "cpi_inf_fr",
        "decpi": "cpi_inf_de",
        "itcpi": "cpi_inf_it",
        "plcpi": "cpi_inf_pl",
        "hucpi": "cpi_inf_hu",
        "czcpi": "cpi_inf_cz",
        "zapric0000": "cpi_inf_za",
        "cacpi": "cpi_inf_ca",
        "aucpi": "cpi_inf_au",
        "krcpi": "cpi_inf_kr"
    }

    # Rename columns and resample to monthly frequency
    cpi_forecast = cpi_forecast.rename(columns=new_column_names)
    cpi_forecast = cpi_forecast.resample('M').mean()
    cpi_forecast = cpi_forecast.pct_change(periods=12) * 100  # Convert to percentage change
    cpi_forecast = cpi_forecast.ffill()  # Forward fill missing values
    cpi_forecast = cpi_forecast.bfill()
    
    # Long-term GDP forecast data
    growth_forecast_lt_tickers = [
        "ih:mb:com:gdp_lt_us:gdp_lt_us", "ih:mb:com:gdp_lt_gb:gdp_lt_gb", "ih:mb:com:gdp_lt_fr:gdp_lt_fr",
        "ih:mb:com:gdp_lt_de:gdp_lt_de", "ih:mb:com:gdp_lt_it:gdp_lt_it", "ih:mb:com:gdp_lt_pl:gdp_lt_pl", 
        "ih:mb:com:gdp_lt_hu:gdp_lt_hu", "ih:mb:com:gdp_lt_za:gdp_lt_za", "ih:mb:com:gdp_lt_ca:gdp_lt_ca", 
        "ih:mb:com:gdp_lt_au:gdp_lt_au", "ih:mb:com:gdp_lt_kr:gdp_lt_kr"
    ]
    
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

    # Rename columns and process
    growth_forecast_lt = growth_forecast_lt.rename(columns=new_column_names)
    growth_forecast_lt = growth_forecast_lt.pct_change(periods=1) * 100  # Convert to percentage change
    growth_forecast_lt = growth_forecast_lt.resample('M')  # Resample to monthly
    growth_forecast_lt = growth_forecast_lt.ffill()  # Forward fill missing values
    
    # Create a date range from March 1947 to the start of gdp_lt
    start_date = '1947-03-31'
    end_date = growth_forecast_lt.index[0] - pd.DateOffset(months=1)
    backfill_dates = pd.date_range(start=start_date, end=end_date, freq='M')

    # Create a DataFrame with the backfill dates and fill with a default value
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
        "hurate0005", "krrate0161", "czrate0064"
    ]
    
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

    # Create a DataFrame with the backfill dates and fill with a default value
    backfill_data = pd.DataFrame(index=backfill_dates, columns=cpi_target.columns)
    backfill_data = backfill_data.fillna(2.0)  # Default value for backfill

    # Concatenate the backfill data with the original data
    cpi_target = cpi_target.fillna(2.0)
    cpi_target = pd.concat([backfill_data, cpi_target])
    
    # Generate unified forecasts
    unified_forecasts = demonstrate_unified_forecasts(
        country_list=country_list,
        country_code_mapping=country_code_mapping,
        growth_forecast=growth_forecast,
        growth_forecast_lt=growth_forecast_lt,
        cpi_forecast=cpi_forecast,
        cpi_target=cpi_target,
        cutoff_date=cutoff_date,
        wind_back_years=wind_back_years
    )
    
    # Convert unified forecasts to DataFrame format for easier use in yield modeller
    forecast_dfs = {}
    
    for country, data in unified_forecasts.items():
        if data['growth'] is not None and data['inflation'] is not None:
            # Start date from 1947
            start_date = '1947-01-01'
            
            # Create date range
            growth_len = len(data['growth'])
            inflation_len = len(data['inflation'])
            max_len = max(growth_len, inflation_len)
            dates = pd.date_range(start=start_date, periods=max_len, freq='M')
            
            # Create DataFrame with proper dates
            df = pd.DataFrame(index=dates)
            
            # Add growth and inflation data
            if growth_len > 0:
                df[f'gdp_forecast_{country_code_mapping[country]}'] = pd.Series(
                    data['growth'], index=dates[:growth_len]
                )
            
            if inflation_len > 0:
                df[f'cpi_forecast_{country_code_mapping[country]}'] = pd.Series(
                    data['inflation'], index=dates[:inflation_len]
                )
            
            forecast_dfs[country] = df
    
    # Combine all forecasts into a single DataFrame
    all_forecasts = pd.concat(forecast_dfs.values(), axis=1)
    
    return {
        "raw_data": {
            "growth_forecast": growth_forecast,
            "growth_forecast_lt": growth_forecast_lt,
            "cpi_forecast": cpi_forecast,
            "cpi_target": cpi_target,
        },
        "unified_forecasts": unified_forecasts,
        "forecast_df": all_forecasts,
        "metadata": {
            "country_list": country_list,
            "country_code_mapping": country_code_mapping,
            "cutoff_date": cutoff_date,
            "wind_back_years": wind_back_years
        }
    }

def get_historical_forecasts(start_date='1990-01-01', forecast_horizons=[24, 60, 120, 360]):
    """
    Generate historical forecasts for use in the yield modeller.
    
    Parameters:
        start_date: str - Start date for historical forecasts
        forecast_horizons: list - List of periods (in months) for forecasts
        
    Returns:
        dict - Dictionary containing historical forecast data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Getting forecast data with horizons {forecast_horizons}")
    
    try:
        # Get forecast data first
        forecast_data = get_forecast_data()
        
        # Generate historical forecasts
        historical_forecasts = generate_historical_forecasts(
            country_list=forecast_data["metadata"]["country_list"],
            country_code_mapping=forecast_data["metadata"]["country_code_mapping"],
            growth_forecast=forecast_data["raw_data"]["growth_forecast"],
            growth_forecast_lt=forecast_data["raw_data"]["growth_forecast_lt"],
            cpi_forecast=forecast_data["raw_data"]["cpi_forecast"],
            cpi_target=forecast_data["raw_data"]["cpi_target"],
            start_date=start_date,
            forecast_horizons=forecast_horizons
        )
        
        # Mapping between month values and tenor names
        tenor_mapping = {
            24: "2yr",
            60: "5yr",
            120: "10yr",
            360: "30yr"
        }
        
        # Convert to DataFrame format for easier use in yield modeller
        historical_dfs = {}
        
        # Dates from historical_forecasts
        dates = historical_forecasts['dates']
        
        # Create DataFrames for each country
        for country, data in historical_forecasts['countries'].items():
            country_code = forecast_data["metadata"]["country_code_mapping"][country]
            df = pd.DataFrame(index=dates)
            
            # Add growth forecasts for different horizons
            for horizon in forecast_horizons:
                horizon_str = str(horizon)
                tenor_label = tenor_mapping.get(horizon, f"{horizon//12}yr")
                
                # Create consistent column naming pattern - both month-based and year-based for flexibility
                if horizon_str in data['growth']:
                    # Format 1: Using years in the column name
                    df[f'gdp_forecast_{tenor_label}_{country_code}'] = data['growth'][horizon_str]
                
                if horizon_str in data['inflation']:
                    # Format 1: Using years in the column name
                    df[f'cpi_forecast_{tenor_label}_{country_code}'] = data['inflation'][horizon_str]
            
            historical_dfs[country] = df
        
        # Combine all historical forecasts into a single DataFrame
        all_historical = pd.concat(historical_dfs.values(), axis=1)
        
        # Add diagnostic info
        logger.info(f"Successfully retrieved historical forecasts")
        logger.info(f"Historical DataFrame shape: {all_historical.shape}")
        logger.info(f"Available columns: {list(all_historical.columns)}")

        return {
            "historical_forecasts": historical_forecasts,
            "historical_df": all_historical,
            "metadata": forecast_data["metadata"]
        }
    
    except Exception as e:
        logger.error(f"Error getting forecast data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty placeholder in case of error
        return {
            'historical_forecasts': None,
            'historical_df': pd.DataFrame(),
            'metadata': {
                'country_list': config.country_list,
                'country_code_mapping': config.country_list_mapping
            }
        }

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
    historical_dates = pd.date_range(start=start_date, end=end_date, freq='M')

    historical_dates = historical_dates + pd.offsets.MonthEnd(0)
    
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
                    
                    # Always calculate with the required number of years
                    # This ensures we always have a forecast even for early dates
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
                    
                    # Always calculate with the required number of years
                    cumulative_inflation = calculate_cumulative_growth(annual_inflation_rates, horizon_years)
                    historical_forecasts['countries'][country]['inflation'][str(horizon)].append(cumulative_inflation)

                else:
                    # No inflation data available - use NaN
                    historical_forecasts['countries'][country]['inflation'][str(horizon)].append(np.nan)
            
            progress_bar.update(1)
    
    historical_dates = historical_dates + pd.offsets.MonthEnd(0)
    
    # Verify that each country and horizon has the correct number of forecast values
    for country in country_list:
        if country in historical_forecasts['countries']:
            for horizon in forecast_horizons:
                horizon_str = str(horizon)
                
                # Check growth forecasts
                if horizon_str in historical_forecasts['countries'][country]['growth']:
                    growth_values = historical_forecasts['countries'][country]['growth'][horizon_str]
                    if len(growth_values) != len(historical_dates):
                        logging.warning(f"Country {country}, Growth Horizon {horizon}: {len(growth_values)} values " 
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
                        logging.warning(f"Country {country}, Inflation Horizon {horizon}: {len(inflation_values)} values " 
                                       f"(expected {len(historical_dates)})")
                        
                        # Pad with NaN if needed to ensure correct length
                        while len(inflation_values) < len(historical_dates):
                            inflation_values.append(np.nan)
                            
                        # Trim if too long (should not happen)
                        if len(inflation_values) > len(historical_dates):
                            inflation_values = inflation_values[:len(historical_dates)]
    
    for item in historical_forecasts:
        print(item)
    
    progress_bar.close()
    return historical_forecasts

def main():
    full_forecast_horizon = 360
    growth_decay_period = 60
    inflation_decay_period = 36 # adapt this by country's historic conversation rates
    forecast_horizon = 84

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

    # GDP data for various countries
    gdp_tickers = ["usgdp", "gbgdp", "frgdp", "degdp", "oecd_qna_00011061", "plnaac0197", "hugdp", "czgdp", "zagdp", "cagdp", "augdp", "krgdp"]

    # Fetch GDP data
    gdp = mb.FetchSeries(gdp_tickers)

    # Create mapping between original column names and more readable names
    new_column_names = {
        "usgdp": "gdp_us",
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



    print(gdp)
    gdp.to_csv("test_gdp.csv")

    # CPI inflation data for various countries
    cpi_inf_tickers = ["uscpi", "gbcpi", "frcpi", "decpi", "itcpi", "plcpi", "hucpi", "czcpi", "zapric0000", "cacpi", "aucpi", "krcpi"]

    # Fetch CPI data
    cpi_inf = mb.FetchSeries(cpi_inf_tickers)

    # Create mapping for CPI columns
    new_column_names = {
        "uscpi": "cpi_inf_us",
        "gbcpi": "cpi_inf_gb",
        "frcpi": "cpi_inf_fr",
        "decpi": "cpi_inf_de",
        "itcpi": "cpi_inf_it",
        "plcpi": "cpi_inf_pl",
        "hucpi": "cpi_inf_hu",
        "czcpi": "cpi_inf_cz",
        "zapric0000": "cpi_inf_za",
        "cacpi": "cpi_inf_ca",
        "aucpi": "cpi_inf_au",
        "krcpi": "cpi_inf_kr"
    }

    # Rename columns and resample to monthly frequency
    cpi_inf = cpi_inf.rename(columns=new_column_names)
    cpi_inf = cpi_inf.resample('M').mean()
    cpi_inf = cpi_inf.pct_change(periods=12) * 100  # Convert to percentage change
    cpi_inf = cpi_inf.ffill()  # Forward fill missing values
    cpi_inf = cpi_inf.bfill()
    cpi_inf.to_csv("test_cpi.csv")

    print(cpi_inf)

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


    # # Growth forecast data (long-term)
    # growth_forecast_lt_tickers = [
    #     "oecd_eo_ltb_00192534","oecd_eo_ltb_00192507", "oecd_eo_ltb_00191952", "oecd_eo_ltb_00191979",
    #     "oecd_eo_ltb_00192114", "oecd_eo_ltb_00192324", "oecd_eo_ltb_00192033", "oecd_eo_ltb_00191871",
    #     "oecd_eo_ltb_00192900", "oecd_eo_ltb_00191844", "oecd_eo_ltb_00191763", "oecd_eo_ltb_00192168"
    # ]

    # # Fetch long-term growth forecast
    # growth_forecast_lt = mb.FetchSeries(growth_forecast_lt_tickers)
    # growth_forecast_lt.index.name = "Date"

    # # Create mapping for growth forecast columns
    # new_column_names = {
    #     "oecd_eo_ltb_00192534": "gdp_lt_us",
    #     "oecd_eo_ltb_00192507": "gdp_lt_gb",
    #     "oecd_eo_ltb_00191952": "gdp_lt_fr",
    #     "oecd_eo_ltb_00191979": "gdp_lt_de",
    #     "oecd_eo_ltb_00192114": "gdp_lt_it",
    #     "oecd_eo_ltb_00192324": "gdp_lt_pl",
    #     "oecd_eo_ltb_00192033": "gdp_lt_hu",
    #     "oecd_eo_ltb_00191871": "gdp_lt_cz",
    #     "oecd_eo_ltb_00192900": "gdp_lt_za",
    #     "oecd_eo_ltb_00191844": "gdp_lt_ca",
    #     "oecd_eo_ltb_00191763": "gdp_lt_au",
    #     "oecd_eo_ltb_00192168": "gdp_lt_kr"
    # }

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
    growth_forecast_lt.to_csv('OECD_forecast.csv')

    print(growth_forecast_lt)

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
    cpi_target.to_csv('cpi_target.csv')

    print(cpi_target)

    unified_forecasts = demonstrate_unified_forecasts(
        country_list=country_list,
        country_code_mapping=country_code_mapping,
        growth_forecast=gdp,
        growth_forecast_lt=growth_forecast_lt,
        cpi_forecast=cpi_inf,
        cpi_target=cpi_target,
        cutoff_date=None,
        wind_back_years=5,
        target_end_date='2060-12-31',
        forecast_horizon=forecast_horizon,
        full_forecast_horizon=full_forecast_horizon,
        growth_decay_period=growth_decay_period,
        inflation_decay_period=inflation_decay_period
    )

    plot_unified_forecasts(unified_forecasts)
    save_results_to_csv(unified_forecasts, gdp, cpi_inf, growth_forecast_lt, cpi_target, country_code_mapping)

    historical_forecasts = generate_historical_forecasts(
        country_list = country_list,
        country_code_mapping=country_code_mapping,
        growth_forecast=gdp,
        growth_forecast_lt=growth_forecast_lt,
        cpi_forecast=cpi_inf,
        cpi_target=cpi_target,
        forecast_horizons=forecast_horizons,
        wind_back_years=5
    )



    save_historical_forecasts_to_csv(historical_forecasts, forecast_horizons)

    return {
        "growth_forecast" : gdp,
        "growth_forecast_lt" : growth_forecast_lt,
        "cpi_forecast" : cpi_inf,
        "cpi_target" : cpi_target,
        "historical_forecasts" : historical_forecasts
    }


if __name__ == "__main__":
    main()

def get_forecast_data_for_modelling(forecast_horizons=[24, 60, 120, 360], start_date='1990-01-01'):
    """
    Get forecast data from forecast_generator for use in yield modeling.
    
    Parameters:
        forecast_horizons: list - List of forecast horizons in months
        start_date: str - Start date for historical forecasts
        
    Returns:
        dict: Dictionary containing historical forecast data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Getting forecast data with horizons {forecast_horizons}")
    
    try:        
        # Get historical forecasts
        historical_forecasts = get_historical_forecasts(
            start_date=start_date,
            forecast_horizons=forecast_horizons
        )
        
        logger.info(f"Successfully retrieved historical forecasts")
        print(historical_forecasts)
        return historical_forecasts
    
    
    except Exception as e:
        logger.error(f"Error getting forecast data: {e}")
        return {
            'historical_forecasts': None,
            'historical_df': pd.DataFrame(),
            'metadata': {
                'country_list': config.country_list,
                'country_code_mapping': config.country_list_mapping
            }
        }
