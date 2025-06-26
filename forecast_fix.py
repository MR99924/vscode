"""
SOLUTION: Fix date alignment issues in forecast generator

Key Changes:
1. Use actual data start dates when fetching from APIs
2. Preserve original date indices throughout the pipeline
3. Fix plotting to use actual data dates instead of artificial ranges
4. Handle date alignment properly in unified forecasts
"""

# ============================================================================
# 1. FIX: Update get_forecast_data_for_modelling to use proper date ranges
# ============================================================================

def get_forecast_data_for_modelling(
    mb_client: Macrobond,
    country_list: List[str], 
    country_code_mapping: Dict[str, str],
    cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
    wind_back_years: int = 5,
    target_end_date: str = '2060-12-31',
    forecast_horizon: int = 60,
    full_forecast_horizon: int = 432,
    growth_decay_period: int = 60,
    inflation_decay_period: int = 36
) -> Dict[str, pd.DataFrame]:
    """
    FIXED VERSION: Use actual start dates for data fetching
    """
    logger.info("Starting forecast data generation for modeling")
    
    try:
        # Load historical GDP data with proper date ranges
        logger.info("Loading historical growth data from Macrobond with proper date ranges")
        growth_tickers = list(config.gdp_tickers.keys())
        
        # CHANGE 1: Fetch data with proper start dates per ticker
        growth_data_dict = {}
        for ticker in growth_tickers:
            start_date = config.gdp_start_dates.get(ticker, config.HISTORICAL_START_DATE)
            try:
                single_series = mb_client.FetchSeries([ticker], start_date=start_date)
                if not single_series.empty:
                    growth_data_dict[ticker] = single_series[ticker]
                    logger.debug(f"Fetched {ticker} from {single_series.index[0]} to {single_series.index[-1]}")
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        # Combine into DataFrame and rename
        growth_forecast = pd.DataFrame(growth_data_dict)
        growth_forecast = growth_forecast.rename(columns=config.COLUMN_MAPPINGS['gdp'])
        growth_forecast = growth_forecast.resample('ME').interpolate('linear')
        growth_forecast = growth_forecast.pct_change(periods=12) * 100

        # Load long-term growth data
        logger.info("Loading long-term growth data from Macrobond")
        lt_growth_tickers = list(config.growth_forecast_lt_tickers.keys())
        growth_forecast_lt = mb_client.FetchSeries(lt_growth_tickers)
        growth_forecast_lt = growth_forecast_lt.rename(columns=config.COLUMN_MAPPINGS['growth_forecast_lt'])
        
        # CHANGE 2: Load CPI data with proper date ranges
        logger.info("Loading historical inflation data from Macrobond with proper date ranges")
        cpi_tickers = list(config.cpi_inf_tickers.keys())
        
        cpi_data_dict = {}
        for ticker in cpi_tickers:
            start_date = config.cpi_inf_start_dates.get(ticker, config.HISTORICAL_START_DATE)
            try:
                single_series = mb_client.FetchSeries([ticker], start_date=start_date)
                if not single_series.empty:
                    cpi_data_dict[ticker] = single_series[ticker]
                    logger.debug(f"Fetched {ticker} from {single_series.index[0]} to {single_series.index[-1]}")
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        # Combine CPI data
        cpi_forecast = pd.DataFrame(cpi_data_dict)
        cpi_forecast = cpi_forecast.rename(columns=config.COLUMN_MAPPINGS['cpi_inf'])
        cpi_forecast = cpi_forecast.pct_change(periods=12) * 100

        logger.info("Loading inflation target data from Macrobond")
        target_tickers = list(config.cpi_target_tickers.keys())
        cpi_target = mb_client.FetchSeries(target_tickers)
        cpi_target = cpi_target.rename(columns=config.COLUMN_MAPPINGS['cpi_target'])
        
        # Apply backfill after first observation to preserve data integrity
        logger.info("Applying backfill processing to historical data")
        growth_forecast = apply_backfill_after_first_observation(growth_forecast)
        growth_forecast_lt = apply_backfill_after_first_observation(growth_forecast_lt)
        cpi_forecast = apply_backfill_after_first_observation(cpi_forecast)
        cpi_target = apply_backfill_after_first_observation(cpi_target)

        # Generate unified forecasts
        logger.info("Generating unified forecasts")
        unified_results = demonstrate_unified_forecasts(
            mb_client=mb_client,
            country_list=country_list,
            country_code_mapping=country_code_mapping,
            growth_forecast=growth_forecast,
            growth_forecast_lt=growth_forecast_lt,
            cpi_forecast=cpi_forecast,
            cpi_target=cpi_target,
            cutoff_date=cutoff_date,
            wind_back_years=wind_back_years,
            target_end_date=target_end_date,
            forecast_horizon=forecast_horizon,
            full_forecast_horizon=full_forecast_horizon,
            growth_decay_period=growth_decay_period,
            inflation_decay_period=inflation_decay_period
        )
        
        # CHANGE 3: Convert results to DataFrames preserving actual dates
        logger.info("Converting results to DataFrames with preserved date indices")
        processed_results = {}
        
        for country, forecasts in unified_results.items():
            if forecasts['growth'] is not None and forecasts['inflation'] is not None:
                # CRITICAL FIX: Use the actual data's date index, not artificial dates
                country_code = country_code_mapping[country]
                growth_col = f"gdp_{country_code}"
                inflation_col = f"cpi_inf_{country_code}"
                
                # Get the actual historical start dates from the source data
                growth_start_date = None
                inflation_start_date = None
                
                if growth_col in growth_forecast.columns:
                    historical_growth = growth_forecast[growth_col].dropna()
                    if not historical_growth.empty:
                        growth_start_date = historical_growth.index[0]
                
                if inflation_col in cpi_forecast.columns:
                    historical_inflation = cpi_forecast[inflation_col].dropna()
                    if not historical_inflation.empty:
                        inflation_start_date = historical_inflation.index[0]
                
                # Use the earlier of the two start dates, or fallback
                if growth_start_date and inflation_start_date:
                    actual_start_date = min(growth_start_date, inflation_start_date)
                elif growth_start_date:
                    actual_start_date = growth_start_date
                elif inflation_start_date:
                    actual_start_date = inflation_start_date
                else:
                    # Fallback to config date
                    actual_start_date = pd.Timestamp(config.HISTORICAL_START_DATE)
                
                # Create date index based on actual data length and start date
                total_periods = max(len(forecasts['growth']), len(forecasts['inflation']))
                dates = pd.date_range(start=actual_start_date, periods=total_periods, freq='ME')
                
                # Ensure arrays are same length as dates
                growth = forecasts['growth']
                inflation = forecasts['inflation']
                min_len = min(len(dates), len(growth), len(inflation))
                
                dates = dates[:min_len]
                df_data = {
                    'growth': growth[:min_len],
                    'inflation': inflation[:min_len]
                }
                
                country_df = pd.DataFrame(df_data, index=dates)
                processed_results[country] = country_df
                
                logger.info(f"Processed {country}: {len(country_df)} data points from {dates[0]} to {dates[-1]}")
            else:
                logger.warning(f"Incomplete data for {country}, skipping")
        
        # Generate plots if requested
        if config.GENERATE_PLOTS:
            logger.info("Generating forecast plots")
            plot_unified_forecasts_fixed(unified_results, growth_forecast, cpi_forecast, country_code_mapping)
        
        logger.info(f"Successfully generated forecast data for {len(processed_results)} countries")
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in get_forecast_data_for_modelling: {e}")
        logger.exception("Full traceback:")
        raise


# ============================================================================
# 2. FIX: Updated plotting function to use actual data dates
# ============================================================================

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
            total_periods = max(len(forecasts['growth']), len(forecasts['inflation']))
            dates = pd.date_range(start=actual_start_date, periods=total_periods, freq='ME')
            
            logger.info(f"\nPlotting data for {country}:")
            logger.info(f"Actual start date: {actual_start_date}")
            logger.info(f"Growth data length: {len(forecasts['growth'])}")
            logger.info(f"Inflation data length: {len(forecasts['inflation'])}")
            logger.info(f"Date range: {dates[0]} to {dates[-1]}")
            
            # Plot growth with actual historical data alignment
            growth_data = forecasts['growth']
            growth_dates = dates[:len(growth_data)]
            ax1.plot(growth_dates, growth_data, 'b-', linewidth=1.5)
            ax1.set_title(f"{country}: Unified GDP Growth Forecast")
            ax1.set_ylabel('GDP Growth (%)')
            ax1.set_xlabel('Date')
            ax1.grid(True)
            
            # Plot inflation with actual historical data alignment
            inflation_data = forecasts['inflation']
            inflation_dates = dates[:len(inflation_data)]
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
                if transition_idx < len(dates):
                    transition_date = dates[transition_idx]
                    ax1.axvline(x=transition_date, color='gray', linestyle='--', alpha=0.7)
                    ax1.text(transition_date, ax1.get_ylim()[1] * 0.9, "Forecast start", 
                            rotation=90, verticalalignment='top')
            
            # Format x-axis to show years appropriately
            years_span = (dates[-1] - dates[0]).days / 365.25
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


# ============================================================================
# 3. FIX: Enhanced Macrobond client method to support date ranges
# ============================================================================

def fetch_series_with_date_range(mb_client: Macrobond, tickers: List[str], 
                                start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Enhanced wrapper for Macrobond FetchSeries with proper date range support
    """
    try:
        if start_date:
            # If the Macrobond client supports date ranges, use them
            if hasattr(mb_client, 'FetchSeriesWithDates'):
                return mb_client.FetchSeriesWithDates(tickers, start_date=start_date)
            else:
                # Fallback: fetch all data and filter
                data = mb_client.FetchSeries(tickers)
                if not data.empty and start_date:
                    start_timestamp = pd.Timestamp(start_date)
                    data = data[data.index >= start_timestamp]
                return data
        else:
            return mb_client.FetchSeries(tickers)
    except Exception as e:
        logger.error(f"Error fetching series {tickers} from {start_date}: {e}")
        return pd.DataFrame()


# ============================================================================
# 4. FIX: Add validation function for date consistency
# ============================================================================

def validate_date_consistency(data_dict: Dict[str, pd.DataFrame], 
                            country_list: List[str],
                            country_code_mapping: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Validate that data dates are consistent with expectations
    """
    validation_report = {}
    
    for country in country_list:
        country_code = country_code_mapping[country]
        validation_report[country] = {}
        
        # Check GDP data
        growth_col = f"gdp_{country_code}"
        if 'growth_forecast' in data_dict and growth_col in data_dict['growth_forecast'].columns:
            growth_data = data_dict['growth_forecast'][growth_col].dropna()
            if not growth_data.empty:
                actual_start = growth_data.index[0].strftime('%Y-%m-%d')
                expected_ticker = None
                for ticker, mapped_country in config.gdp_tickers.items():
                    if mapped_country == country:
                        expected_ticker = ticker
                        break
                
                expected_start = config.gdp_start_dates.get(expected_ticker, 'Unknown')
                validation_report[country]['gdp_actual_start'] = actual_start
                validation_report[country]['gdp_expected_start'] = expected_start
                validation_report[country]['gdp_length'] = len(growth_data)
        
        # Check CPI data
        inflation_col = f"cpi_inf_{country_code}"
        if 'cpi_forecast' in data_dict and inflation_col in data_dict['cpi_forecast'].columns:
            inflation_data = data_dict['cpi_forecast'][inflation_col].dropna()
            if not inflation_data.empty:
                actual_start = inflation_data.index[0].strftime('%Y-%m-%d')
                expected_ticker = None
                for ticker, mapped_country in config.cpi_inf_tickers.items():
                    if mapped_country == country:
                        expected_ticker = ticker
                        break
                
                expected_start = config.cpi_inf_start_dates.get(expected_ticker, 'Unknown')
                validation_report[country]['cpi_actual_start'] = actual_start
                validation_report[country]['cpi_expected_start'] = expected_start
                validation_report[country]['cpi_length'] = len(inflation_data)
    
    # Log validation results
    logger.info("=== DATE CONSISTENCY VALIDATION ===")
    for country, report in validation_report.items():
        logger.info(f"\n{country}:")
        for key, value in report.items():
            logger.info(f"  {key}: {value}")
    
    return validation_report


# ============================================================================
# 5. USAGE: How to implement the fix
# ============================================================================

# Replace the existing get_forecast_data_for_modelling function with the fixed version above
# Replace the existing plot_unified_forecasts function with plot_unified_forecasts_fixed
# Add validation call in your main pipeline:

def main_pipeline_with_fixes():
    """
    Example of how to use the fixed functions
    """
    # Initialize clients
    mb_client = Macrobond()
    
    # Use the fixed data fetching
    forecast_data = get_forecast_data_for_modelling(
        mb_client=mb_client,
        country_list=config.country_list,
        country_code_mapping=config.country_list_mapping
    )
    
    # Validate date consistency
    validation_report = validate_date_consistency(
        {'growth_forecast': growth_forecast, 'cpi_forecast': cpi_forecast},
        config.country_list,
        config.country_list_mapping
    )
    
    return forecast_data, validation_report