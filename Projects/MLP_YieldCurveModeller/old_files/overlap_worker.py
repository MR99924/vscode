
import sys
import pandas as pd
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def analyze_data_overlap_issues(country_list, country_code_mapping, yield_list, yield_names, pol_rat, cpi_inf, act_track,
                                risk_rating, historical_forecasts, unemployment_rate):
    """
    Analyze data overlap issues to diagnose why some models have no overlapping data.
    Generates a comprehensive report with date ranges for each data source.
    
    Parameters:
        country_list: List of countries
        country_code_mapping: Mapping from country names to codes
        yield_list: List of yield DataFrames
        yield_names: List of yield tenor names
        pol_rat, cpi_inf, act_track, risk_rating: Feature DataFrames
        historical_forecasts: Historical forecast data from forecast_generator
        debt_gdp: Debt to GDP ratio data (optional)
    
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
        'risk_rating': risk_rating,
        'historical_forecasts' : historical_forecasts,
        'act_track' : act_track,
        'unemployment_rate' : unemployment_rate
    }
    
    # # Add debt_gdp if provided
    # if debt_gdp is not None and not debt_gdp.empty:
    #     feature_sources['debt_gdp'] = debt_gdp
    
    # Add historical forecasts if provided
    historical_df = None
    if historical_forecasts is not None and 'historical_df' in historical_forecasts:
        historical_df = historical_forecasts['historical_df']
        if not historical_df.empty:
            feature_sources['historical_forecasts'] = historical_df
    
    # Define which features to use for each tenor
    tenor_features = {
        'yld_2yr': ['policy_rates', 'inflation', 'activity', 'historical_forecasts_2yr', 'unemployment_rate'],
        'yld_5yr': ['policy_rates', 'inflation', 'risk_rating', 'historical_forecasts_5yr', 'unemployment_rate'],
        'yld_10yr': ['policy_rates', 'inflation', 'activity', 'risk_rating', 'historical_forecasts_10yr', 'unemployment_rate'],
        'yld_30yr': ['policy_rates', 'inflation', 'activity', 'risk_rating', 'historical_forecasts_30yr', 'unemployment_rate']
    }
    
    # # Add debt_gdp and historical_forecasts to appropriate tenors if available
    # if 'debt_gdp' in feature_sources:
    #     for tenor in ['yld_5yr', 'yld_10yr', 'yld_30yr']:
    #         tenor_features[tenor].append('debt_gdp')
    
    if 'historical_forecasts' in feature_sources:
        for tenor in ['yld_2yr', 'yld_5yr', 'yld_10yr', 'yld_30yr']:
            tenor_features[tenor].append('historical_forecasts')
    
    # Mapping from tenor to forecast horizon
    tenor_to_horizon = {
        'yld_2yr': '2yr',
        'yld_5yr': '5yr',
        'yld_10yr': '10yr',
        'yld_30yr': '30yr'
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
            
            # Check for historical forecasts first - these are handled differently
            if 'historical_forecasts' in required_features and 'historical_forecasts' in feature_sources:
                horizon = tenor_to_horizon.get(tenor_name)
                
                # Look for forecast columns for this country and tenor
                if historical_df is not None:
                    forecast_cols = [col for col in historical_df.columns if 
                                    (col.endswith(f"_{country_code}") and 
                                     (col.startswith(f"gdp_forecast_{horizon}") or 
                                      col.startswith(f"cpi_forecast_{horizon}")))]
                    
                    if forecast_cols:
                        print(f"    Found {len(forecast_cols)} forecast columns for {tenor_name}")
                        forecast_data = historical_df[forecast_cols].copy()
                        
                        if not forecast_data.empty:
                            # Calculate stats
                            forecast_start = forecast_data.index.min().strftime('%Y-%m-%d')
                            forecast_end = forecast_data.index.max().strftime('%Y-%m-%d')
                            forecast_count = len(forecast_data)
                            nan_percentage = forecast_data.isna().mean().mean() * 100
                            
                            print(f"    Forecasts: {forecast_start} to {forecast_end} ({forecast_count} points, {nan_percentage:.2f}% NaN)")
                            
                            # Store data and stats
                            all_feature_data['historical_forecasts'] = forecast_data
                            feature_stats['historical_forecasts'] = {
                                'start_date': forecast_start,
                                'end_date': forecast_end,
                                'count': forecast_count,
                                'nan_pct': nan_percentage,
                                'columns': forecast_cols
                            }
                        else:
                            print(f"    Forecast data exists but is empty for {country} - {tenor_name}")
                    else:
                        print(f"    No forecast columns found for {country} - {tenor_name}")
            
            # Process regular feature sources
            for source_name in [s for s in required_features if s != 'historical_forecasts']:
                source_df = feature_sources.get(source_name)
                
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
            combined_features.to_csv(f"combined_features_{country_code}")

            
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



def create_data_availability_summary(country_list, country_code_mapping, yield_list, yield_names, pol_rat, cpi_inf, act_track,
                                     risk_rating, historical_forecasts, unemployment_rate, iip_gdp, predicted_yields=None):
    """
    Create a comprehensive summary of data availability across all data sources,
    countries, and tenors to identify gaps and coverage.
    
    Parameters:
        country_list: List of countries
        country_code_mapping: Mapping from country names to codes
        yield_list: List of yield DataFrames
        yield_names: List of yield tenor names
        pol_rat, cpi_inf, act_track, risk_rating: Feature DataFrames
        historical_forecasts: Historical forecast data from forecast_generator
        debt_gdp: Debt to GDP ratio data (optional)
        
    Returns:
        dict: Comprehensive data availability statistics
    """
    print("\n=== DATA AVAILABILITY SUMMARY ===")
    
    # Define data sources
    data_sources = {
        'policy_rates': pol_rat,
        'inflation': cpi_inf,
        'risk_rating': risk_rating,
        'historical_forecasts' : historical_forecasts,
        'unemployment_rate' : unemployment_rate,
        'predicted_yields' : predicted_yields,
        'act_track' : act_track,
        'iip_gdp' : iip_gdp
    }
    
    
    # Add historical forecasts if provided
    if historical_forecasts is not None and 'historical_df' in historical_forecasts:
        historical_df = historical_forecasts['historical_df']
        if not historical_df.empty:
            data_sources['historical_forecasts'] = historical_df
    
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
    
    # Mapping from tenor to forecast horizon
    tenor_to_horizon = {
        'yld_2yr': '2yr',
        'yld_5yr': '5yr',
        'yld_10yr': '10yr',
        'yld_30yr': '30yr'
    }
    
    # Define a function to calculate date range statistics
    def get_date_stats(df, country_code=None, column_filter=None):
        """
        Calculate statistics for a DataFrame.
        
        Parameters:
            df: DataFrame to analyze
            country_code: Country code to filter columns (optional)
            column_filter: Additional filter function for columns (optional)
            
        Returns:
            dict: Statistics about the DataFrame
        """
        if df is None or df.empty:
            return {
                'start_date': None,
                'end_date': None,
                'count': 0,
                'nan_pct': 100.0,
                'columns': []
            }
        
        # If country_code is provided, filter columns
        if country_code:
            # Start with columns that end with country_code
            country_cols = [col for col in df.columns if col.endswith(f"_{country_code}")]
            
            # Apply additional column filter if provided
            if column_filter and callable(column_filter):
                country_cols = [col for col in country_cols if column_filter(col)]
            
            if country_cols:
                df_subset = df[country_cols]
            else:
                return {
                    'start_date': None,
                    'end_date': None,
                    'count': 0,
                    'nan_pct': 100.0,
                    'columns': []
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
                'nan_pct': nan_pct,
                'columns': list(df_subset.columns)
            }
        else:
            return {
                'start_date': None,
                'end_date': None,
                'count': 0,
                'nan_pct': 100.0,
                'columns': []
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
        if source_name == 'historical_forecasts':
            print(f"    Available columns: {len(stats['columns'])}")
            # Print a sample of column names to help debug
            if stats['columns']:
                print(f"    Sample columns: {stats['columns'][:5]}...")
    
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
            # For historical forecasts, use a special filter based on tenor
            if source_name == 'historical_forecasts':
                all_forecast_stats = {}
                
                for tenor_name in yield_names:
                    horizon = tenor_to_horizon.get(tenor_name)
                    
                    # Filter function to find relevant forecast columns
                    def forecast_filter(col):
                        return (col.startswith(f"gdp_forecast_{horizon}") or 
                                col.startswith(f"cpi_forecast_{horizon}"))
                    
                    tenor_stats = get_date_stats(source_df, country_code, forecast_filter)
                    
                    if tenor_stats['columns']:
                        print(f"      {source_name} for {tenor_name}: {tenor_stats['start_date'] or 'N/A'} to "
                              f"{tenor_stats['end_date'] or 'N/A'} ({len(tenor_stats['columns'])} columns, "
                              f"{tenor_stats['count']} obs, {tenor_stats['nan_pct']:.2f}% NaN)")
                        
                        all_forecast_stats[tenor_name] = tenor_stats
                
                # Store all forecast stats
                if all_forecast_stats:
                    availability['by_country'][country]['sources'][source_name] = all_forecast_stats
                else:
                    print(f"      {source_name}: No data available")
                    availability['by_country'][country]['sources'][source_name] = {
                        'start_date': None,
                        'end_date': None,
                        'count': 0,
                        'nan_pct': 100.0,
                        'columns': []
                    }
            else:
                # Regular data source
                stats = get_date_stats(source_df, country_code)
                availability['by_country'][country]['sources'][source_name] = stats
                
                date_range = f"{stats['start_date'] or 'N/A'} to {stats['end_date'] or 'N/A'}"
                print(f"      {source_name}: {date_range} ({len(stats['columns'])} columns, {stats['count']} obs, {stats['nan_pct']:.2f}% NaN)")
        
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
                        'nan_pct': 0.0,  # We already dropped NaN values
                        'columns': [yield_col]
                    }
                else:
                    stats = {
                        'start_date': None,
                        'end_date': None,
                        'count': 0,
                        'nan_pct': 100.0,
                        'columns': []
                    }
            else:
                stats = {
                    'start_date': None,
                    'end_date': None,
                    'count': 0,
                    'nan_pct': 100.0,
                    'columns': []
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
            horizon = tenor_to_horizon.get(tenor_name)
            
            # Define which features we need for this tenor
            if tenor_name == 'yld_2yr':
                required_sources = ['policy_rates', 'inflation', 'activity']
            elif tenor_name == 'yld_5yr':
                required_sources = ['policy_rates', 'inflation', 'risk_rating']
                # if 'debt_gdp' in data_sources:
                #     required_sources.append('debt_gdp')
            else:  # 10yr, 30yr
                required_sources = ['policy_rates', 'inflation', 'activity', 'risk_rating']
                # if 'debt_gdp' in data_sources:
                #     required_sources.append('debt_gdp')
            
            # Add historical forecasts if available
            if 'historical_forecasts' in data_sources:
                required_sources.append('historical_forecasts')
            
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
                source_df = data_sources.get(source_name)
                
                if source_df is None or source_df.empty:
                    sources_available[source_name] = False
                    continue
                
                # Special handling for historical forecasts
                if source_name == 'historical_forecasts':
                    # Find forecast columns for this tenor and country
                    forecast_cols = [col for col in source_df.columns if 
                                    col.endswith(f"_{country_code}") and 
                                    (col.startswith(f"gdp_forecast_{horizon}") or 
                                     col.startswith(f"cpi_forecast_{horizon}"))]
                    
                    if forecast_cols:
                        forecast_data = source_df[forecast_cols]
                        if not forecast_data.empty:
                            sources_available[source_name] = True
                        else:
                            sources_available[source_name] = False
                    else:
                        sources_available[source_name] = False
                else:
                    # Regular source
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
            
            # Add special analysis for forecasts
            if 'historical_forecasts' in data_sources:
                forecast_availability = []
                historical_df = data_sources['historical_forecasts']
                
                for country in country_list:
                    if country not in country_code_mapping:
                        continue
                    
                    country_code = country_code_mapping[country]
                    row = {'Country': country}
                    
                    for tenor_name in yield_names:
                        horizon = tenor_to_horizon.get(tenor_name)
                        
                        # Count forecast columns for this tenor and country
                        forecast_cols = [col for col in historical_df.columns if 
                                        col.endswith(f"_{country_code}") and 
                                        (col.startswith(f"gdp_forecast_{horizon}") or 
                                         col.startswith(f"cpi_forecast_{horizon}"))]
                        
                        row[tenor_name] = len(forecast_cols)
                    
                    forecast_availability.append(row)
                
                if forecast_availability:
                    forecast_df = pd.DataFrame(forecast_availability)
                    forecast_df.set_index('Country', inplace=True)
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(forecast_df, cmap='YlGnBu', annot=True, fmt='d')
                    
                    plt.title('Forecast Column Availability by Country and Tenor', fontsize=16)
                    plt.tight_layout()
                    plt.savefig('forecast_availability_heatmap.png')
                    plt.close()
                    
                    print("\nForecast availability heatmap saved to 'forecast_availability_heatmap.png'")
                    forecast_df.to_csv('forecast_availability_summary.csv')
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== END OF DATA AVAILABILITY SUMMARY ===\n")
    
    return availability