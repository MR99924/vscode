import numpy as np
import os
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback

# Project imports
import config

# Module level logger (will be configured by main application)
logger = logging.getLogger(__name__)


def visualize_model_prediction(country, tenor_name, actual_data, predicted_data, 
                              feature_data=None, model_metrics=None, 
                              output_dir=None):
    """
    Create a visualization of model predictions vs actual data.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        actual_data: pd.Series - Actual yield data with datetime index
        predicted_data: pd.Series - Predicted yield data with datetime index
        feature_data: pd.DataFrame - Feature data for additional plots (optional)
        model_metrics: dict - Dictionary containing model performance metrics (optional)
        output_dir: str - Directory to save plots (optional, uses config if not provided)
        
    Returns:
        str: Path to saved visualization file
    """
    logger.info(f"Creating model prediction visualization for {country} - {tenor_name}")
    
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = getattr(config, 'VISUALIZATION_DIR', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the main prediction plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual data
        if not actual_data.empty:
            ax.plot(actual_data.index, actual_data.values, 'b-', 
                   label='Actual', linewidth=2, alpha=0.8)
        
        # Plot predicted data
        if not predicted_data.empty:
            ax.plot(predicted_data.index, predicted_data.values, 'r--', 
                   label='Predicted', linewidth=2, alpha=0.8)
        
        # Add model metrics annotation if provided
        if model_metrics:
            metrics_text = _format_metrics_text(model_metrics)
            ax.annotate(metrics_text, 
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
        
        # Formatting
        ax.set_title(f"{country} - {tenor_name}: Model Prediction", fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        _format_date_axis(ax)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{country}_{tenor_name}_prediction.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Model prediction visualization saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating model prediction visualization: {e}")
        logger.exception("Full traceback:")
        return None


def visualize_model_with_features(country, tenor_name, actual_data, predicted_data,
                                 feature_data, feature_correlations=None, 
                                 output_dir=None):
    """
    Create a comprehensive visualization showing model predictions and key features.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name
        actual_data: pd.Series - Actual yield data
        predicted_data: pd.Series - Predicted yield data
        feature_data: pd.DataFrame - Feature data to plot
        feature_correlations: dict - Feature correlations with target (optional)
        output_dir: str - Directory to save plots (optional)
        
    Returns:
        str: Path to saved visualization file
    """
    logger.info(f"Creating comprehensive visualization for {country} - {tenor_name}")
    
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = getattr(config, 'VISUALIZATION_DIR', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit number of features to display (top 5 most correlated)
        if feature_correlations:
            top_features = sorted(feature_correlations.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            selected_features = [feat[0] for feat in top_features if feat[0] in feature_data.columns]
        else:
            selected_features = list(feature_data.columns)[:5]
        
        # Create subplots
        n_plots = len(selected_features) + 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        
        # Ensure axes is always a list
        if n_plots == 1:
            axes = [axes]
        
        # Plot yield data in the first subplot
        axes[0].plot(actual_data.index, actual_data.values, 'b-', 
                    label='Actual', linewidth=2)
        if not predicted_data.empty:
            axes[0].plot(predicted_data.index, predicted_data.values, 'r--', 
                        label='Predicted', linewidth=2)
        
        axes[0].set_title(f"{country} - {tenor_name}: Yield and Key Features", 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Yield (%)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot each selected feature
        for i, feature in enumerate(selected_features):
            ax = axes[i + 1]
            
            if feature in feature_data.columns:
                feature_series = feature_data[feature].dropna()
                ax.plot(feature_series.index, feature_series.values, 'k-', linewidth=1.5)
                ax.set_ylabel(_format_feature_name(feature), fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add correlation annotation if available
                if feature_correlations and feature in feature_correlations:
                    corr_value = feature_correlations[feature]
                    ax.annotate(f"Corr: {corr_value:.3f}", 
                               xy=(0.02, 0.85), xycoords='axes fraction',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                               fontsize=9)
            else:
                ax.text(0.5, 0.5, f"Feature '{feature}' not available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(feature, fontsize=10)
        
        # Format x-axis on bottom subplot
        _format_date_axis(axes[-1])
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{country}_{tenor_name}_with_features.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Feature visualization saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating feature visualization: {e}")
        logger.exception("Full traceback:")
        return None


def visualize_available_data(country, tenor_name, yield_data, output_dir=None):
    """
    Create a simple visualization of available yield data when model training isn't possible.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name
        yield_data: pd.Series - Available yield data
        output_dir: str - Directory to save plots (optional)
        
    Returns:
        str: Path to saved visualization file
    """
    logger.info(f"Creating available data visualization for {country} - {tenor_name}")
    
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = getattr(config, 'VISUALIZATION_DIR', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if not yield_data.empty:
            plt.plot(yield_data.index, yield_data.values, 'b-', 
                    label=f'Available {tenor_name} Data', linewidth=2)
            
            # Add data statistics
            stats_text = (f"Data Points: {len(yield_data)}\n"
                         f"Range: {yield_data.min():.2f}% - {yield_data.max():.2f}%\n"
                         f"Mean: {yield_data.mean():.2f}%")
            
            plt.annotate(stats_text, 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)
        else:
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=16)
        
        plt.title(f"{country} - {tenor_name}: Available Data", fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Yield (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        _format_date_axis(plt.gca())
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{country}_{tenor_name}_available_data.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Available data visualization saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating available data visualization: {e}")
        logger.exception("Full traceback:")
        return None


def create_model_summary_visualization(results_summary, output_dir=None):
    """
    Create comprehensive visualization of model training results.
    
    Parameters:
        results_summary: dict - Dictionary containing model results
        output_dir: str - Directory to save plots (optional)
        
    Returns:
        list: Paths to saved visualization files
    """
    logger.info("Creating model summary visualization")
    
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = getattr(config, 'VISUALIZATION_DIR', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Create training status summary
        status_file = _create_training_status_chart(results_summary, output_dir)
        if status_file:
            saved_files.append(status_file)
        
        # Create performance summary for successful models
        if results_summary.get('success_count', 0) > 0:
            performance_file = _create_performance_summary_chart(results_summary, output_dir)
            if performance_file:
                saved_files.append(performance_file)
        
        # Create country-tenor heatmap
        heatmap_file = _create_model_success_heatmap(results_summary, output_dir)
        if heatmap_file:
            saved_files.append(heatmap_file)
        
        logger.info(f"Created {len(saved_files)} summary visualizations")
        return saved_files
        
    except Exception as e:
        logger.error(f"Error creating model summary visualization: {e}")
        logger.exception("Full traceback:")
        return []


def create_performance_comparison_chart(performance_data, output_dir=None):
    """
    Create a comparison chart of model performance across countries and tenors.
    
    Parameters:
        performance_data: pd.DataFrame - DataFrame with columns: Country, Tenor, Metric, Value
        output_dir: str - Directory to save plots (optional)
        
    Returns:
        str: Path to saved visualization file
    """
    logger.info("Creating performance comparison chart")
    
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = getattr(config, 'VISUALIZATION_DIR', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # RMSE comparison
        rmse_data = performance_data[performance_data['Metric'] == 'RMSE']
        if not rmse_data.empty:
            rmse_pivot = rmse_data.pivot(index='Country', columns='Tenor', values='Value')
            sns.heatmap(rmse_pivot, annot=True, fmt='.3f', ax=ax1, 
                       cmap='Reds', cbar_kws={'label': 'RMSE'})
            ax1.set_title('Model RMSE by Country and Tenor', fontweight='bold')
        
        # R² comparison
        r2_data = performance_data[performance_data['Metric'] == 'R2']
        if not r2_data.empty:
            r2_pivot = r2_data.pivot(index='Country', columns='Tenor', values='Value')
            sns.heatmap(r2_pivot, annot=True, fmt='.3f', ax=ax2, 
                       cmap='Greens', cbar_kws={'label': 'R²'})
            ax2.set_title('Model R² by Country and Tenor', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = "model_performance_comparison.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Performance comparison chart saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating performance comparison chart: {e}")
        logger.exception("Full traceback:")
        return None


def _create_training_status_chart(results_summary, output_dir):
    """Create chart showing training status distribution."""
    try:
        # Count models by status
        status_counts = {}
        for country in results_summary.get('model_results', {}):
            for tenor in results_summary['model_results'][country]:
                status = results_summary['model_results'][country][tenor].get('status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
        
        if not status_counts:
            logger.warning("No status data found for visualization")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Define colors for different statuses
        colors = {
            'Success': '#2E8B57',           # SeaGreen
            'Failed - Insufficient Data': '#DC143C',     # Crimson
            'Failed - Too Few Data Points': '#FF8C00',   # DarkOrange
            'Failed - Training Error': '#8B0000',        # DarkRed
            'Failed - Exception': '#2F4F4F',             # DarkSlateGray
            'Data Prepared': '#4169E1',                   # RoyalBlue
            'Not Started': '#A9A9A9'                     # DarkGray
        }
        
        statuses = list(status_counts.keys())
        counts = [status_counts[s] for s in statuses]
        bar_colors = [colors.get(s, '#4169E1') for s in statuses]
        
        bars = plt.bar(statuses, counts, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Training Status Summary', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Training Status', fontsize=14)
        plt.ylabel('Number of Models', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add summary statistics
        total_models = sum(counts)
        success_rate = (status_counts.get('Success', 0) / total_models * 100) if total_models > 0 else 0
        
        plt.text(0.02, 0.98, f'Total Models: {total_models}\nSuccess Rate: {success_rate:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        filename = "model_training_status.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating training status chart: {e}")
        return None


def _create_performance_summary_chart(results_summary, output_dir):
    """Create performance summary chart for successful models."""
    try:
        # Extract performance data
        performance_data = []
        
        for country in results_summary.get('model_results', {}):
            for tenor in results_summary['model_results'][country]:
                result = results_summary['model_results'][country][tenor]
                
                if (result.get('status', '').startswith('Success') and 
                    'metrics' in result and 'test' in result['metrics']):
                    
                    test_metrics = result['metrics']['test']
                    performance_data.append({
                        'Country': country,
                        'Tenor': tenor,
                        'RMSE': test_metrics.get('rmse', np.nan),
                        'R2': test_metrics.get('r2', np.nan)
                    })
        
        if not performance_data:
            logger.warning("No performance data found for visualization")
            return None
        
        df = pd.DataFrame(performance_data)
        
        # Create grouped bar chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # RMSE chart
        _create_grouped_bar_chart(df, 'RMSE', ax1, 'Model RMSE by Country and Tenor', 'RMSE')
        
        # R² chart
        _create_grouped_bar_chart(df, 'R2', ax2, 'Model R² by Country and Tenor', 'R²')
        
        plt.tight_layout()
        
        filename = "model_performance_summary.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating performance summary chart: {e}")
        return None


def _create_model_success_heatmap(results_summary, output_dir):
    """Create heatmap showing model success by country and tenor."""
    try:
        # Create success matrix
        countries = list(results_summary.get('model_results', {}).keys())
        tenors = set()
        
        # Get all unique tenors
        for country in results_summary.get('model_results', {}):
            tenors.update(results_summary['model_results'][country].keys())
        
        tenors = sorted(list(tenors))
        
        if not countries or not tenors:
            logger.warning("No data found for success heatmap")
            return None
        
        # Create success matrix
        success_matrix = []
        for country in countries:
            row = []
            for tenor in tenors:
                if tenor in results_summary['model_results'][country]:
                    status = results_summary['model_results'][country][tenor].get('status', '')
                    success = 1 if status.startswith('Success') else 0
                else:
                    success = np.nan
                row.append(success)
            success_matrix.append(row)
        
        # Create DataFrame
        success_df = pd.DataFrame(success_matrix, index=countries, columns=tenors)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(success_df, annot=True, fmt='.0f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Success (1) / Failure (0)'}, 
                   linewidths=0.5)
        
        plt.title('Model Training Success by Country and Tenor', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Tenor', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        
        plt.tight_layout()
        
        filename = "model_success_heatmap.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating success heatmap: {e}")
        return None


def _create_grouped_bar_chart(df, metric, ax, title, ylabel):
    """Helper function to create grouped bar chart."""
    tenors = df['Tenor'].unique()
    countries = df['Country'].unique()
    
    x = np.arange(len(countries))
    width = 0.2
    
    for i, tenor in enumerate(tenors):
        tenor_data = df[df['Tenor'] == tenor]
        values = [tenor_data[tenor_data['Country'] == country][metric].iloc[0] 
                 if not tenor_data[tenor_data['Country'] == country].empty 
                 else np.nan for country in countries]
        
        ax.bar(x + i * width, values, width, label=tenor, alpha=0.8)
    
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(tenors) - 1) / 2)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.legend(title='Tenor')
    ax.grid(axis='y', alpha=0.3)


def _format_metrics_text(model_metrics):
    """Format model metrics for display in plots."""
    metrics_lines = []
    
    for split in ['train', 'test', 'validation']:
        if split in model_metrics:
            split_metrics = model_metrics[split]
            if 'rmse' in split_metrics and 'r2' in split_metrics:
                metrics_lines.append(f"{split.title()} RMSE: {split_metrics['rmse']:.4f}")
                metrics_lines.append(f"{split.title()} R²: {split_metrics['r2']:.4f}")
    
    return '\n'.join(metrics_lines)


def _format_feature_name(feature_name):
    """Format feature names for better display."""
    # Replace underscores with spaces and capitalize
    formatted = feature_name.replace('_', ' ').title()
    
    # Handle common abbreviations
    replacements = {
        'Pol Rat': 'Policy Rate',
        'Cpi Inf': 'CPI Inflation',
        'Act Track': 'Activity Tracker',
        'Iip Gdp': 'IIP/GDP',
        'Gdp': 'GDP',
        'Us': 'US',
        'Ecb': 'ECB'
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted


def _format_date_axis(ax):
    """Format date axis for consistent display."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis='x', rotation=45)