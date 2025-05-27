"""
Model testing and sensitivity analysis utilities for yield curve modeling.
This module provides comprehensive model testing, feature sensitivity analysis,
and economic theory conformity validation.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List
import re
from matplotlib.gridspec import GridSpec
import traceback
import json

import config

# Initialize logging
logger = logging.getLogger(__name__)


def prepare_feature_impacts(consolidated_results: Dict[str, Any]) -> List[Dict]:
    """
    Prepare feature impacts from consolidated results.
    
    Parameters:
        consolidated_results: Consolidated sensitivity analysis results
    
    Returns:
        List of dictionaries containing feature impact data
    """
    all_feature_impacts = []
    
    for country_tenor, results in consolidated_results['feature_impact_summary'].items():
        for feature, impact_data in results['feature_impacts'].items():
            feature_impact = {
                'feature': feature,
                'country_tenor': country_tenor,
                'mean_impact': float(impact_data['mean_impact']),
                'std_impact': float(impact_data.get('std_impact', 0))
            }
            all_feature_impacts.append(feature_impact)
    
    return all_feature_impacts


def create_economic_theory_conformity_charts(consolidated_results: Dict[str, Any], 
                                           output_dir: str) -> None:
    """
    Create comprehensive economic theory conformity visualizations.
    
    Parameters:
        consolidated_results: Consolidated sensitivity analysis results
        output_dir: Directory to save visualization outputs
    """
    # Aggregate feature impacts across all models
    all_feature_impacts = []
    for model_results in consolidated_results['feature_impact_summary'].values():
        for feature, impact_data in model_results['feature_impacts'].items():
            # Define expected direction based on feature name
            expected_direction = determine_expected_direction(feature)
            
            all_feature_impacts.append({
                'feature': feature,
                'mean_impact': impact_data['mean_impact'],
                'expected_direction': expected_direction
            })
    
    # Convert to DataFrame
    impacts_df = pd.DataFrame(all_feature_impacts)
    
    # Determine feature conformity
    def check_conformity(row):
        if row['expected_direction'] > 0 and row['mean_impact'] > 0:
            return 'Positive Match'
        elif row['expected_direction'] < 0 and row['mean_impact'] < 0:
            return 'Negative Match'
        else:
            return 'Mismatch'
    
    impacts_df['conformity'] = impacts_df.apply(check_conformity, axis=1)
    
    # 1. Overall Conformity Pie Chart
    plt.figure(figsize=(10, 6))
    conformity_counts = impacts_df['conformity'].value_counts()
    
    colors = {
        'Positive Match': '#2ecc71',   # Green
        'Negative Match': '#299c99',   # Teal
        'Neutral': '#3498db',          # Blue
        'Mismatch': '#e74c3c'          # Red
    }
    
    plt.pie(
        conformity_counts, 
        labels=conformity_counts.index, 
        autopct='%1.1f%%',
        colors=[colors.get(conf, '#95a5a6') for conf in conformity_counts.index]
    )
    plt.title('Feature Impact Conformity to Economic Expectations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_conformity_pie.png'))
    plt.close()
    
    # 2. Detailed Conformity Bar Chart
    plt.figure(figsize=(12, 6))
    conformity_by_feature = impacts_df.groupby('feature')['conformity'].agg(
        lambda x: pd.Series({
            'Positive Match': (x == 'Positive Match').mean() * 100,
            'Negative Match': (x == 'Negative Match').mean() * 100,
            'Neutral': (x == 'Neutral').mean() * 100,
            'Mismatch': (x == 'Mismatch').mean() * 100
        })
    )
    
    # Sort by total conformity
    conformity_by_feature['total_conformity'] = (
        conformity_by_feature['Positive Match'] + 
        conformity_by_feature['Negative Match'] + 
        conformity_by_feature.get('Neutral', 0)
    )
    conformity_by_feature = conformity_by_feature.sort_values('total_conformity', ascending=False)
    
    # Plot stacked bar chart
    plt.figure(figsize=(15, 8))
    conformity_by_feature[['Positive Match', 'Negative Match', 'Neutral', 'Mismatch']].plot(
        kind='bar', 
        stacked=True, 
        color=[
            '#2ecc71',   # Positive Match Green
            '#299c99',   # Negative Match Teal
            '#3498db',   # Neutral Blue
            '#e74c3c'    # Mismatch Red
        ]
    )
    plt.title('Feature Impact Conformity Breakdown')
    plt.xlabel('Features')
    plt.ylabel('Percentage of Conformity')
    plt.legend(title='Conformity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_conformity_breakdown.png'))
    plt.close()
    
    # 3. Detailed Conformity Summary CSV
    conformity_summary = impacts_df.groupby('feature')['conformity'].value_counts(normalize=True).unstack()
    conformity_summary = conformity_summary.fillna(0) * 100
    conformity_summary.to_csv(os.path.join(output_dir, 'feature_conformity_summary.csv'))


def determine_expected_direction(feature: str) -> int:
    """
    Determine the expected economic impact direction based on feature name.
    
    Parameters:
        feature: Feature name
    
    Returns:
        Expected direction (1: positive, -1: negative, 0: neutral)
    """
    # Mapping of feature prefixes to expected economic impact
    impact_mapping = {
        # Positive correlations
        'pred_yld': 1,       # Predicted yields typically positive
        'risk_rating': 1,    # Risk ratings typically positive
        'cpi_forecast': 1,   # Inflation forecasts typically positive
        'cpi_inf': 1,        # Inflation indicators typically positive
        'gdp_forecast': 1,   # GDP forecasts typically positive
        'act_track': 1,      # Activity tracking typically positive
        
        # Negative correlations
        'unemployment_rate': -1,  # Unemployment rates typically negative
        
        # Neutral or context-dependent
        'pol_rat': 1        # Policy rates can be complex
    }
    
    # Check feature against mapping
    for prefix, direction in impact_mapping.items():
        if prefix.lower() in feature.lower():
            return direction
    
    # Default to neutral if no match
    return 0


def create_enhanced_visualizations(sensitivity_results: Dict[str, Any], country: str, 
                                 tenor: str, output_file_prefix: str, 
                                 shock_results: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Create comprehensive set of visualizations for model sensitivity analysis.
    
    Parameters:
        sensitivity_results: Results from sensitivity analysis
        country: Country name
        tenor: Yield tenor
        output_file_prefix: Prefix for output files
        shock_results: Results from multiple shock levels (optional)
        
    Returns:
        DataFrame containing impact data
    """
    # Extract feature impacts
    feature_impacts = sensitivity_results['feature_impacts']
    expected_impacts = sensitivity_results.get('expected_impacts', {})
    
    # Prepare data for visualization
    impact_data = []
    for feature, impact in feature_impacts.items():
        # Determine expected impact direction
        expected_direction = 0
        for prefix, direction in expected_impacts.items():
            if prefix in feature.lower():
                expected_direction = direction
                break
        
        # Determine actual direction and alignment
        actual_direction = np.sign(impact['mean_impact'])
        alignment = 'Neutral'
        if expected_direction > 0 and actual_direction > 0:
            alignment = 'Positive Match'
        elif expected_direction < 0 and actual_direction < 0:
            alignment = 'Negative Match'
        elif expected_direction > 0 and actual_direction < 0:
            alignment = 'Positive Mismatch'
        elif expected_direction < 0 and actual_direction > 0:
            alignment = 'Negative Mismatch'
        elif expected_direction == 0:
            alignment = 'Neutral'
        
        impact_data.append({
            'feature': feature,
            'mean_impact': impact['mean_impact'],
            'std_impact': impact['std_impact'],
            'min_impact': impact.get('min_impact', impact['mean_impact'] - impact['std_impact']),
            'max_impact': impact.get('max_impact', impact['mean_impact'] + impact['std_impact']),
            'expected_direction': expected_direction,
            'actual_direction': actual_direction,
            'alignment': alignment,
            'abs_impact': abs(impact['mean_impact'])
        })
    
    # Convert to DataFrame and sort by absolute impact for better visualization
    impact_df = pd.DataFrame(impact_data)
    impact_df = impact_df.sort_values('abs_impact', ascending=False)
    
    # Create color map based on alignment
    color_map = {
        'Positive Match': 'green',
        'Negative Match': 'blue',
        'Positive Mismatch': 'red',
        'Negative Mismatch': 'orange',
        'Neutral': 'gray'
    }
    
    # 1. Comprehensive Feature Impact Visualization
    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    # 1.1 Main impact bar chart
    ax1 = plt.subplot(gs[0, 0])
    bars = sns.barplot(x='mean_impact', y='feature', data=impact_df, 
                     palette=[color_map[a] for a in impact_df['alignment']], 
                     hue='alignment', dodge=False, ax=ax1)
    
    # Add error bars
    for i, row in impact_df.iterrows():
        ax1.errorbar(
            x=row['mean_impact'], 
            y=i, 
            xerr=row['std_impact'],
            color='black',
            capsize=3
        )
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.set_title(f'Feature Impact Magnitude for {country} - {tenor}', fontsize=14)
    ax1.set_xlabel('Impact on Yield (percentage points)')
    ax1.set_ylabel('Features')
    
    # 1.2 Alignment pie chart
    ax2 = plt.subplot(gs[0, 1])
    alignment_counts = impact_df['alignment'].value_counts()
    wedges, texts, autotexts = ax2.pie(
        alignment_counts,
        labels=None,
        autopct='%1.1f%%',
        colors=[color_map[a] for a in alignment_counts.index],
        startangle=90
    )
    ax2.set_title('Impact Alignment Distribution', fontsize=12)
    ax2.legend(wedges, alignment_counts.index, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 1.3 Absolute impact comparison
    ax3 = plt.subplot(gs[1, 0])
    sns.barplot(x='abs_impact', y='feature', data=impact_df.head(10), 
               palette=[color_map[a] for a in impact_df.head(10)['alignment']], 
               hue='alignment', dodge=False, ax=ax3)
    ax3.set_title('Top 10 Features by Absolute Impact', fontsize=12)
    ax3.set_xlabel('Absolute Impact Magnitude')
    ax3.set_ylabel('Features')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend([],[], frameon=False)  # Remove duplicate legend
    
    # 1.4 Expected vs. Actual direction table
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    alignment_summary = impact_df.groupby('alignment').size().reset_index()
    alignment_summary.columns = ['Alignment', 'Count']
    alignment_summary['Percentage'] = alignment_summary['Count'] / alignment_summary['Count'].sum() * 100
    alignment_text = "Alignment Summary:\n"
    for _, row in alignment_summary.iterrows():
        alignment_text += f"{row['Alignment']}: {row['Count']} ({row['Percentage']:.1f}%)\n"
    ax4.text(0, 0.5, alignment_text, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_comprehensive_impact.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create comparison chart for different shock levels if available
    if shock_results and len(shock_results) > 1:
        # Extract top 5 features by absolute impact
        top_features = impact_df.nlargest(5, 'abs_impact')['feature'].tolist()
        
        # Prepare data for multi-shock visualization
        shock_comparison_data = []
        for shock_amount, results in shock_results.items():
            for feature in top_features:
                if feature in results['feature_impacts']:
                    shock_comparison_data.append({
                        'feature': feature,
                        'shock_amount': f"{int(shock_amount*100)}%",
                        'mean_impact': results['feature_impacts'][feature]['mean_impact']
                    })
        
        shock_df = pd.DataFrame(shock_comparison_data)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(x='feature', y='mean_impact', hue='shock_amount', data=shock_df)
        plt.title(f'Impact Comparison Across Shock Levels for {country} - {tenor}')
        plt.xlabel('Features')
        plt.ylabel('Impact on Yield')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_file_prefix}_shock_comparison.png", dpi=300)
        plt.close()
    
    # 3. Create feature correlation heatmap if data available
    try:
        # This would need actual data - if available, compute correlation matrix
        # For now, just create a placeholder/example
        if os.path.exists('combined_features.csv'):
            x_ref = pd.read_csv('combined_features.csv', index_col=0)
            if set(impact_df['feature']) <= set(x_ref.columns):
                # Filter to just the features we need
                top_features = impact_df.nlargest(10, 'abs_impact')['feature'].tolist()
                x_subset = x_ref[top_features]
                
                # Calculate correlation matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(x_subset.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                plt.title(f'Feature Correlation Matrix for {country} - {tenor}')
                plt.tight_layout()
                plt.savefig(f"{output_file_prefix}_correlation_heatmap.png", dpi=300)
                plt.close()
    except Exception as e:
        logger.warning(f"Could not create correlation heatmap: {e}")
    
    # 4. Save impact data to CSV
    impact_df.to_csv(f"{output_file_prefix}_feature_impacts.csv", index=False)
    return impact_df


def load_best_model(model_path: str) -> Dict[str, Any]:
    """
    Load a saved model package.
    
    Parameters:
        model_path: Path to the saved model package
        
    Returns:
        Model package containing model, scaler, and metadata or None if error
    """
    try:
        model_package = joblib.load(model_path)
        return model_package
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None


def generate_synthetic_features(feature_names: List[str], n_samples: int = 100) -> pd.DataFrame:
    """
    Generate synthetic feature data based on feature names.
    
    Parameters:
        feature_names: List of feature names
        n_samples: Number of synthetic samples to generate
    
    Returns:
        Synthetic feature dataset
    """
    synthetic_data = {}
    
    # Deterministic seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data for each feature
    for feature in feature_names:
        # Feature-specific generation strategies
        if 'rate' in feature.lower() or 'interest' in feature.lower():
            # Interest rates: Beta distribution scaled to 0-10%
            synthetic_data[feature] = np.random.beta(2, 5, n_samples) * 10
        elif 'inflation' in feature.lower() or 'cpi' in feature.lower():
            # Inflation: Right-skewed normal around 2-3%
            synthetic_data[feature] = np.abs(np.random.normal(2.5, 1, n_samples))
        elif 'gdp' in feature.lower() or 'growth' in feature.lower():
            # GDP/Growth: Normal distribution around 2-3%
            synthetic_data[feature] = np.random.normal(2.5, 1, n_samples)
        elif 'unemployment' in feature.lower():
            # Unemployment: Right-skewed distribution
            synthetic_data[feature] = np.abs(np.random.normal(5, 2, n_samples))
        elif 'debt' in feature.lower() or 'deficit' in feature.lower():
            # Debt/Deficit: Normal distribution around 0
            synthetic_data[feature] = np.random.normal(0, 5, n_samples)
        elif 'risk' in feature.lower():
            # Risk ratings: Constrained normal distribution
            synthetic_data[feature] = np.clip(np.random.normal(50, 10, n_samples), 0, 100)
        else:
            # Default: Standard normal distribution
            synthetic_data[feature] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(synthetic_data)


def extract_feature_names_from_model(model_package: Dict[str, Any]) -> List[str]:
    """
    Comprehensively extract feature names from a model package.
    
    Parameters:
        model_package: Loaded model package
    
    Returns:
        Extracted feature names
    """
    # Multiple strategies to extract feature names
    extraction_strategies = [
        # Strategy 1: Directly from model package
        lambda: model_package.get('feature_names', []),
        
        # Strategy 2: From model attributes (for sklearn models)
        lambda: (
            getattr(model_package['model'], 'feature_names_in_', []) 
            if hasattr(model_package['model'], 'feature_names_in_') else []
        ),
        
        # Strategy 3: From scaler (if available)
        lambda: (
            getattr(model_package['scaler'], 'feature_names_in_', []) 
            if hasattr(model_package['scaler'], 'feature_names_in_') else []
        )
    ]
    
    # Try each strategy
    for strategy in extraction_strategies:
        feature_names = strategy()
        if feature_names:
            logger.info(f"Extracted {len(feature_names)} feature names")
            return feature_names
    
    # Fallback
    logger.warning("Could not extract feature names. Using generic naming.")
    return [f'feature_{i}' for i in range(model_package['scaler'].n_features_in_)]


def run_sensitivity_analysis(model_directory: str, output_base_dir: str) -> Dict[str, Any]:
    """
    Run comprehensive sensitivity analysis across all models.
    
    Parameters:
        model_directory: Directory containing model files
        output_base_dir: Base directory for saving outputs
        
    Returns:
        Consolidated results from sensitivity analysis
    """
    # Collect all model files
    all_models = []
    for root, _, files in os.walk(model_directory):
        for file in files:
            if file.endswith('_best_model.pkl'):
                all_models.append(os.path.join(root, file))
    
    # Consolidated results storage
    consolidated_results = {
        'total_models': len(all_models),
        'analyzed_models': 0,
        'feature_impact_summary': {}
    }
    
    # Process each model
    for model_path in all_models:
        try:
            # Extract country and tenor from filename
            filename = os.path.basename(model_path)
            match = re.search(r'(.+)_(\w+)_best_model\.pkl', filename)
            if not match:
                logger.warning(f"Could not parse filename: {filename}")
                continue
            
            country, tenor = match.groups()
            
            # Create output directory for this model
            output_dir = os.path.join(output_base_dir, country, tenor)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load model
            model_package = joblib.load(model_path)
            
            # Extract feature names
            feature_names = extract_feature_names_from_model(model_package)
            
            # Perform sensitivity analysis
            sensitivity_results = perform_feature_sensitivity_analysis(
                model_package, 
                feature_names
            )
            
            # Create visualizations
            create_comprehensive_visualizations(
                sensitivity_results, 
                country, 
                tenor, 
                output_dir
            )

            # Update consolidated results
            consolidated_results['analyzed_models'] += 1
            consolidated_results['feature_impact_summary'][f'{country}_{tenor}'] = {
                'feature_impacts': sensitivity_results['feature_impacts'],
                'top_features': sorted(
                    sensitivity_results['feature_impacts'].items(), 
                    key=lambda x: abs(x[1]['mean_impact']), 
                    reverse=True
                )[:5]
            }
            
            logger.info(f"Completed analysis for {country} - {tenor}")
        
        except Exception as e:
            logger.error(f"Error processing {model_path}: {e}")
            logger.error(traceback.format_exc())

    all_feature_impacts = prepare_feature_impacts(consolidated_results)
    
    # Create expected impact charts
    create_expected_impact_charts(
        all_feature_impacts, 
        output_base_dir
    )
    
    # Create overall summary
    summary_path = os.path.join(output_base_dir, 'sensitivity_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Sensitivity Analysis Summary\n")
        f.write("==========================\n\n")
        f.write(f"Total Models: {consolidated_results['total_models']}\n")
        f.write(f"Analyzed Models: {consolidated_results['analyzed_models']}\n\n")
        
        f.write("Top Features by Absolute Impact\n")
        for model, results in consolidated_results['feature_impact_summary'].items():
            f.write(f"\n{model}:\n")
            for feature, impact in results['top_features']:
                f.write(f"  {feature}: {impact['mean_impact']:.4f} (±{impact['std_impact']:.4f})\n")
    
    logger.info(f"Sensitivity analysis complete. Summary at {summary_path}")
    
    return consolidated_results


def create_expected_impact_charts(all_feature_impacts: List[Dict], 
                                output_dir: str) -> pd.DataFrame:
    """
    Create comprehensive expected impact charts from feature impacts.
    
    Parameters:
        all_feature_impacts: List of dictionaries containing feature impact data
        output_dir: Directory to save visualization outputs
    
    Returns:
        Processed feature impacts with conformity information
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    impacts_df = pd.DataFrame(all_feature_impacts)
    
    def determine_expected_direction_comprehensive(feature: str) -> int:
        """
        Comprehensively classify feature direction for bond yield models.
        Ensures every feature is explicitly mapped to a direction.
        """
        # Positive impact indicators
        positive_indicators = [
            'pred_yld', # Predicted yields
            'rating', # Risk ratings
            'cpi', # Inflation indicators
            'forecast', # Economic forecasts
            'gdp', # GDP-related metrics
            'activity', # Economic activity
            'risk', # Risk-related metrics
            'historical', # Historical data
            'pol_rat' # Policy rates
        ]

        # Negative impact indicators
        negative_indicators = [
            'unemployment', # Unemployment metrics
            'u_rat', # Unemployment rates
            'deficit', # Fiscal deficits
            'debt' # Debt metrics
        ]

        # Convert feature to lowercase for robust matching
        lower_feature = feature.lower()

        # Check for positive indicators first
        for indicator in positive_indicators:
            if indicator in lower_feature:
                return 1

        # Then check for negative indicators
        for indicator in negative_indicators:
            if indicator in lower_feature:
                return -1

        # If no match found, log warning and default to positive
        logger.warning(f"Unclassified feature defaulted to positive: {feature}")
        return 1
    
    # Add expected direction to DataFrame
    impacts_df['expected_direction'] = impacts_df['feature'].apply(determine_expected_direction_comprehensive)
    
    def check_conformity(row) -> str:
        """
        Determine feature impact conformity to economic expectations.
        
        Returns:
            Conformity classification
        """
        # Positive expectation match
        if row['expected_direction'] > 0 and row['mean_impact'] > 0:
            return 'Positive'
        
        # Negative expectation match
        elif row['expected_direction'] < 0 and row['mean_impact'] < 0:
            return 'Negative'
        
        # Neutral cases
        elif row['expected_direction'] == 0:
            return 'Neutral'
        
        # Misaligned
        else:
            return 'Misaligned'
    
    # Add conformity classification
    impacts_df['conformity'] = impacts_df.apply(check_conformity, axis=1)
    
    # Extract country and feature type
    impacts_df['country'] = impacts_df['country_tenor'].apply(lambda x: x.split('_')[0])
    impacts_df['feature_type'] = impacts_df['feature'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    # 0. Overall Conformity Pie Chart
    plt.figure(figsize=(10, 8))
    conformity_counts = impacts_df['conformity'].value_counts()
    
    # Color mapping
    color_map = {
        'Positive': '#2ecc71',     # Green
        'Negative': '#2ecc71',     # Green
        'Neutral': '#3498db',      # Blue
        'Misaligned': '#e74c3c'    # Red
    }
    
    colors = [color_map.get(conf, '#95a5a6') for conf in conformity_counts.index]
    
    plt.pie(
        conformity_counts, 
        labels=conformity_counts.index, 
        autopct='%1.1f%%',
        colors=colors
    )
    plt.title('Overall Model Conformity Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_conformity_pie.png'))
    plt.close()
    
    # 1. Country-level Conformity Chart
    plt.figure(figsize=(12, 8))
    
    # Compute country-level conformity
    country_conformity = impacts_df.groupby('country').apply(
        lambda x: (x['conformity'].isin(['Positive', 'Negative'])).mean() * 100
    ).sort_values(ascending=True)
    
    plt.barh(country_conformity.index, country_conformity.values)
    plt.title('Model Conformity to Expected Relationships by Country')
    plt.xlabel('Percentage of Variables Matching Expectations')
    plt.ylabel('Country')
    plt.axvline(x=66.0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'country_tenor_conformity.png'))
    plt.close()
    
    # Optional: Save detailed results
    conformity_summary = {
        'overall_conformity': conformity_counts.to_dict(),
        'country_conformity': country_conformity.to_dict(),
        'variable_conformity': variable_conformity.to_dict(),
        'country_tenor_conformity': country_tenor_conformity.to_dict()
    }
    
    with open(os.path.join(output_dir, 'conformity_summary.json'), 'w') as f:
        json.dump(conformity_summary, f, indent=2)
    
    return impacts_df


def modify_prepare_data(original_func):
    """
    Decorator to add diagnostic logging to prepare_data function.
    
    Parameters:
        original_func: Original prepare_data function
    
    Returns:
        Modified prepare_data function with diagnostic logging
    """
    def wrapper(*args, **kwargs):
        logger.info("DIAGNOSTIC: prepare_data CALLED")
        logger.info(f"Positional args: {args}")
        logger.info(f"Keyword args: {kwargs}")
        
        # Call original function
        x, y, feature_details = original_func(*args, **kwargs)
        
        # Add extensive logging
        if x is not None:
            logger.info("DIAGNOSTIC: Feature Preparation Success")
            logger.info(f"Feature matrix shape: {x.shape}")
            
            # ENSURE feature names are always captured
            feature_details['feature_columns'] = list(x.columns)
            logger.info(f"Captured feature columns: {feature_details['feature_columns']}")
        else:
            logger.critical("CRITICAL: Feature matrix is None")
        
        return x, y, feature_details
    
    return wrapper


def diagnostic_model_saving(save_func):
    """
    Decorator to add diagnostic logging to model saving function.
    
    Parameters:
        save_func: Original model saving function
    
    Returns:
        Modified model saving function with diagnostic logging
    """
    def wrapper(results_summary):
        logger.info("DIAGNOSTIC: Model Saving Process Started")
        logger.info(f"Total countries: {len(results_summary.get('best_models', {}))}")
        
        try:
            # Call original saving function with enhanced error tracking
            result = save_func(results_summary)
            
            logger.info("Model saving process completed successfully")
            return result
        
        except Exception as e:
            logger.critical(f"FATAL ERROR in model saving: {e}")
            logger.critical(f"Results summary keys: {list(results_summary.keys())}")
            logger.critical(traceback.format_exc())
            
            raise
    return wrapper


def configure_diagnostic_logging():
    """
    Configure logging for maximum diagnostic information.
    """
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('feature_extraction_diagnostic.log')
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set logging for specific modules
    logging.getLogger('prepare_data').setLevel(logging.DEBUG)
    logging.getLogger('data_worker').setLevel(logging.DEBUG)


def extract_feature_names(model_result: Dict[str, Any], country: str, tenor: str) -> List[str]:
    """
    Comprehensive feature name extraction with extensive logging.
    
    Parameters:
        model_result: Model result dictionary
        country: Country name
        tenor: Tenor type
    
    Returns:
        Extracted feature names
    """
    logger.info(f"DIAGNOSTIC: Extracting feature names for {country} - {tenor}")
    logger.info(f"Model result keys: {list(model_result.keys())}")
    
    # Multiple extraction strategies
    extraction_strategies = [
        # Strategy 1: Check feature_details first
        lambda: (
            model_result.get('feature_details', {}).get('feature_columns') or
            model_result.get('feature_details', {}).get('feature_names')
        ),
        
        # Strategy 2: Check data_summary
        lambda: (
            model_result.get('data_summary', {}).get('feature_columns') or
            model_result.get('data_summary', {}).get('feature_names')
        ),
        
        # Strategy 3: Direct model result keys
        lambda: (
            model_result.get('feature_columns') or
            model_result.get('feature_names')
        ),
        
        # Strategy 4: Check nested dictionaries
        lambda: (
            model_result.get('feature_details', {}).get('combined_features', {}).get('feature_columns') or
            model_result.get('data_summary', {}).get('combined_features', {}).get('feature_columns')
        )
    ]
    
    # Try each strategy
    for i, strategy in enumerate(extraction_strategies, 1):
        feature_names = strategy()
        
        if feature_names:
            # Ensure we have a list
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            
            logger.info(f"Strategy {i} successful: Found {len(feature_names)} feature names")
            logger.info(f"Feature names: {feature_names}")
            return feature_names
        
        logger.info(f"Strategy {i} failed to extract feature names")
    
    # Last resort: generic error with maximum debugging information
    logger.critical(f"FATAL: No feature names found for {country} - {tenor}")
    logger.critical(f"Full model result structure: {model_result}")
    
    raise ValueError(f"""
    Impossible to extract feature names for {country} - {tenor}
    
    Debugging Information:
    ---------------------
    Country: {country}
    Tenor: {tenor}
    Model Result Keys: {list(model_result.keys())}
    
    Possible solutions:
    1. Check data preparation process
    2. Verify feature name capture in train_evaluate_model
    3. Confirm feature details are being correctly populated
    """)


def perform_feature_sensitivity_analysis(model_package: Dict[str, Any], 
                                       feature_names: List[str],
                                       shock_amounts: List[float] = [0.25, 0.5, 1.0]) -> Dict[str, Any]:
    """
    Perform detailed sensitivity analysis on specific features.
    
    Parameters:
        model_package: Dictionary containing model, scaler, and metadata
        feature_names: List of feature names
        shock_amounts: List of shock amounts to test (default: [0.25, 0.5, 1.0])
        
    Returns:
        Dictionary containing sensitivity analysis results
    """
    model = model_package['model']
    scaler = model_package['scaler']
    
    # Generate synthetic reference data
    x_ref = generate_synthetic_features(feature_names)
    
    # Scale reference data
    x_scaled_ref = scaler.transform(x_ref)
    
    # Baseline predictions
    baseline_pred = model.predict(x_scaled_ref)
    
    # Results storage
    sensitivity_results = {
        'baseline_pred': baseline_pred,
        'feature_impacts': {},
        'shock_levels': {}
    }
    
    # Shock analysis
    for shock_level in shock_amounts:
        feature_level_impacts = {}
        
        for i, feature in enumerate(feature_names):
            # Create a copy of scaled reference data
            x_shocked = x_scaled_ref.copy()
            
            # Shock the feature
            x_shocked[:, i] += x_scaled_ref[:, i] * shock_level
            
            # Generate predictions
            shocked_pred = model.predict(x_shocked)
            
            # Calculate impact
            impact = shocked_pred - baseline_pred
            
            # Store results
            feature_level_impacts[feature] = {
                'mean_impact': np.mean(impact),
                'std_impact': np.std(impact),
                'max_impact': np.max(impact),
                'min_impact': np.min(impact)
            }
        
        # Store impacts for this shock level
        sensitivity_results['shock_levels'][shock_level] = feature_level_impacts
    
    # Aggregate impacts across shock levels
    for feature in feature_names:
        feature_impacts = [
            level_impacts[feature]['mean_impact'] 
            for level_impacts in sensitivity_results['shock_levels'].values()
        ]
        
        sensitivity_results['feature_impacts'][feature] = {
            'mean_impact': np.mean(feature_impacts),
            'std_impact': np.std(feature_impacts),
            'max_impact': np.max(feature_impacts),
            'min_impact': np.min(feature_impacts)
        }
    
    return sensitivity_results


def create_comprehensive_visualizations(sensitivity_results: Dict[str, Any], 
                                      country: str, tenor: str, output_dir: str) -> None:
    """
    Create a comprehensive set of visualizations for sensitivity analysis.
    
    Parameters:
        sensitivity_results: Dictionary containing sensitivity analysis results
        country: Country name
        tenor: Tenor name
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract feature impacts
    feature_impacts = sensitivity_results['feature_impacts']
    
    # Prepare data for visualization
    impact_data = []
    for feature, impact in feature_impacts.items():
        impact_data.append({
            'feature': feature,
            'mean_impact': impact['mean_impact'],
            'std_impact': impact['std_impact'],
            'abs_impact': abs(impact['mean_impact'])
        })
    
    # Convert to DataFrame and sort
    impact_df = pd.DataFrame(impact_data)
    impact_df = impact_df.sort_values('abs_impact', ascending=False)
    
    # 1. Feature Impact Bar Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='mean_impact', y='feature', data=impact_df, 
                palette='coolwarm', orient='h')
    plt.title(f'Feature Impact on {country} {tenor} Yield')
    plt.xlabel('Mean Impact')
    
    # 2. Impact Magnitude Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(impact_df['mean_impact'], kde=True)
    plt.title('Distribution of Feature Impacts')
    plt.xlabel('Impact Magnitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{country}_{tenor}_feature_impacts.png'))
    plt.close()
    
    # 3. Shock Level Comparison
    shock_levels = sensitivity_results['shock_levels']
    plt.figure(figsize=(15, 6))
    
    # Prepare data for comparison
    shock_comparison_data = []
    for shock_level, level_impacts in shock_levels.items():
        for feature, impact in level_impacts.items():
            shock_comparison_data.append({
                'feature': feature,
                'shock_level': shock_level,
                'mean_impact': impact['mean_impact']
            })
    
    shock_comp_df = pd.DataFrame(shock_comparison_data)
    
    # Pivot for easier plotting
    shock_pivot = shock_comp_df.pivot(
        index='feature', 
        columns='shock_level', 
        values='mean_impact'
    )
    
    # Select top 10 features by total absolute impact
    top_features = shock_pivot.abs().sum(axis=1).nlargest(10).index
    shock_pivot_top = shock_pivot.loc[top_features]
    
    sns.heatmap(shock_pivot_top, cmap='coolwarm', center=0, annot=True)
    plt.title(f'Feature Impact Across Shock Levels - {country} {tenor}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{country}_{tenor}_shock_comparison.png'))
    plt.close()
    
    # 4. Detailed Impact CSV
    impact_df.to_csv(os.path.join(output_dir, f'{country}_{tenor}_feature_impacts.csv'), index=False)


def main():
    """
    Main execution function for model testing and sensitivity analysis.
    """
    # Run analysis
    run_sensitivity_analysis(config.MODEL_DIR, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()(x=66.0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'country_conformity.png'))
    plt.close()
    
    # 2. Variable Type Conformity Chart
    plt.figure(figsize=(12, 8))
    
    # Compute variable type conformity
    variable_conformity = impacts_df.groupby('feature_type').apply(
        lambda x: (x['conformity'].isin(['Positive', 'Negative'])).mean() * 100
    ).sort_values(ascending=True)
    
    # Color coding based on conformity
    colors = ['green' if conf > 50 else 'red' for conf in variable_conformity]
    
    plt.barh(variable_conformity.index, variable_conformity.values, color=colors)
    plt.title('Model Conformity to Expected Relationships by Variable Type')
    plt.xlabel('Percentage of Variables Matching Expectations')
    plt.ylabel('Variable Type')
    plt.axvline(x=66.0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_type_conformity.png'))
    plt.close()
    
    # 3. Country-Tenor Conformity Chart
    plt.figure(figsize=(15, 10))
    
    # Compute country-tenor conformity
    country_tenor_conformity = impacts_df.groupby('country_tenor').apply(
        lambda x: (x['conformity'].isin(['Positive', 'Negative'])).mean() * 100
    ).sort_values(ascending=True)
    
    plt.barh(country_tenor_conformity.index, country_tenor_conformity.values)
    plt.title('Model Conformity to Expected Relationships by Country-Tenor')
    plt.xlabel('Percentage of Variables Matching Expectations')
    plt.ylabel('Country-Tenor')
    plt.axvline