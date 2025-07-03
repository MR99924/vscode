"""
Enhanced model testing and sensitivity analysis utilities for yield curve modeling.
This module provides comprehensive model testing, feature sensitivity analysis,
and economic theory conformity validation with improved robustness and accuracy.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import sys
from typing import Dict, Any, List, Tuple, Optional
import re
from matplotlib.gridspec import GridSpec
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from datetime import datetime, timedelta
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize logging
logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class EconomicRelationshipValidator:
    """
    Enhanced economic relationship validation with country and time-period awareness.
    """

    def __init__(self):
        self.base_relationships = self._initialize_base_relationships()
        self.country_adjustments = self._initialize_country_adjustments()

    def _initialize_base_relationships(self) -> Dict[str, Dict[str, float]]:
        """Initialize base economic relationships with confidence levels."""
        return {
            # Inflation-related features (positive correlation with yields)
            'cpi': {'direction': 1, 'confidence': 0.9, 'magnitude': 'high'},
            'inflation': {'direction': 1, 'confidence': 0.9, 'magnitude': 'high'},
            'cpi_forecast': {'direction': 1, 'confidence': 0.8, 'magnitude': 'high'},
            'cpi_target': {'direction': 1, 'confidence': 0.7, 'magnitude': 'medium'},
            # Policy rate features (positive correlation)
            'pol_rat': {'direction': 1, 'confidence': 0.8, 'magnitude': 'high'},
            'policy_rate': {'direction': 1, 'confidence': 0.8, 'magnitude': 'high'},
            # Economic activity (positive correlation)
            'gdp': {'direction': 1, 'confidence': 0.7, 'magnitude': 'medium'},
            'gdp_forecast': {'direction': 1, 'confidence': 0.6, 'magnitude': 'medium'},
            'act_track': {'direction': 1, 'confidence': 0.7, 'magnitude': 'medium'},
            # Risk/Rating features (negative correlation - higher ratings = lower yields)
            'rating': {'direction': -1, 'confidence': 0.8, 'magnitude': 'medium'},
            'consolidated_rating': {'direction': -1, 'confidence': 0.8, 'magnitude': 'medium'},
            # Unemployment (negative correlation)
            'unemployment': {'direction': -1, 'confidence': 0.7, 'magnitude': 'medium'},
            'u_rat': {'direction': -1, 'confidence': 0.7, 'magnitude': 'medium'},
            # International investment position (context-dependent)
            'iip_gdp': {'direction': 0, 'confidence': 0.4, 'magnitude': 'low'},
            # Historical forecasts (should be positive correlation)
            'historical_forecast': {'direction': 1, 'confidence': 0.6, 'magnitude': 'medium'},
            'pred_yld': {'direction': 1, 'confidence': 0.8, 'magnitude': 'high'}
        }

    def _initialize_country_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Initialize country-specific adjustments to base relationships."""
        return {
            # Emerging markets may have different sensitivity patterns
            'pl': {'risk_sensitivity': 1.2, 'inflation_sensitivity': 1.1},
            'hu': {'risk_sensitivity': 1.2, 'inflation_sensitivity': 1.1},
            'cz': {'risk_sensitivity': 1.2, 'inflation_sensitivity': 1.1},
            'za': {'risk_sensitivity': 1.3, 'inflation_sensitivity': 1.2},
            'kr': {'risk_sensitivity': 1.1, 'inflation_sensitivity': 1.0},
            # Developed markets baseline
            'us': {'risk_sensitivity': 1.0, 'inflation_sensitivity': 1.0},
            'gb': {'risk_sensitivity': 1.0, 'inflation_sensitivity': 1.0},
            'fr': {'risk_sensitivity': 0.9, 'inflation_sensitivity': 0.95},
            'de': {'risk_sensitivity': 0.9, 'inflation_sensitivity': 0.95},
            'it': {'risk_sensitivity': 1.1, 'inflation_sensitivity': 1.0},
            'ca': {'risk_sensitivity': 1.0, 'inflation_sensitivity': 1.0},
            'au': {'risk_sensitivity': 1.0, 'inflation_sensitivity': 1.0}
        }
    
    def get_expected_relationship(self, feature: str, country: str = None) -> Dict[str, Any]:
        """
        Get expected economic relationship for a feature with country adjustments.
        Parameters:
            feature: Feature name
            country: Country code for adjustments
        Returns:
            Dictionary with direction, confidence, and magnitude
        """
        # Clean feature name
        feature_clean = feature.lower().strip()

        # Find matching base relationship
        base_relationship = None
        for pattern, relationship in self.base_relationships.items():
            if pattern in feature_clean:
                base_relationship = relationship.copy()
                break

        # Default neutral relationship if no match
        if base_relationship is None:
            logger.warning(f"No economic relationship defined for feature: {feature}")
            return {'direction': 0, 'confidence': 0.3, 'magnitude': 'low'}

        # Apply country adjustments if available
        if country and country in self.country_adjustments:
            adjustments = self.country_adjustments[country]
            # Adjust based on feature type
            if any(term in feature_clean for term in ['rating', 'risk']):
                base_relationship['confidence'] *= adjustments.get('risk_sensitivity', 1.0)
            elif any(term in feature_clean for term in ['cpi', 'inflation']):
                base_relationship['confidence'] *= adjustments.get('inflation_sensitivity', 1.0)

        return base_relationship
    
class EnhancedDataGenerator:
    """
    Generate realistic economic data with proper correlations and temporal patterns.
    """

    def __init__(self, country_list: List[str]):
        self.country_list = country_list
        self.economic_ranges = self._initialize_economic_ranges()

    def _initialize_economic_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Initialize realistic economic data ranges by country group."""
        return {
            # Developed markets
            'developed': {
                'policy_rates': (0.0, 6.0),
                'inflation': (0.0, 4.0),
                'gdp_growth': (-3.0, 5.0),
                'unemployment': (3.0, 12.0),
                'bond_yields': (0.0, 8.0)
            },
            # Emerging markets
            'emerging': {
                'policy_rates': (0.5, 15.0),
                'inflation': (0.0, 8.0),
                'gdp_growth': (-5.0, 8.0),
                'unemployment': (3.0, 25.0),
                'bond_yields': (1.0, 12.0)
            }
        }

    def _get_country_group(self, country: str) -> str:
        """Determine if country is developed or emerging market."""
        if country in config.developed_markets:
            return 'developed'
        elif country in config.emerging_markets:
            return 'emerging'
        else:
            return 'developed'  # Default

    def generate_correlated_features(self, feature_names: List[str],
                                     n_samples: int = 1000,
                                     country: str = None,
                                     random_seed: int = None) -> pd.DataFrame:
        """
        Generate realistic economic data with proper correlations.
        Parameters:
            feature_names: List of feature names
            n_samples: Number of samples to generate
            country: Country for realistic ranges
            random_seed: Random seed for reproducibility (None for random)
        Returns:
            DataFrame with correlated economic features
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        country_group = self._get_country_group(country) if country else 'developed'
        ranges = self.economic_ranges[country_group]
        synthetic_data = {}

        # Generate base economic factors first
        base_factors = self._generate_base_factors(n_samples, ranges)

        # Generate features based on patterns and correlations
        for feature in feature_names:
            feature_clean = feature.lower()
            if any(term in feature_clean for term in ['pol_rat', 'policy', 'rate']):
                synthetic_data[feature] = self._add_noise(base_factors['policy_rate'], noise_level=0.2)
            elif any(term in feature_clean for term in ['cpi', 'inflation']):
                synthetic_data[feature] = self._add_noise(base_factors['inflation'], noise_level=0.3)
            elif any(term in feature_clean for term in ['gdp', 'growth']):
                synthetic_data[feature] = self._add_noise(base_factors['gdp_growth'], noise_level=0.4)
            elif any(term in feature_clean for term in ['unemployment', 'u_rat']):
                synthetic_data[feature] = self._add_noise(base_factors['unemployment'], noise_level=0.3)
            elif any(term in feature_clean for term in ['yld', 'yield']):
                yield_base = (
                    0.6 * base_factors['policy_rate'] +
                    0.4 * base_factors['inflation'] +
                    0.2 * base_factors['gdp_growth']
                )
                synthetic_data[feature] = self._add_noise(yield_base, noise_level=0.2)
            elif any(term in feature_clean for term in ['rating', 'risk']):
                risk_base = 100 - (base_factors['policy_rate'] + base_factors['inflation']) * 5
                synthetic_data[feature] = np.clip(
                    self._add_noise(risk_base, noise_level=0.1), 0, 100
                )
            elif any(term in feature_clean for term in ['act_track', 'activity']):
                synthetic_data[feature] = self._add_noise(
                    base_factors['gdp_growth'] * 20 + 50, noise_level=0.3
                )
            else:
                synthetic_data[feature] = np.random.normal(2.0, 1.0, n_samples)

        return pd.DataFrame(synthetic_data)
    
    def _generate_persistent_series(self, n_samples: int, min_val: float, max_val: float, persistence: float = 0.9) -> np.ndarray:
        series = np.zeros(n_samples)
        series[0] = np.random.uniform(min_val, max_val)
        for i in range(1, n_samples):
            innovation = np.random.normal(0, 0.1)
            series[i] = persistence * series[i - 1] + innovation
            series[i] = np.clip(series[i], min_val, max_val)
        return series

    def _generate_cyclical_series(self, n_samples: int, min_val: float, max_val: float, cycle_length: int = 40) -> np.ndarray:
        t = np.arange(n_samples)
        cyclical = np.sin(2 * np.pi * t / cycle_length)
        noise = np.random.normal(0, 0.3, n_samples)
        series = cyclical + noise
        series = (series - series.min()) / (series.max() - series.min())
        return series * (max_val - min_val) + min_val

    def _generate_lagged_inverse(self, base_series: np.ndarray, min_val: float, max_val: float, lag: int = 2) -> np.ndarray:
        lagged = np.roll(base_series, lag)
        lagged[:lag] = lagged[lag]
        inverted = -lagged
        inverted = (inverted - inverted.min()) / (inverted.max() - inverted.min())
        return inverted * (max_val - min_val) + min_val

    def _add_noise(self, base_series: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        noise = np.random.normal(0, noise_level * np.std(base_series), len(base_series))
        return base_series + noise

    def _generate_base_factors(self, n_samples: int, ranges: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """Generate base economic factors with realistic distributions."""
        factors = {}

        # Policy rate: tends to be persistent, occasional jumps
        policy_min, policy_max = ranges['policy_rates']
        factors['policy_rate'] = self._generate_persistent_series(
            n_samples, policy_min, policy_max, persistence=0.95
        )

        # Inflation: somewhat persistent, more volatile than policy rates
        inf_min, inf_max = ranges['inflation']
        factors['inflation'] = self._generate_persistent_series(
            n_samples, inf_min, inf_max, persistence=0.90
        )

        # GDP growth: cyclical with some persistence
        gdp_min, gdp_max = ranges['gdp_growth']
        factors['gdp_growth'] = self._generate_cyclical_series(
            n_samples, gdp_min, gdp_max, cycle_length=40
        )

        # Unemployment: lagged inverse of GDP growth
        unemp_min, unemp_max = ranges['unemployment']
        factors['unemployment'] = self._generate_lagged_inverse(
            factors['gdp_growth'], unemp_min, unemp_max, lag=2
        )

        return factors
    
class StatisticalTester:
    """
    Statistical significance testing for feature impacts.
    """

    @staticmethod
    def test_impact_significance(baseline_pred: np.ndarray,
                                  shocked_pred: np.ndarray,
                                  significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test statistical significance of feature impact.
        Parameters:
            baseline_pred: Baseline predictions
            shocked_pred: Predictions after feature shock
            significance_level: Statistical significance threshold
        Returns:
            Dictionary with test results
        """
        differences = shocked_pred - baseline_pred
        t_statistic, p_value = stats.ttest_1samp(differences, 0)
        effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
        return {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': p_value < significance_level,
            'effect_size': effect_size,
            'confidence_interval': stats.t.interval(
                1 - significance_level, len(differences) - 1,
                loc=np.mean(differences),
                scale=stats.sem(differences)
            )
        }

class EnhancedSensitivityAnalyzer:
    """
    Enhanced sensitivity analysis with statistical testing and economic validation.
    """

    def __init__(self):
        self.validator = EconomicRelationshipValidator()
        self.data_generator = EnhancedDataGenerator(config.country_list)
        self.statistical_tester = StatisticalTester()

    def analyze_model_sensitivity(self, model_package: Dict[str, Any],
                                  feature_names: List[str],
                                  country: str,
                                  tenor: str,
                                  shock_scenarios: Dict[str, List[float]] = None,
                                  n_samples: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive sensitivity analysis with statistical testing.
        Parameters:
            model_package: Dictionary containing model, scaler, and metadata
            feature_names: List of feature names
            country: Country code
            tenor: Bond tenor
            shock_scenarios: Custom shock scenarios
            n_samples: Number of synthetic samples
        Returns:
            Comprehensive sensitivity analysis results
        """
        if shock_scenarios is None:
            shock_scenarios = {
                'small': [0.1, 0.25],
                'medium': [0.5, 0.75],
                'large': [1.0, 1.5]
            }

        model = model_package['model']
        scaler = model_package['scaler']

        # Generate realistic synthetic data
        x_ref = self.data_generator.generate_correlated_features(
            feature_names, n_samples=n_samples, country=country, random_seed=42
        )

        # Scale reference data
        x_scaled_ref = scaler.transform(x_ref)

        # Baseline predictions
        baseline_pred = model.predict(x_scaled_ref)

        # Results storage
        results = {
            'country': country,
            'tenor': tenor,
            'n_samples': n_samples,
            'baseline_statistics': {
                'mean': np.mean(baseline_pred),
                'std': np.std(baseline_pred),
                'min': np.min(baseline_pred),
                'max': np.max(baseline_pred)
            },
            'feature_impacts': {},
            'scenario_analysis': {},
            'economic_conformity': {}
        }

        # Analyze each feature
        for i, feature in enumerate(feature_names):
            feature_results = self._analyze_feature_impact(
                model, scaler, x_scaled_ref, baseline_pred,
                feature, i, shock_scenarios, country
            )
            results['feature_impacts'][feature] = feature_results

        # Perform interaction analysis on top features
        top_features = sorted(
            results['feature_impacts'].items(),
            key=lambda x: abs(x[1]['mean_impact']),
            reverse=True
        )[:5]

        results['interaction_analysis'] = self._analyze_feature_interactions(
            model, scaler, x_scaled_ref, baseline_pred,
            [f[0] for f in top_features], feature_names
        )

        # Economic conformity assessment
        results['economic_conformity'] = self._assess_economic_conformity(
            results['feature_impacts'], country
        )

        return results

    def _analyze_feature_impact(self, model, scaler, x_scaled_ref: np.ndarray,
                                baseline_pred: np.ndarray, feature: str,
                                feature_idx: int, shock_scenarios: Dict[str, List[float]],
                                country: str) -> Dict[str, Any]:
        """Analyze impact of a single feature with statistical testing."""
        feature_results = {
            'feature_index': feature_idx,
            'shock_results': {},
            'statistical_tests': {},
            'expected_relationship': self.validator.get_expected_relationship(feature, country)
        }

        all_impacts = []

        for scenario_name, shock_levels in shock_scenarios.items():
            scenario_impacts = []
            for shock in shock_levels:
                for direction in [1, -1]:
                    x_shocked = x_scaled_ref.copy()
                    x_shocked[:, feature_idx] += direction * shock * np.std(x_scaled_ref[:, feature_idx])
                    shocked_pred = model.predict(x_shocked)
                    impact = shocked_pred - baseline_pred

                    stat_test = self.statistical_tester.test_impact_significance(
                        baseline_pred, shocked_pred
                    )

                    scenario_impacts.extend(impact)
                    shock_key = f"{scenario_name}_{shock}_{'+' if direction > 0 else '-'}"
                    feature_results['shock_results'][shock_key] = {
                        'mean_impact': np.mean(impact),
                        'std_impact': np.std(impact),
                        'statistical_test': stat_test
                    }

            all_impacts.extend(scenario_impacts)

        feature_results.update({
            'mean_impact': np.mean(all_impacts),
            'std_impact': np.std(all_impacts),
            'min_impact': np.min(all_impacts),
            'max_impact': np.max(all_impacts),
            'impact_range': np.max(all_impacts) - np.min(all_impacts),
            'abs_mean_impact': np.mean(np.abs(all_impacts))
        })

        return feature_results

    def _analyze_feature_interactions(self, model, scaler, x_scaled_ref: np.ndarray,
                                      baseline_pred: np.ndarray, top_features: List[str],
                                      all_features: List[str]) -> Dict[str, Any]:
        """Analyze interactions between top features."""
        interactions = {}
        feature_indices = {feat: all_features.index(feat) for feat in top_features}

        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i + 1:]:
                x_shocked = x_scaled_ref.copy()
                idx1, idx2 = feature_indices[feat1], feature_indices[feat2]
                shock_size = 0.5
                x_shocked[:, idx1] += shock_size * np.std(x_scaled_ref[:, idx1])
                x_shocked[:, idx2] += shock_size * np.std(x_scaled_ref[:, idx2])
                shocked_pred = model.predict(x_shocked)
                interaction_impact = shocked_pred - baseline_pred
                interactions[f"{feat1}_{feat2}"] = {
                    'mean_impact': np.mean(interaction_impact),
                    'std_impact': np.std(interaction_impact)
                }

        return interactions

    def _assess_economic_conformity(self, feature_impacts: Dict[str, Any],
                                    country: str) -> Dict[str, Any]:
        """Assess how well model behavior conforms to economic theory."""
        conformity_assessment = {
            'total_features': len(feature_impacts),
            'conforming_features': 0,
            'non_conforming_features': 0,
            'neutral_features': 0,
            'detailed_assessment': {}
        }

        for feature, impact_data in feature_impacts.items():
            expected = impact_data['expected_relationship']
            actual_direction = np.sign(impact_data['mean_impact'])
            expected_direction = expected['direction']

            if expected_direction == 0:
                conformity = 'neutral'
                conformity_assessment['neutral_features'] += 1
            elif (expected_direction > 0 and actual_direction > 0) or \
                 (expected_direction < 0 and actual_direction < 0):
                conformity = 'conforming'
                conformity_assessment['conforming_features'] += 1
            else:
                conformity = 'non_conforming'
                conformity_assessment['non_conforming_features'] += 1

            conformity_assessment['detailed_assessment'][feature] = {
                'conformity': conformity,
                'expected_direction': expected_direction,
                'actual_direction': actual_direction,
                'impact_magnitude': abs(impact_data['mean_impact']),
                'confidence': expected['confidence']
            }

        total_definite = (conformity_assessment['conforming_features'] +
                          conformity_assessment['non_conforming_features'])

        if total_definite > 0:
            conformity_assessment['conformity_score'] = (
                conformity_assessment['conforming_features'] / total_definite
            )
        else:
            conformity_assessment['conformity_score'] = 0.0
        return conformity_assessment

class ModelTester:
    """
    Main model testing class with enhanced capabilities.
    """

    def __init__(self):
        self.sensitivity_analyzer = EnhancedSensitivityAnalyzer()
        self.output_dir = config.OUTPUT_DIR

    def load_model_safely(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Safely load a model with comprehensive error handling and validation.
        Parameters:
            model_path: Path to the saved model package
        Returns:
            Model package or None if loading fails
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file does not exist: {model_path}")
                return None

            model_package = joblib.load(model_path)

            # Validate model package structure
            required_keys = ['model', 'scaler']
            missing_keys = [key for key in required_keys if key not in model_package]
            if missing_keys:
                logger.error(f"Model package missing required keys: {missing_keys}")
                return None

            # Validate model has predict method
            if not hasattr(model_package['model'], 'predict'):
                logger.error("Model does not have predict method")
                return None

            # Validate scaler
            if not hasattr(model_package['scaler'], 'transform'):
                logger.error("Scaler does not have transform method")
                return None

            return model_package

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            logger.error(traceback.format_exc())
            return None

    def extract_feature_names_robust(self, model_package: Dict[str, Any],
                                     country: str, tenor: str) -> List[str]:
        """
        Robust feature name extraction with multiple fallback strategies.
        Parameters:
            model_package: Loaded model package
            country: Country name
            tenor: Tenor type
        Returns:
            List of feature names
        """
        extraction_strategies = [
            lambda: model_package.get('feature_names'),
            lambda: model_package.get('feature_columns'),
            lambda: model_package.get('metadata', {}).get('feature_names'),
            lambda: model_package.get('feature_details', {}).get('feature_names'),
            lambda: model_package.get('data_summary', {}).get('feature_names'),
            lambda: getattr(model_package['model'], 'feature_names_in_', None),
            lambda: getattr(model_package['scaler'], 'feature_names_in_', None),
            lambda: self._reconstruct_feature_names(country, tenor)
        ]

        for i, strategy in enumerate(extraction_strategies, 1):
            try:
                feature_names = strategy()
                if feature_names and len(feature_names) > 0:
                    if not isinstance(feature_names, list):
                        feature_names = list(feature_names)
                    logger.info(f"Feature extraction strategy {i} successful: {len(feature_names)} features")
                    return feature_names
            except Exception as e:
                logger.debug(f"Feature extraction strategy {i} failed: {e}")

        n_features = getattr(model_package['scaler'], 'n_features_in_', 10)
        logger.warning(f"Using generic feature names for {country}_{tenor}")
        return [f'feature_{i}' for i in range(n_features)]

    def _reconstruct_feature_names(self, country: str, tenor: str) -> List[str]:
        """Reconstruct likely feature names from config."""
        country_code = config.country_list_mapping.get(country, country.lower())
        tenor_features = config.TENOR_FEATURES.get(f'yld_{tenor}', [])
        feature_names = []

        for feature_type in tenor_features:
            if feature_type == 'policy_rates':
                feature_names.append(f'pol_rat_{country_code}')
            elif feature_type == 'inflation':
                feature_names.extend([f'cpi_inf_{country_code}', f'cpi_target_{country_code}'])
            elif feature_type == 'act_track':
                feature_names.append(f'act_track_{country_code}')
            elif feature_type == 'consolidated_ratings':
                feature_names.append(f'rating_{country_code}')
            elif feature_type == 'iip_gdp':
                feature_names.append(f'iip_gdp_{country_code}')
            elif feature_type == 'unemployment_rate':
                feature_names.append(f'u_rat_{country_code}')
            elif 'historical_forecasts' in feature_type:
                feature_names.append(f'pred_yld_{tenor}_{country_code}')

        return feature_names if feature_names else [f'feature_{i}' for i in range(10)]

    def run_comprehensive_analysis(self, model_directory: str,
                                   output_directory: str = None,
                                   max_workers: int = 4) -> Dict[str, Any]:
        """
        Run comprehensive analysis on all models with parallel processing.
        Parameters:
            model_directory: Directory containing model files
            output_directory: Output directory (uses config default if None)
            max_workers: Maximum number of parallel workers
        Returns:
            Consolidated analysis results
        """
        if output_directory is None:
            output_directory = os.path.join(self.output_dir, 'model_analysis')
        os.makedirs(output_directory, exist_ok=True)

        model_files = self._discover_model_files(model_directory)
        if not model_files:
            logger.error(f"No model files found in {model_directory}")
            return {}

        logger.info(f"Found {len(model_files)} model files for analysis")

        consolidated_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_models': len(model_files),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'model_results': {},
            'summary_statistics': {},
            'conformity_overview': {}
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(
                    self._analyze_single_model,
                    model_file,
                    output_directory
                ): model_file for model_file in model_files
            }

            for future in as_completed(future_to_model):
                model_file = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        model_key = result['model_key']
                        consolidated_results['model_results'][model_key] = result
                        consolidated_results['successful_analyses'] += 1
                        logger.info(f"Successfully analyzed {model_key}")
                    else:
                        consolidated_results['failed_analyses'] += 1
                        logger.warning(f"Failed to analyze {model_file}")
                except Exception as e:
                    consolidated_results['failed_analyses'] += 1
                    logger.error(f"Error analyzing {model_file}: {e}")

        self._generate_consolidated_reports(consolidated_results, output_directory)
        return consolidated_results

    def _discover_model_files(self, model_directory: str) -> List[str]:
        """Discover all model files in directory."""
        model_files = []
        patterns = ['*_best_model.pkl', '*_model.pkl', '*.pkl']
        for pattern in patterns:
            for root, _, files in os.walk(model_directory):
                for file in files:
                    if file.endswith('.pkl') and ('model' in file.lower()):
                        model_files.append(os.path.join(root, file))
        return list(set(model_files))  # Remove duplicates

    def _analyze_single_model(self, model_path: str, output_directory: str) -> Optional[Dict[str, Any]]:
        """Analyze a single model file."""
        try:
            filename = os.path.basename(model_path)
            country, tenor = self._parse_model_filename(filename)
            if not country or not tenor:
                logger.warning(f"Could not parse filename: {filename}")
                return None

            model_package = self.load_model_safely(model_path)
            if not model_package:
                return None

            feature_names = self.extract_feature_names_robust(model_package, country, tenor)
            model_output_dir = os.path.join(output_directory, country, tenor)
            os.makedirs(model_output_dir, exist_ok=True)

            sensitivity_results = self.sensitivity_analyzer.analyze_model_sensitivity(
                model_package, feature_names, country, tenor
            )

            self._create_model_visualizations(sensitivity_results, model_output_dir)
            
            try:
                results_file = os.path.join(model_output_dir, 'sensitivity_analysis.json')
                with open(results_file, 'w') as f:
                    json_results = self._convert_for_json(sensitivity_results)
                    json.dump(json_results, f, indent=2)
            
            except Exception as e:
                logger.warning(f"Could not save JSON results: {e}")

            return {
                'model_key': f"{country}_{tenor}",
                'country': country,
                'tenor': tenor,
                'model_path': model_path,
                'feature_count': len(feature_names),
                'sensitivity_results': sensitivity_results,
                'output_directory': model_output_dir
            }

        except Exception as e:
            logger.error(f"Error analyzing model {model_path}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _parse_model_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            r'(.+?)_yld_(\d+yr)_.*\.pkl',  # Matches 'Country Name_yld_10yr_best_model.pkl'
        ]
        for pattern in patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                country, tenor = match.groups()
                tenor = tenor.lower()
                return country.strip(), tenor
        return None, None

    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            return obj.item()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif obj is None:
            return None
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)

    def _create_model_visualizations(self, sensitivity_results: Dict[str, Any],
                                     output_dir: str) -> None:
        """Create comprehensive visualizations for a single model."""
        country = sensitivity_results['country']
        tenor = sensitivity_results['tenor']
        feature_impacts = sensitivity_results['feature_impacts']

        impact_data = []
        for feature, impact in feature_impacts.items():
            expected = impact['expected_relationship']
            actual_direction = np.sign(impact['mean_impact'])
            conformity = 'Neutral'
            if expected['direction'] != 0:
                if (expected['direction'] > 0 and actual_direction > 0) or \
                   (expected['direction'] < 0 and actual_direction < 0):
                    conformity = 'Conforming'
                else:
                    conformity = 'Non-Conforming'
            impact_data.append({
                'feature': feature,
                'mean_impact': impact['mean_impact'],
                'std_impact': impact['std_impact'],
                'abs_impact': abs(impact['mean_impact']),
                'expected_direction': expected['direction'],
                'actual_direction': actual_direction,
                'conformity': conformity,
                'confidence': expected['confidence']
            })

        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values('abs_impact', ascending=False)

        # 1. Main Feature Impact Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1.1 Feature Impact Bar Chart
        color_map = {'Conforming': 'green', 'Non-Conforming': 'red', 'Neutral': 'gray'}
        colors = [color_map[conf] for conf in impact_df['conformity']]
        axes[0, 0].barh(range(len(impact_df)), impact_df['mean_impact'], color=colors)
        axes[0, 0].set_yticks(range(len(impact_df)))
        axes[0, 0].set_yticklabels(impact_df['feature'], fontsize=8)
        axes[0, 0].set_xlabel('Mean Impact on Yield')
        axes[0, 0].set_title(f'Feature Impacts - {country} {tenor}')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.7)

        # 1.2 Conformity Pie Chart
        conformity_counts = impact_df['conformity'].value_counts()
        axes[0, 1].pie(conformity_counts.values, labels=conformity_counts.index,
                       autopct='%1.1f%%', colors=[color_map[c] for c in conformity_counts.index])
        axes[0, 1].set_title('Economic Conformity Distribution')

        # 1.3 Impact vs Confidence Scatter
        scatter_colors = [color_map[conf] for conf in impact_df['conformity']]
        axes[1, 0].scatter(impact_df['confidence'], impact_df['abs_impact'],
                           c=scatter_colors, alpha=0.7)
        axes[1, 0].set_xlabel('Expected Relationship Confidence')
        axes[1, 0].set_ylabel('Absolute Impact')
        axes[1, 0].set_title('Impact Magnitude vs Confidence')

        # 1.4 Top Features Table
        axes[1, 1].axis('off')
        top_features = impact_df.head(10)[['feature', 'mean_impact', 'conformity']]
        table_data = []
        for _, row in top_features.iterrows():
            table_data.append([
                row['feature'][:20] + '...' if len(row['feature']) > 20 else row['feature'],
                f"{row['mean_impact']:.3f}",
                row['conformity']
            ])
        table = axes[1, 1].table(cellText=table_data,
                                 colLabels=['Feature', 'Impact', 'Conformity'],
                                 cellLoc='center',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        axes[1, 1].set_title('Top 10 Features by Impact')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{country}_{tenor}_comprehensive_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Economic Conformity Detailed Analysis
        plt.figure(figsize=(12, 8))
        conforming = impact_df[impact_df['conformity'] == 'Conforming']['abs_impact']
        non_conforming = impact_df[impact_df['conformity'] == 'Non-Conforming']['abs_impact']
        neutral = impact_df[impact_df['conformity'] == 'Neutral']['abs_impact']

        plt.hist([conforming, non_conforming, neutral],
                 label=['Conforming', 'Non-Conforming', 'Neutral'],
                 color=['green', 'red', 'gray'], alpha=0.7, bins=15)
        plt.xlabel('Absolute Impact Magnitude')
        plt.ylabel('Number of Features')
        plt.title(f'Impact Distribution by Economic Conformity - {country} {tenor}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{country}_{tenor}_conformity_distribution.png'), dpi=300)
        plt.close()

        # 3. Save detailed CSV
        impact_df.to_csv(os.path.join(output_dir, f'{country}_{tenor}_feature_analysis.csv'), index=False)

    def _generate_consolidated_reports(self, consolidated_results: Dict[str, Any],
                                       output_directory: str) -> None:
        """Generate consolidated reports across all models."""
        summary_path = os.path.join(output_directory, 'analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Model Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Timestamp: {consolidated_results['analysis_timestamp']}\n")
            f.write(f"Total Models Found: {consolidated_results['total_models']}\n")
            f.write(f"Successful Analyses: {consolidated_results['successful_analyses']}\n")
            f.write(f"Failed Analyses: {consolidated_results['failed_analyses']}\n\n")

            if consolidated_results['model_results']:
                f.write("Model Results Summary:\n")
                f.write("-" * 30 + "\n")
                for model_key, result in consolidated_results['model_results'].items():
                    conformity = result['sensitivity_results']['economic_conformity']
                    f.write(f"\n{model_key}:\n")
                    f.write(f" Features: {result['feature_count']}\n")
                    f.write(f" Conformity Score: {conformity['conformity_score']:.2%}\n")
                    f.write(f" Conforming Features: {conformity['conforming_features']}\n")
                    f.write(f" Non-Conforming Features: {conformity['non_conforming_features']}\n")

        self._create_cross_model_visualizations(consolidated_results, output_directory)

        results_path = os.path.join(output_directory, 'consolidated_results.json')
        with open(results_path, 'w') as f:
            json_results = self._convert_for_json(consolidated_results)
            json.dump(json_results, f, indent=2)

        logger.info(f"Consolidated reports saved to {output_directory}")

    def _create_cross_model_visualizations(self, consolidated_results: Dict[str, Any],
                                           output_directory: str) -> None:
        """Create visualizations comparing across all models."""
        if not consolidated_results['model_results']:
            return

        cross_model_data = []
        for model_key, result in consolidated_results['model_results'].items():
            conformity = result['sensitivity_results']['economic_conformity']
            cross_model_data.append({
                'model': model_key,
                'country': result['country'],
                'tenor': result['tenor'],
                'conformity_score': conformity['conformity_score'],
                'total_features': conformity['total_features'],
                'conforming_features': conformity['conforming_features']
            })

        cross_df = pd.DataFrame(cross_model_data)

        # 1. Conformity Score by Country
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 2, 1)
        country_conformity = cross_df.groupby('country')['conformity_score'].mean().sort_values(ascending=True)
        plt.barh(range(len(country_conformity)), country_conformity.values)
        plt.yticks(range(len(country_conformity)), country_conformity.index)
        plt.xlabel('Average Conformity Score')
        plt.title('Economic Conformity by Country')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)

        # 2. Conformity Score by Tenor
        plt.subplot(2, 2, 2)
        tenor_conformity = cross_df.groupby('tenor')['conformity_score'].mean().sort_values(ascending=True)
        plt.bar(tenor_conformity.index, tenor_conformity.values)
        plt.xlabel('Bond Tenor')
        plt.ylabel('Average Conformity Score')
        plt.title('Economic Conformity by Tenor')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        # 3. Scatter: Total Features vs Conforming Features
        plt.subplot(2, 2, 3)
        plt.scatter(cross_df['total_features'], cross_df['conforming_features'], alpha=0.7)
        plt.xlabel('Total Features')
        plt.ylabel('Conforming Features')
        plt.title('Total vs Conforming Features')
        max_features = cross_df['total_features'].max()
        plt.plot([0, max_features], [0, max_features], 'r--', alpha=0.7)

        # 4. Overall Distribution
        plt.subplot(2, 2, 4)
        plt.hist(cross_df['conformity_score'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Conformity Score')
        plt.ylabel('Number of Models')
        plt.title('Distribution of Conformity Scores')
        plt.axvline(x=cross_df['conformity_score'].mean(), color='red', linestyle='--',
                    label=f'Mean: {cross_df["conformity_score"].mean():.2f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'cross_model_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save cross-model data
        cross_df.to_csv(os.path.join(output_directory, 'cross_model_summary.csv'), index=False)

def main():
    """
    Main function to run comprehensive model testing with enhanced capabilities.
    """
    logger = config.configure_logging(level=config.DEFAULT_LOG_LEVEL)

    if len(sys.argv) == 1:
        print("Enhanced Model Tester")
        print("=" * 30)
        model_dir = config.MODEL_DIR
        output_dir = os.path.join(config.OUTPUT_DIR, 'model_analysis')
        max_workers = 4
        
    else:
        parser = argparse.ArgumentParser(
            description="Enhanced model diagnostics with statistical testing and economic validation"
        )
        parser.add_argument("--model-dir", required=False, default=config.MODEL_DIR,
                            help="Directory containing model files")
        parser.add_argument("--output-dir", required=False,
                            default=os.path.join(config.OUTPUT_DIR, 'model_analysis'),
                            help="Output directory for analysis results")
        parser.add_argument("--max-workers", type=int, default=4,
                            help="Maximum number of parallel workers")
        parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            default='INFO', help="Logging level")
        args = parser.parse_args()
        model_dir = args.model_dir
        output_dir = args.output_dir
        max_workers = args.max_workers
        if args.log_level:
            level = getattr(logging, args.log_level)
            logging.getLogger().setLevel(level)

    if not os.path.isdir(model_dir):
        logger.error(f"Model directory does not exist: {model_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    tester = ModelTester()

    logger.info(f"Starting enhanced model analysis...")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max workers: {max_workers}")

    try:
        results = tester.run_comprehensive_analysis(
            model_directory=model_dir,
            output_directory=output_dir,
            max_workers=max_workers
        )
        logger.info("Analysis Complete!")
        logger.info(f"Total models processed: {results['total_models']}")
        logger.info(f"Successful analyses: {results['successful_analyses']}")
        logger.info(f"Failed analyses: {results['failed_analyses']}")
        if results['successful_analyses'] > 0:
            conformity_scores = [
                r['sensitivity_results']['economic_conformity']['conformity_score']
                for r in results['model_results'].values()
            ]
            avg_conformity = np.mean(conformity_scores)
            logger.info(f"Average economic conformity score: {avg_conformity:.2%}")
            logger.info(f"Detailed results saved to: {output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
