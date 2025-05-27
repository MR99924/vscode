"""
Custom model classes for yield curve modeling.
This module contains sklearn-compatible model wrappers for various model types,
focusing on traditional ML and statistical methods without TensorFlow dependencies.
"""
import config
import numpy as np
import pandas as pd
import logging
import joblib
import os
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import warnings

# Configure logging
logger = logging.getLogger(__name__)


def add_missing_mlp_attributes(model):
    """
    Safely add missing attributes to an MLPRegressor model.
    This fixes the '_best_coefs' attribute error.
    """
    if hasattr(model, '__class__') and model.__class__.__name__ == 'MLPRegressor':
        if hasattr(model, 'coefs_') and not hasattr(model, '_best_coefs'):
            model._best_coefs = model.coefs_
        if hasattr(model, 'intercepts_') and not hasattr(model, '_best_intercepts'):
            model._best_intercepts = model.intercepts_
        # Add best_score attribute if needed
        if hasattr(model, 'best_loss_') and not hasattr(model, 'best_score_'):
            model.best_score_ = -model.best_loss_
    
    return model

class TimeSeriesEstimator(BaseEstimator, RegressorMixin):
    """
    A custom sklearn-compatible wrapper for time series models like ARIMA.
    
    Parameters:
        p (int): AR order
        d (int): Differencing order
        q (int): MA order
        seasonal (bool): Whether to use seasonal component
        seasonal_order (tuple): Seasonal order parameters (P,D,Q,s)
        trend (str): Trend component ('n', 'c', 't', 'ct')
    """
    def __init__(self, p=1, d=1, q=0, seasonal=False, seasonal_order=(0,0,0,0), trend='c'):
        self.p = p
        self.d = d
        self.q = q
        self.seasonal = seasonal
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.result_ = None
        
    def fit(self, X, y=None):
        """
        Fit the time series model.
        For ARIMA models, we only use the target series y.
        
        Parameters:
            X: Not used, maintained for sklearn compatibility
            y: Time series data to fit
            
        Returns:
            self: The fitted estimator
        """
        # For ARIMA, we only need y (time series)
        try:
            if self.seasonal:
                self.model = SARIMAX(
                    y, 
                    order=(self.p, self.d, self.q), 
                    seasonal_order=self.seasonal_order,
                    trend=self.trend
                )
            else:
                self.model = ARIMA(
                    y, 
                    order=(self.p, self.d, self.q),
                    trend=self.trend
                )
                
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.result_ = self.model.fit(disp=False)
            
            return self
        except Exception as e:
            logger.error(f"Error fitting time series model: {str(e)}")
            # Return self to maintain compatibility even if fitting fails
            return self
        
    def predict(self, X):
        """
        Generate predictions from the fitted model.
        
        Parameters:
            X: Not used, maintained for sklearn compatibility
            
        Returns:
            array: Model predictions
        """
        if self.result_ is None:
            # Return zeros if model failed to fit
            return np.zeros(len(X))
        
        # Return in-sample predictions for the training data
        try:
            preds = self.result_.fittedvalues
            
            # If X is longer than fitted values, pad with the last value
            if len(X) > len(preds):
                preds = np.append(preds, np.full(len(X) - len(preds), preds[-1]))
            # If X is shorter, truncate
            elif len(X) < len(preds):
                preds = preds[:len(X)]
                
            return preds
        except Exception as e:
            logger.error(f"Error in ARIMA predict: {str(e)}")
            return np.zeros(len(X))
    
    def forecast(self, steps=1):
        """
        Generate out-of-sample forecasts.
        
        Parameters:
            steps (int): Number of steps to forecast
            
        Returns:
            array: Forecast values
        """
        if self.result_ is None:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            return self.result_.forecast(steps=steps)
        except Exception as e:
            logger.error(f"Error in forecast: {str(e)}")
            return np.zeros(steps)

class VAREstimator(BaseEstimator, RegressorMixin):
    """
    Vector Autoregression (VAR) model for multivariate time series.
    
    Parameters:
        lags (int): Number of lags to include
        trend (str): Trend specification
    """
    def __init__(self, lags=1, trend='c'):
        self.lags = lags
        self.trend = trend
        self.model = None
        self.result_ = None
        self.target_idx = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        """
        Fit the VAR model using both features and target.
        
        Parameters:
            X: Feature matrix
            y: Target variable
            
        Returns:
            self: The fitted estimator
        """
        try:
            # Combine features and target into one DataFrame
            if isinstance(X, pd.DataFrame):
                data = X.copy()
                self.feature_names = X.columns.tolist()
            else:
                data = pd.DataFrame(X)
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y_name = y.name if y.name else 'target'
                data[y_name] = y
                self.target_idx = data.columns.get_loc(y_name)
            else:
                data['target'] = y
                self.target_idx = data.columns.get_loc('target')
            
            # Create and fit the VAR model
            try:
                self.model = VAR(data)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.result_ = self.model.fit(self.lags, trend=self.trend)
                
                return self
            except Exception as e:
                logger.error(f"Error fitting VAR model: {str(e)}")
                return self
        except Exception as e:
            logger.error(f"Error preparing data for VAR: {str(e)}")
            return self
    
    def predict(self, X):
        """
        Generate predictions from the fitted model.
        
        Parameters:
            X: Feature matrix
            
        Returns:
            array: Predictions for target variable
        """
        if self.result_ is None:
            # Return zeros if model failed to fit
            return np.zeros(len(X))
        
        try:
            # For in-sample predictions
            all_preds = self.result_.fittedvalues
            
            # Extract only target variable predictions
            target_preds = all_preds[:, self.target_idx]
            
            # Return predictions aligned with X
            result = np.full(len(X), np.nan)
            result[self.lags:] = target_preds
            
            # Fill NaN values at beginning with the first prediction
            if np.any(np.isnan(result)) and len(target_preds) > 0:
                result[np.isnan(result)] = target_preds[0]
            
            return result
        except Exception as e:
            logger.error(f"Error in VAR predict: {str(e)}")
            return np.zeros(len(X))

class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model combining predictions from multiple models with optional optimal weighting.
    
    Parameters:
        models (list): List of sklearn-compatible models
        weights (list): List of weights for each model (optional)
        optimize_weights (bool): Whether to optimize weights using validation data
    """
    def __init__(self, models=None, weights=None, optimize_weights=False):
        self.models = models if models is not None else []
        self.weights = weights
        self.optimize_weights = optimize_weights
        self.fitted_models_ = []
        self.is_fitted_ = False
        
    def fit(self, X, y=None, validation_data=None):
        """
        Fit all models in the ensemble.
        
        Parameters:
            X (array): Feature matrix
            y (array): Target vector
            validation_data (tuple): (X_val, y_val) for weight optimization
            
        Returns:
            self: The fitted estimator
        """
        # Fit each model individually
        self.fitted_models_ = []
        for i, model in enumerate(self.models):
            try:
                logger.info(f"Fitting ensemble model {i+1}/{len(self.models)}")
                fitted_model = clone(model).fit(X, y)
                self.fitted_models_.append(fitted_model)
            except Exception as e:
                logger.error(f"Error fitting model {i+1}: {str(e)}")
                # Add a dummy model that returns zeros
                self.fitted_models_.append(DummyModel())
        
        # Optimize weights if requested and validation data provided
        if self.optimize_weights and validation_data is not None and len(self.fitted_models_) > 0:
            X_val, y_val = validation_data
            
            # Get predictions from each model
            val_predictions = []
            for model in self.fitted_models_:
                try:
                    val_predictions.append(model.predict(X_val))
                except Exception as e:
                    logger.error(f"Error in model prediction during weight optimization: {str(e)}")
                    val_predictions.append(np.zeros(len(X_val)))
            
            # Convert to array
            val_predictions = np.array(val_predictions)
            
            # Define objective function (MSE)
            def objective(weights):
                # Ensure weights are positive and sum to 1
                weights = np.abs(weights)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
                ensemble_pred = np.sum(val_predictions.T * weights, axis=1)
                return mean_squared_error(y_val, ensemble_pred)
            
            # Initial weights (equal)
            initial_weights = np.ones(len(self.fitted_models_)) / len(self.fitted_models_)
            
            # Bounds (all weights between 0 and 1)
            bounds = [(0, 1) for _ in range(len(self.fitted_models_))]
            
            try:
                # Optimize
                result = minimize(
                    objective,
                    initial_weights,
                    bounds=bounds,
                    method='SLSQP',
                    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
                )
                
                # Set optimized weights
                self.weights = result.x / np.sum(result.x)
                logger.info(f"Optimized ensemble weights: {self.weights}")
            except Exception as e:
                logger.error(f"Error optimizing weights: {str(e)}")
                # Use equal weights if optimization fails
                self.weights = np.ones(len(self.fitted_models_)) / len(self.fitted_models_)
        
        self.is_fitted_ = len(self.fitted_models_) > 0
        return self
    
    def predict(self, X):
        """
        Generate predictions as a weighted average of all models.
        
        Parameters:
            X (array): Feature matrix
            
        Returns:
            array: Ensemble predictions
        """
        if not self.is_fitted_ or len(self.fitted_models_) == 0:
            logger.warning("Ensemble not fitted or no models available. Returning zeros.")
            return np.zeros(len(X))
            
        # Get predictions from each model
        predictions = []
        for model in self.fitted_models_:
            try:
                model_preds = model.predict(X)
                predictions.append(model_preds)
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                predictions.append(np.zeros(len(X)))
        
        predictions = np.array(predictions)
        
        # Apply weights if specified, otherwise use simple average
        if self.weights is None:
            weights = np.ones(len(self.fitted_models_)) / len(self.fitted_models_)
        else:
            weights = np.array(self.weights) / np.sum(self.weights)
            
        # Return weighted average
        return np.sum(predictions.T * weights, axis=1)


    def add_model(self, model):
        """
        Add a pre-fitted model to the ensemble.
        
        Parameters:
            model: A fitted model to add to the ensemble
            
        Returns:
            self: Updated ensemble
        """
        self.fitted_models_.append(model)
        self.is_fitted_ = len(self.fitted_models_) > 0
        
        # Update weights to equal weighting if not specified
        if self.weights is None or len(self.weights) != len(self.fitted_models_):
            self.weights = np.ones(len(self.fitted_models_)) / len(self.fitted_models_)
            
        return self
    
    def evaluate_individual_models(self, X, y):
        """
        Evaluate each model in the ensemble individually.
        
        Parameters:
            X (array): Feature matrix
            y (array): Target vector
            
        Returns:
            dict: Performance metrics for each model
        """
        results = {}
        
        for i, model in enumerate(self.fitted_models_):
            try:
                preds = model.predict(X)
                mse = mean_squared_error(y, preds)
                r2 = r2_score(y, preds)
                mae = mean_absolute_error(y, preds)
                
                results[f'model_{i}'] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'mae': mae
                }
            except Exception as e:
                logger.error(f"Error evaluating model {i}: {str(e)}")
                results[f'model_{i}'] = {
                    'mse': float('nan'),
                    'rmse': float('nan'),
                    'r2': float('nan'),
                    'mae': float('nan'),
                    'error': str(e)
                }
        
        # Also evaluate the ensemble
        try:
            ensemble_preds = self.predict(X)
            ensemble_mse = mean_squared_error(y, ensemble_preds)
            ensemble_r2 = r2_score(y, ensemble_preds)
            ensemble_mae = mean_absolute_error(y, ensemble_preds)
            
            results['ensemble'] = {
                'mse': ensemble_mse,
                'rmse': np.sqrt(ensemble_mse),
                'r2': ensemble_r2,
                'mae': ensemble_mae
            }
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            results['ensemble'] = {
                'mse': float('nan'),
                'rmse': float('nan'),
                'r2': float('nan'),
                'mae': float('nan'),
                'error': str(e)
            }
        
        return results

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble model that uses predictions from base models as features for a meta-model.
    
    Parameters:
        base_models (list): List of sklearn-compatible base models
        meta_model: Sklearn-compatible meta model
    """
    def __init__(self, base_models=None, meta_model=None):
        self.base_models = base_models if base_models is not None else []
        self.meta_model = meta_model if meta_model is not None else Ridge()
        self.fitted_base_models_ = []
        self.fitted_meta_model_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit the stacking ensemble using cross-validation to get base model predictions.
        
        Parameters:
            X (array): Feature matrix
            y (array): Target vector
            
        Returns:
            self: The fitted estimator
        """
        # Create cross-validation folds (time series aware if possible)
        try:
            if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
                # Time series cross-validation for time series data
                kf = TimeSeriesSplit(n_splits=3)
            else:
                # Regular K-fold for non-time series data
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
        except Exception as e:
            logger.error(f"Error creating CV folds: {str(e)}")
            # Fallback to regular KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Train base models and collect meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)} for stacking")
            # Create a copy of the model for cross-validation
            cv_model = clone(model)
            
            # Use cross-validation to generate unbiased predictions for meta-training
            for train_idx, val_idx in kf.split(X):
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]
                
                try:
                    # Train model on training fold
                    cv_model.fit(X_train, y_train)
                    
                    # Generate predictions for validation fold
                    meta_features[val_idx, i] = cv_model.predict(X_val)
                except Exception as e:
                    logger.error(f"Error in CV fold for base model {i+1}: {str(e)}")
                    # Fill with zeros if prediction fails
                    meta_features[val_idx, i] = 0
            
            # Fit final model on all data
            try:
                final_model = clone(model).fit(X, y)
                self.fitted_base_models_.append(final_model)
            except Exception as e:
                logger.error(f"Error fitting final base model {i+1}: {str(e)}")
                self.fitted_base_models_.append(DummyModel())
        
        # Train meta model on meta features
        try:
            self.fitted_meta_model_ = clone(self.meta_model).fit(meta_features, y)
            self.is_fitted_ = True
        except Exception as e:
            logger.error(f"Error fitting meta model: {str(e)}")
            # Use a dummy model if meta model fitting fails
            self.fitted_meta_model_ = DummyModel()
            self.is_fitted_ = False
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the stacking ensemble.
        
        Parameters:
            X (array): Feature matrix
            
        Returns:
            array: Stacking ensemble predictions
        """
        if not self.is_fitted_ or self.fitted_meta_model_ is None:
            logger.warning("Stacking ensemble not properly fitted. Returning zeros.")
            return np.zeros(len(X))
        
        # Generate meta-features from base models
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models_)))
        
        for i, model in enumerate(self.fitted_base_models_):
            try:
                meta_features[:, i] = model.predict(X)
            except Exception as e:
                logger.error(f"Error generating meta-features from model {i+1}: {str(e)}")
                # Fill with zeros if prediction fails
        
        # Use meta-model to make final predictions
        try:
            return self.fitted_meta_model_.predict(meta_features)
        except Exception as e:
            logger.error(f"Error in meta-model prediction: {str(e)}")
            return np.zeros(len(X))

class EnhancedMLP(BaseEstimator, RegressorMixin):
    """
    Enhanced MLP with regularization, weight initialization techniques, and ensemble capabilities.
    
    Parameters:
        hidden_layer_sizes (tuple): Size of hidden layers
        activation (str): Activation function
        solver (str): Optimization algorithm
        alpha (float): L2 regularization parameter
        batch_size (int/str): Batch size for gradient-based optimization
        learning_rate (str): Learning rate schedule
        learning_rate_init (float): Initial learning rate
        max_iter (int): Maximum number of iterations
        early_stopping (bool): Whether to use early stopping
        validation_fraction (float): Fraction of training data for validation
        n_iter_no_change (int): Max epochs with no validation improvement
        ensemble_size (int): Number of MLPs to train with different initializations
    """
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='adaptive',
                 learning_rate_init=0.001, max_iter=1000, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=10, ensemble_size=5):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.ensemble_size = ensemble_size
        self.models_ = []
        self.is_fitted_ = False
        self.scaler_ = None
        
    def fit(self, X, y=None):
        """
        Fit the enhanced MLP ensemble.
        
        Parameters:
            X (array): Feature matrix
            y (array): Target vector
            
        Returns:
            self: The fitted estimator
        """
        # Scale data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Train ensemble of MLPs with different random initializations
        self.models_ = []
        for i in range(self.ensemble_size):
            try:
                logger.info(f"Training MLP {i+1}/{self.ensemble_size}")
                
                mlp =  MLPRegressor(
                hidden_layer_sizes=(32, 16),  # Smaller network
                activation='relu',
                solver='adam',
                alpha=0.01,  # Stronger regularization
                max_iter=2000,  # More iterations
                tol=1e-4,  # More relaxed tolerance
                random_state=42,
                n_iter_no_change=20,
                early_stopping=True,
                validation_fraction=0.2  # Different seed for each model
                )
                
                mlp.fit(X_scaled, y)
                self.models_.append(mlp)
                
                # Log the training progress
                if hasattr(mlp, 'loss_'):
                    logger.info(f"MLP {i+1}/{self.ensemble_size} trained - Loss: {mlp.loss_:.6f}")
            except Exception as e:
                logger.error(f"Error training MLP {i+1}: {str(e)}")
                # Skip failed model
        
        self.is_fitted_ = len(self.models_) > 0
        return self
    
    def predict(self, X):
        """
        Generate predictions from the MLP ensemble.
        
        Parameters:
            X (array): Feature matrix
            
        Returns:
            array: Ensemble predictions
        """
        if not self.is_fitted_ or len(self.models_) == 0:
            logger.warning("MLP ensemble not fitted. Returning zeros.")
            return np.zeros(len(X))
        
        # Scale the input data
        X_scaled = self.scaler_.transform(X)
        
        # Get predictions from each MLP
        predictions = []
        for mlp in self.models_:
            try:
                pred = mlp.predict(X_scaled)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error in MLP prediction: {str(e)}")
                # Use zeros for failed predictions
                predictions.append(np.zeros(len(X)))
        
        # Convert to array and handle empty predictions
        if not predictions:
            return np.zeros(len(X))
            
        predictions = np.array(predictions)
        
        # Return average prediction
        return np.mean(predictions, axis=0)
    
    def feature_importances(self, X):
        """
        Compute feature importance based on connection weights.
        
        Parameters:
            X (array): Feature matrix
            
        Returns:
            array: Feature importance scores
        """
        if not self.is_fitted_ or len(self.models_) == 0:
            logger.warning("Cannot calculate feature importances for unfitted model")
            return np.zeros(X.shape[1])
        
        importances = []
        for mlp in self.models_:
            try:
                # Get weights of connections from input to first hidden layer
                weights = np.abs(mlp.coefs_[0])
                
                # Average the weights across all nodes in the first hidden layer
                importance = np.mean(weights, axis=1)
                
                # Normalize to sum to 1
                importance = importance / np.sum(importance)
                importances.append(importance)
            except Exception as e:
                logger.error(f"Error calculating feature importance: {str(e)}")
                # Skip failed importance calculation
        
        if not importances:
            return np.zeros(X.shape[1])
            
        # Average importances across all models
        avg_importance = np.mean(np.array(importances), axis=0)
        
        return avg_importance

class GaussianProcessYieldModel(BaseEstimator, RegressorMixin):
    """
    Gaussian Process Regression model specifically designed for yield curve modeling.
    Incorporates both temporal patterns and feature relationships.
    
    Parameters:
        length_scale (float): Length scale parameter for RBF kernel
        alpha (float): Noise parameter
        n_restarts_optimizer (int): Number of restarts for hyperparameter optimization
        normalize_y (bool): Whether to normalize the target variable
    """
    def __init__(self, length_scale=1.0, alpha=0.1, n_restarts_optimizer=5, normalize_y=True):
        self.length_scale = length_scale
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.model = None
        self.scaler = None
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit the Gaussian Process model.
        
        Parameters:
            X (array): Feature matrix
            y (array): Target vector
            
        Returns:
            self: The fitted estimator
        """
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create a kernel that can capture both temporal patterns and feature relationships
        # RBF kernel for feature relationships
        rbf_kernel = RBF(length_scale=self.length_scale)
        
        # White kernel for noise
        white_kernel = WhiteKernel(noise_level=self.alpha)
        
        # Rational quadratic kernel can model multiple length scales
        rational_kernel = RationalQuadratic(length_scale=self.length_scale, alpha=0.1)
        
        # Combine kernels
        kernel = rbf_kernel + white_kernel + rational_kernel
        
        try:
            # Create and fit GP model
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=42
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model.fit(X_scaled, y)
            
            self.is_fitted_ = True
            logger.info("Gaussian Process model fitted successfully")
            
            # Log the optimized kernel parameters
            logger.info(f"Optimized kernel: {self.model.kernel_}")
            
        except Exception as e:
            logger.error(f"Error fitting Gaussian Process model: {str(e)}")
            self.is_fitted_ = False
            
        return self
    
    def predict(self, X, return_std=False):
        """
        Generate predictions and optionally standard deviations.
        
        Parameters:
            X (array): Feature matrix
            return_std (bool): Whether to return standard deviations
            
        Returns:
            array or tuple: Predictions (and standard deviations if requested)
        """
        if not self.is_fitted_:
            logger.warning("Gaussian Process model not fitted. Returning zeros.")
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            if return_std:
                mean, std = self.model.predict(X_scaled, return_std=True)
                return mean, std
            else:
                return self.model.predict(X_scaled)
                
        except Exception as e:
            logger.error(f"Error in Gaussian Process prediction: {str(e)}")
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))
    
    def get_kernel_parameters(self):
        """
        Get the optimized kernel parameters.
        
        Returns:
            dict: Dictionary of kernel parameters
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted"}
            
        try:
            return {
                "kernel": str(self.model.kernel_),
                "log_marginal_likelihood": float(self.model.log_marginal_likelihood(self.model.kernel_.theta))
            }
        except Exception as e:
            logger.error(f"Error retrieving kernel parameters: {str(e)}")
            return {"error": str(e)}


class YieldCurveNelsonSiegel(BaseEstimator, RegressorMixin):
    """
    Nelson-Siegel model for yield curve fitting and forecasting.
    Models the yield curve as a function of tenor using the Nelson-Siegel parameterization.
    
    Parameters:
        estimation_method (str): Method for parameter estimation ('ols', 'mle')
        lambda_fixed (bool): Whether to fix the lambda parameter
        lambda_value (float): Fixed value for lambda if lambda_fixed is True
    """
    def __init__(self, estimation_method='ols', lambda_fixed=True, lambda_value=0.7308):
        self.estimation_method = estimation_method
        self.lambda_fixed = lambda_fixed
        self.lambda_value = lambda_value
        self.params_ = None
        self.is_fitted_ = False
        
    def nelson_siegel_factors(self, tau, lambda_val):
        """
        Calculate Nelson-Siegel factors for given tenors and lambda.
        
        Parameters:
            tau (array): Array of tenors (in years)
            lambda_val (float): Lambda parameter
            
        Returns:
            tuple: Three factors (level, slope, curvature)
        """
        # Level factor (constant 1)
        factor1 = np.ones_like(tau)
        
        # Slope factor (1 - exp(-lambda*tau))/(lambda*tau)
        factor2 = (1 - np.exp(-lambda_val * tau)) / (lambda_val * tau)
        
        # Curvature factor ((1 - exp(-lambda*tau))/(lambda*tau) - exp(-lambda*tau))
        factor3 = (1 - np.exp(-lambda_val * tau)) / (lambda_val * tau) - np.exp(-lambda_val * tau)
        
        return factor1, factor2, factor3
    
    def fit(self, X, y=None):
        """
        Fit the Nelson-Siegel model using yield curve data.
        
        Parameters:
            X (array): Tenors (in years)
            y (array): Yields for corresponding tenors
            
        Returns:
            self: The fitted estimator
        """
        try:
            # Store tenors and yields
            tenors = X.flatten()  # Ensure 1D array
            yields = y.flatten()  # Ensure 1D array
            
            if self.lambda_fixed:
                # Use fixed lambda
                lambda_val = self.lambda_value
                
                # Calculate factors
                factor1, factor2, factor3 = self.nelson_siegel_factors(tenors, lambda_val)
                
                # Create design matrix
                X_design = np.column_stack((factor1, factor2, factor3))
                
                # Estimate parameters (beta0, beta1, beta2) using OLS
                from sklearn.linear_model import LinearRegression
                model = LinearRegression(fit_intercept=False)
                model.fit(X_design, yields)
                
                # Store parameters
                self.params_ = {
                    'beta0': model.coef_[0],  # Level
                    'beta1': model.coef_[1],  # Slope
                    'beta2': model.coef_[2],  # Curvature
                    'lambda': lambda_val
                }
                
            else:
                # Optimize lambda along with other parameters
                def objective_function(params):
                    beta0, beta1, beta2, lambda_val = params
                    
                    # Ensure lambda is positive
                    lambda_val = abs(lambda_val)
                    
                    # Calculate factors
                    factor1, factor2, factor3 = self.nelson_siegel_factors(tenors, lambda_val)
                    
                    # Calculate predicted yields
                    y_pred = beta0 * factor1 + beta1 * factor2 + beta2 * factor3
                    
                    # Return sum of squared errors
                    return np.sum((yields - y_pred) ** 2)
                
                # Initial parameter guess
                initial_params = [yields.mean(), -1, 0, 0.7308]
                
                # Bounds for parameters (beta0, beta1, beta2, lambda)
                bounds = [(None, None), (None, None), (None, None), (0.01, 5.0)]
                
                # Optimize
                from scipy.optimize import minimize
                result = minimize(
                    objective_function,
                    initial_params,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                # Store optimized parameters
                beta0, beta1, beta2, lambda_val = result.x
                self.params_ = {
                    'beta0': beta0,  # Level
                    'beta1': beta1,  # Slope
                    'beta2': beta2,  # Curvature
                    'lambda': abs(lambda_val)
                }
            
            self.is_fitted_ = True
            logger.info("Nelson-Siegel model fitted successfully")
            logger.info(f"Fitted parameters: {self.params_}")
            
        except Exception as e:
            logger.error(f"Error fitting Nelson-Siegel model: {str(e)}")
            self.is_fitted_ = False
            
        return self
    
    def predict(self, X):
        """
        Predict yields for given tenors using the fitted Nelson-Siegel model.
        
        Parameters:
            X (array): Tenors (in years)
            
        Returns:
            array: Predicted yields
        """
        if not self.is_fitted_:
            logger.warning("Nelson-Siegel model not fitted. Returning zeros.")
            return np.zeros(len(X))
        
        try:
            # Extract parameters
            beta0 = self.params_['beta0']
            beta1 = self.params_['beta1']
            beta2 = self.params_['beta2']
            lambda_val = self.params_['lambda']
            
            # Reshape input if needed
            tenors = X.flatten()  # Ensure 1D array
            
            # Calculate factors
            factor1, factor2, factor3 = self.nelson_siegel_factors(tenors, lambda_val)
            
            # Calculate predicted yields
            y_pred = beta0 * factor1 + beta1 * factor2 + beta2 * factor3
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Error in Nelson-Siegel prediction: {str(e)}")
            return np.zeros(len(X))


class DummyModel(BaseEstimator, RegressorMixin):
    """
    A dummy model that always returns zeros. Used as a fallback when other models fail.
    """
    def __init__(self):
        self.is_fitted_ = True
        
    def fit(self, X, y=None):
        """
        Dummy fit method, does nothing.
        
        Parameters:
            X (array): Not used
            y (array): Not used
            
        Returns:
            self: The dummy estimator
        """
        return self
        
    def predict(self, X):
        """
        Return zeros for all inputs.
        
        Parameters:
            X (array): Input array
            
        Returns:
            array: Array of zeros with same length as X
        """
        return np.zeros(len(X))


class ModelFactory:
    """
    Factory class for creating and configuring various yield curve models.
    """
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a model of the specified type with given parameters.
        
        Parameters:
            model_type (str): Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            model: Instantiated model object
        """
        model_type = model_type.lower()
        
        if model_type == 'mlp':
            # Neural network model
            params = {
                'hidden_layer_sizes': kwargs.get('hidden_layer_sizes', (64, 32)),
                'activation': kwargs.get('activation', 'relu'),
                'solver': kwargs.get('solver', 'adam'),
                'alpha': kwargs.get('alpha', 0.0001),
                'max_iter': kwargs.get('max_iter', 1000),
                'early_stopping': kwargs.get('early_stopping', True),
                'validation_fraction': kwargs.get('validation_fraction', 0.1),
                'n_iter_no_change': kwargs.get('n_iter_no_change', 10),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return MLPRegressor(**params)
            
        elif model_type == 'enhanced_mlp':
            # Enhanced MLP with ensemble capabilities
            params = {
                'hidden_layer_sizes': kwargs.get('hidden_layer_sizes', (64, 32)),
                'activation': kwargs.get('activation', 'relu'),
                'solver': kwargs.get('solver', 'adam'),
                'alpha': kwargs.get('alpha', 0.0001),
                'max_iter': kwargs.get('max_iter', 1000),
                'early_stopping': kwargs.get('early_stopping', True),
                'validation_fraction': kwargs.get('validation_fraction', 0.1),
                'n_iter_no_change': kwargs.get('n_iter_no_change', 10),
                'ensemble_size': kwargs.get('ensemble_size', 5)
            }
            
            return EnhancedMLP(**params)
            
        elif model_type == 'elasticnet':
            # Elastic Net model
            params = {
                'alpha': kwargs.get('alpha', 0.1),
                'l1_ratio': kwargs.get('l1_ratio', 0.5),
                'max_iter': kwargs.get('max_iter', 1000),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return ElasticNet(**params)
            
        elif model_type == 'ridge':
            # Ridge regression model
            params = {
                'alpha': kwargs.get('alpha', 1.0),
                'max_iter': kwargs.get('max_iter', 1000),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return Ridge(**params)
            
        elif model_type == 'lasso':
            # Lasso regression model
            params = {
                'alpha': kwargs.get('alpha', 0.1),
                'max_iter': kwargs.get('max_iter', 1000),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return Lasso(**params)
            
        elif model_type == 'gbm':
            # Gradient Boosting Machine
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'learning_rate': kwargs.get('learning_rate', 0.1),
                'max_depth': kwargs.get('max_depth', 3),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return GradientBoostingRegressor(**params)
            
        elif model_type == 'randomforest':
            # Random Forest model
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', None),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return RandomForestRegressor(**params)
            
        elif model_type == 'xgboost':
            # XGBoost model
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'learning_rate': kwargs.get('learning_rate', 0.1),
                'max_depth': kwargs.get('max_depth', 3),
                'reg_alpha': kwargs.get('reg_alpha', 0),
                'reg_lambda': kwargs.get('reg_lambda', 1),
                'random_state': kwargs.get('random_state', 42)
            }
            
            return xgb.XGBRegressor(**params)
            
        elif model_type == 'ensemble':
            # Ensemble of models
            base_models = kwargs.get('base_models', [])
            weights = kwargs.get('weights', None)
            optimize_weights = kwargs.get('optimize_weights', False)
            
            if not base_models:
                # Create default ensemble with common models
                base_models = [
                    ModelFactory.create_model('mlp'),
                    ModelFactory.create_model('gbm'),
                    ModelFactory.create_model('elasticnet')
                ]
            
            return EnsembleModel(models=base_models, weights=weights, optimize_weights=optimize_weights)
            
        elif model_type == 'stacking':
            # Stacking ensemble
            base_models = kwargs.get('base_models', [])
            meta_model = kwargs.get('meta_model', Ridge())
            
            if not base_models:
                # Create default stacking ensemble with common models
                base_models = [
                    ModelFactory.create_model('mlp'),
                    ModelFactory.create_model('gbm'),
                    ModelFactory.create_model('elasticnet')
                ]
            
            return StackingEnsemble(base_models=base_models, meta_model=meta_model)
            
        elif model_type == 'arima':
            # ARIMA time series model
            params = {
                'p': kwargs.get('p', 1),
                'd': kwargs.get('d', 1),
                'q': kwargs.get('q', 0),
                'seasonal': kwargs.get('seasonal', False),
                'seasonal_order': kwargs.get('seasonal_order', (0,0,0,0)),
                'trend': kwargs.get('trend', 'c')
            }
            
            return TimeSeriesEstimator(**params)
            
        elif model_type == 'var':
            # VAR time series model
            params = {
                'lags': kwargs.get('lags', 1),
                'trend': kwargs.get('trend', 'c')
            }
            
            return VAREstimator(**params)
            
        elif model_type == 'gp':
            # Gaussian Process model
            params = {
                'length_scale': kwargs.get('length_scale', 1.0),
                'alpha': kwargs.get('alpha', 0.1),
                'n_restarts_optimizer': kwargs.get('n_restarts_optimizer', 5),
                'normalize_y': kwargs.get('normalize_y', True)
            }
            
            return GaussianProcessYieldModel(**params)
            
        elif model_type == 'nelson_siegel':
            # Nelson-Siegel yield curve model
            params = {
                'estimation_method': kwargs.get('estimation_method', 'ols'),
                'lambda_fixed': kwargs.get('lambda_fixed', True),
                'lambda_value': kwargs.get('lambda_value', 0.7308)
            }
            
            return YieldCurveNelsonSiegel(**params)
            
        else:
            logger.error(f"Unknown model type: {model_type}")
            return DummyModel()
    
    @staticmethod
    def create_pipeline(model_type, with_scaler=True, with_target_transform=False, **kwargs):
        """
        Create a pipeline with optional preprocessing steps.
        
        Parameters:
            model_type (str): Type of model to create
            with_scaler (bool): Whether to include a scaler
            with_target_transform (bool): Whether to log-transform the target
            **kwargs: Model-specific parameters
            
        Returns:
            Pipeline: Scikit-learn pipeline with preprocessing and model
        """
        model = ModelFactory.create_model(model_type, **kwargs)
        
        steps = []
        
        # Add scaler if requested
        if with_scaler:
            steps.append(('scaler', StandardScaler()))
        
        # Add model
        steps.append(('model', model))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        # Wrap with target transformer if needed
        if with_target_transform:
            # Function to log-transform the target while handling zeros and negative values
            def func(y):
                # Shift to make all values positive
                min_y = min(y) if min(y) < 0 else 0
                shifted = y - min_y + 0.01
                return np.log(shifted)
                
            def inverse_func(y):
                result = np.exp(y)
                # Reverse the shift
                min_y = min(y) if min(y) < 0 else 0
                return result + min_y - 0.01
            
            pipeline = TransformedTargetRegressor(
                regressor=pipeline,
                func=func,
                inverse_func=inverse_func
            )
        
        return pipeline


def save_model_package(model, scaler, feature_names, country, tenor, output_dir=None):
    """
    Save a model package including the model, scaler, and metadata.
    
    Parameters:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        country: Country name
        tenor: Tenor name
        output_dir: Directory to save the model
        
    Returns:
        str: Path to saved model package
    """
    if output_dir is None:
        output_dir = os.path.join(config.MODEL_DIR, country)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metadata': {
            'country': country,
            'tenor': tenor,
            'created_at': timestamp,
            'model_type': type(model).__name__
        }
    }
    
    # Save model package
    model_path = os.path.join(output_dir, f"{country}_{tenor}_best_model.pkl")
    joblib.dump(model_package, model_path)
    
    logger.info(f"Model package saved to {model_path}")
    
    return model_path


def load_model_package(model_path):
    """
    Load a saved model package.
    
    Parameters:
        model_path: Path to saved model package
        
    Returns:
        dict: Model package with model, scaler, and metadata
    """
    try:
        model_package = joblib.load(model_path)
        
        # Check if the package has the expected structure
        expected_keys = ['model', 'scaler', 'feature_names', 'metadata']
        for key in expected_keys:
            if key not in model_package:
                logger.warning(f"Model package missing expected key: {key}")
        
        logger.info(f"Loaded model package from {model_path}")
        
        return model_package
    except Exception as e:
        logger.error(f"Error loading model package: {str(e)}")
        return None

def safe_mlp_get_attribute(model, attribute_name, default_value=None):
    """
    Safely get an attribute from an MLPRegressor model, accounting for version differences.
    
    Parameters:
        model: MLPRegressor model
        attribute_name: Name of the attribute to get
        default_value: Default value to return if attribute doesn't exist
        
    Returns:
        The attribute value or default_value if not found
    """
    # For '_best_coefs' specifically, try alternative attributes first
    if attribute_name == '_best_coefs':
        if hasattr(model, 'coefs_'):
            return model.coefs_
        elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'coefs_'):
            return model.best_estimator_.coefs_
    
    # General case - try to get the attribute directly
    return getattr(model, attribute_name, default_value)

def create_model_diagnostics(model, X, y, scaler=None, feature_names=None, country=None, tenor=None):
    """
    Create comprehensive model diagnostics.
    
    Parameters:
        model: Trained model
        X: Feature matrix
        y: Target vector
        scaler: Fitted scaler (optional)
        feature_names: List of feature names (optional)
        country: Country name (optional)
        tenor: Tenor name (optional)
        
    Returns:
        dict: Diagnostic information
    """
    diagnostics = {
        'model_type': type(model).__name__,
        'country': country,
        'tenor': tenor,
        'data_points': len(X),
        'features': X.shape[1],
        'feature_names': feature_names,
        'performance': {},
        'residuals': {},
        'feature_importance': {},
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        # Scale input data if scaler is provided
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Generate predictions
        y_pred = model.predict(X_scaled if scaler is not None else X)
        
        # Calculate performance metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        diagnostics['performance'] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mae': float(mae)
        }
        
        # Calculate residuals
        residuals = y - y_pred
        
        diagnostics['residuals'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q50': float(np.percentile(residuals, 50)),
            'q75': float(np.percentile(residuals, 75))
        }
        
        # Try to get feature importance if available
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importance = model.feature_importances_
                
                if feature_names is not None:
                    importance_dict = {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}
                else:
                    importance_dict = {f"feature_{i}": float(importance[i]) for i in range(len(importance))}
                
                diagnostics['feature_importance'] = importance_dict
                
            elif hasattr(model, 'feature_importances'):
                # For custom models with feature_importances method
                importance = model.feature_importances(X)
                
                if feature_names is not None:
                    importance_dict = {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}
                else:
                    importance_dict = {f"feature_{i}": float(importance[i]) for i in range(len(importance))}
                
                diagnostics['feature_importance'] = importance_dict
                
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_
                
                if feature_names is not None:
                    coef_dict = {feature_names[i]: float(coef[i]) for i in range(len(feature_names))}
                else:
                    coef_dict = {f"feature_{i}": float(coef[i]) for i in range(len(coef))}
                
                diagnostics['feature_importance'] = coef_dict
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error creating model diagnostics: {str(e)}")
        diagnostics['error'] = str(e)
    
    return diagnostics