import numpy as np
import os
import sys
import config
from data_worker import prepare_data
import pandas as pd
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import bloomberg
import logging
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


def visualize_a_model_prediction(country, tenor_name, country_code_mapping, tenor_data, pol_rat, cpi_inf, act_track, risk_rating, historical_forecasts, unemployment_rate):
    """
    Create a single visualization of what a model predicts for a specific country and tenor,
    based on available feature data, even if there's not enough data to train a proper model.
    
    Parameters:
        country: str - Country name
        tenor_name: str - Yield tenor name (e.g., 'yld_2yr')
        country_code_mapping: dict - Mapping from country names to country codes
        tenor_data: DataFrame - Yield data for the specified tenor
        pol_rat, cpi_inf, act_track, risk_rating: Feature DataFrames
        historical_forecasts: dict - Historical forecasts from forecast_generator
        debt_gdp: DataFrame - Debt to GDP ratio data (optional)
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
    try:
        x, y, feature_details = prepare_data(
            country=country,
            tenor_name=tenor_name,
            country_code_mapping=country_code_mapping,
            tenor=tenor_data,
            pol_rat=pol_rat,
            cpi_inf=cpi_inf,
            act_track=act_track,
            risk_rating=risk_rating,
            historical_forecasts=historical_forecasts,
            unemployment_rate=unemployment_rate
        )
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None
    
    if x.empty or len(y) == 0:
        print(f"Error: Insufficient data to create model")
        
        # Even though we can't train a model, let's still visualize what data we have
        plt.figure(figsize=(12, 6))
        try:
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
        except Exception as e:
            print(f"Error creating available data visualization: {e}")
        
        return None
    
    # We have overlapping data, so we can create a model
    print(f"Creating model with {len(x)} data points")
    
    try:
        # Split data into train (80%) and validation (10%) sets
        train_size = int(len(x) * 0.8)
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
        
        # Validation metrics
        val_rmse = None
        val_r2 = None
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
        if val_rmse is not None and val_r2 is not None:
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
                    try:
                        correlation = x[column].corr(y)
                        ax.annotate(f"Corr: {correlation:.4f}", 
                                   xy=(0.05, 0.85), xycoords='axes fraction',
                                   verticalalignment='top', horizontalalignment='left',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                   fontsize=9)
                    except Exception as e:
                        print(f"Error calculating correlation for {column}: {e}")
                
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
    except Exception as e:
        print(f"Error creating model prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print("--- END OF MODEL PREDICTION VISUALIZATION ---\n")
    
    return mlp, scaler

def create_performance_summary_visualization(results_summary):
    """
    Create visualization of model performance metrics.
    
    Parameters:
        results_summary: dict - Dictionary containing model results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Prepare data for performance visualization
        performance_data = []
        
        for country in results_summary['model_results']:
            for tenor in results_summary['model_results'][country]:
                result = results_summary['model_results'][country][tenor]
                
                if result['status'] == 'Success' and 'metrics' in result:
                    metrics = result['metrics']
                    
                    if 'test' in metrics:
                        performance_data.append({
                            'Country': country,
                            'Tenor': tenor,
                            'Test_RMSE': metrics['test'].get('rmse', float('nan')),
                            'Test_R2': metrics['test'].get('r2', float('nan'))
                        })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Create RMSE comparison chart
            plt.figure(figsize=(14, 10))
            
            # Group by tenor
            tenors = df['Tenor'].unique()
            groups = df.groupby('Tenor')
            
            positions = np.arange(len(df['Country'].unique()))
            width = 0.2
            
            # Create grouped bar chart
            for i, tenor in enumerate(tenors):
                tenor_data = groups.get_group(tenor)
                plt.bar(
                    positions + i*width, 
                    tenor_data['Test_RMSE'],
                    width=width,
                    label=f"{tenor}"
                )
            
            plt.xlabel('Country', fontsize=14)
            plt.ylabel('Test RMSE', fontsize=14)
            plt.title('Model Performance by Country and Tenor', fontsize=16)
            plt.xticks(positions + width, df['Country'].unique(), rotation=45, ha='right')
            plt.legend(title='Tenor')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            output_path = os.path.join(config.VISUALIZATION_DIR, 'model_performance_summary.png')
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Model performance visualization saved to '{output_path}'")
    
    except Exception as e:
        logger.error(f"Error creating performance visualization: {e}")

def create_model_summary_visualization(results_summary):
    """
    Create comprehensive visualization of model training results.
    
    Parameters:
        results_summary: dict - Dictionary containing model results
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating model summary visualization")
    
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
            'Failed - Exception': 'black',
            'Data Prepared': 'blue',
            'Not Started': 'gray'
        }
        
        bar_colors = [colors.get(s, 'blue') for s in statuses]
        
        plt.bar(statuses, counts, color=bar_colors)
        plt.title('Model Training Results', fontsize=16)
        plt.xlabel('Status', fontsize=14)
        plt.ylabel('Number of Models', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save to configured output directory
        output_path = os.path.join(config.VISUALIZATION_DIR, 'model_training_results.png')
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Model training results visualization saved to '{output_path}'")
        
        # Create additional visualizations for successful models
        if results_summary['success_count'] > 0:
            create_performance_summary_visualization(results_summary)
    
    except Exception as e:
        logger.error(f"Error creating model summary visualization: {e}")
