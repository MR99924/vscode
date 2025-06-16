import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

output_dir = r'C:\Users\MR99924\workspace\vscode\Projects\EM_RatesRF'

# Check if the directory exists
if not os.path.exists(output_dir):
    raise FileNotFoundError(f"The directory {output_dir} does not exist.")

print("Files in directory:", os.listdir(output_dir))  # Print directory contents

# Define the list of feature name endings and the corresponding standardized variable names
endings = ['_br', '_cl', '_mx', '_cz', '_pl', '_in', '_tr', '_kr', '_co', '_hu', '_il', '_za', '_my', '_ph', '_th', '_tw', '_pe', '_id', '_cn', '_us']
variables = ["h_inf", "c_inf", "neer", "pca_act", "u_rat", "ca_rat", "fb_rat", "embig_data", "cur_vol", "res_rat", "ppi_rat", "term_prem", "mkt_vol"]  # Add more variables as needed

def generate_feature_name_mapping(endings, variables):
    mapping = {}
    for ending in endings:
        for variable in variables:
            old_name = f"{variable}{ending}"
            new_name = variable
            mapping[old_name] = new_name
    return mapping

# Generate the feature name mapping
feature_name_mapping = generate_feature_name_mapping(endings, variables)

def update_feature_names(model, feature_name_mapping):
    # Get the current feature names
    current_feature_names = model.feature_names_in_
    
    # Create a new list of feature names based on the mapping
    new_feature_names = [feature_name_mapping.get(name, name) for name in current_feature_names]
    
    # Update the model's feature names
    model.feature_names_in_ = new_feature_names
    return model

# Function to calculate deviation score
def calculate_deviation_score(good_predictions, bad_predictions):
    differences = abs(good_predictions - bad_predictions)
    directional_consistency = (np.sign(good_predictions - 1) == np.sign(bad_predictions - 1)).astype(int)
    penalty = np.where(directional_consistency == 0, differences * 2, differences)
    
    # Separate hawkish and dovish deviations
    hawkish_deviation = np.where((good_predictions > bad_predictions) & (directional_consistency == 0), differences, 0)
    dovish_deviation = np.where((good_predictions < bad_predictions) & (directional_consistency == 0), differences, 0)
    
    overall_score = np.sum(penalty)
    hawkish_score = np.sum(hawkish_deviation)
    dovish_score = np.sum(dovish_deviation)

    # Debugging output
    print("Differences:", differences)
    print("Directional consistency:", directional_consistency)
    print("Penalty:", penalty)
    print("Hawkish deviation:", hawkish_deviation)
    print("Dovish deviation:", dovish_deviation)
    
    return overall_score, hawkish_score, dovish_score

# Load and update the US model
us_rf_model = joblib.load(os.path.join(output_dir, 'US_rf_model.joblib'))
us_rf_model = update_feature_names(us_rf_model, feature_name_mapping)
joblib.dump(us_rf_model, os.path.join(output_dir, 'US_rf_model_updated.joblib'))

# Load and update other country models
other_country_models = {
    'Brazil': joblib.load(os.path.join(output_dir, 'Brazil_rf_model.joblib')),
    'Chile': joblib.load(os.path.join(output_dir, 'Chile_rf_model.joblib')),
    'China': joblib.load(os.path.join(output_dir, 'China_rf_model.joblib')),
    'Colombia': joblib.load(os.path.join(output_dir, 'Colombia_rf_model.joblib')),
    'Czechia': joblib.load(os.path.join(output_dir, 'Czechia_rf_model.joblib')),
    'Hungary': joblib.load(os.path.join(output_dir, 'Hungary_rf_model.joblib')),
    'India': joblib.load(os.path.join(output_dir, 'India_rf_model.joblib')),
    'Indonesia': joblib.load(os.path.join(output_dir, 'Indonesia_rf_model.joblib')),
    'Israel': joblib.load(os.path.join(output_dir, 'Israel_rf_model.joblib')),
    'Korea': joblib.load(os.path.join(output_dir, 'Korea_rf_model.joblib')),
    'Malaysia': joblib.load(os.path.join(output_dir, 'Malaysia_rf_model.joblib')),
    'Mexico': joblib.load(os.path.join(output_dir, 'Mexico_rf_model.joblib')),
    'Philippines': joblib.load(os.path.join(output_dir, 'Philippines_rf_model.joblib')),
    'Peru': joblib.load(os.path.join(output_dir, 'Peru_rf_model.joblib')),
    'Poland': joblib.load(os.path.join(output_dir, 'Poland_rf_model.joblib')),
    'South Africa': joblib.load(os.path.join(output_dir, 'South Africa_rf_model.joblib')),
    'Taiwan': joblib.load(os.path.join(output_dir, 'Taiwan_rf_model.joblib')),
    'Thailand': joblib.load(os.path.join(output_dir, 'Thailand_rf_model.joblib')),
    'Turkey': joblib.load(os.path.join(output_dir, 'Turkey_rf_model.joblib')),
}

# Update feature names in each country model and save the updated models
for country, model in other_country_models.items():
    updated_model = update_feature_names(model, feature_name_mapping)
    joblib.dump(updated_model, os.path.join(output_dir, f'{country}_rf_model_updated.joblib'))

# Load the collected sample of testing data
testing_data_path = os.path.join(output_dir, 'collected_X_test_2.csv')
print("Testing data file path:", testing_data_path)  # Print the testing data file path
testing_data = pd.read_csv(testing_data_path)


# Check for NaN, infinity or large values in the input data and handle them appropriately
if testing_data.isnull().values.any():
    print("Warning: NaN values found in the input data. Filling NaNs with column mean.")
    testing_data.fillna(testing_data.mean(), inplace=True)

if np.isinf(testing_data.values).any():
    print("Warning: Infinity values found in the input data. Replacing infinities with large finite numbers.")
    testing_data.replace([np.inf, -np.inf], np.nanmax(testing_data.values[np.isfinite(testing_data.values)]), inplace=True)

if (testing_data.values > np.finfo(np.float32).max).any():
    print("Warning: Large values found in the input data. Clipping values to float32 max.")
    testing_data.clip(upper=np.finfo(np.float32).max, inplace=True)

# Function to add missing variables to the testing data
def add_missing_variables(testing_data, model):
    missing_features = set(model.feature_names_in_) - set(testing_data.columns)
    for feature in missing_features:
        testing_data[feature] = 0  # You can use other default values if needed
    return testing_data

# Ensure the testing data uses the new standardized feature names
X_test = testing_data

# Filter the input data to include only the features expected by the US model
X_test_filtered = X_test[us_rf_model.feature_names_in_]

# Print statements to debug feature mismatch issue
print("Shape of X_test_filtered:", X_test_filtered.shape)
print("Columns in X_test_filtered:", X_test_filtered.columns.tolist())
print("Expected number of features by US model:", us_rf_model.n_features_in_)
print("Feature names expected by US model:", us_rf_model.feature_names_in_)

# Generate predictions using the updated US model (good student)
us_rf_model = joblib.load(os.path.join(output_dir, 'US_rf_model_updated.joblib'))
us_predictions = us_rf_model.predict(X_test_filtered)

# Initialize dictionaries to store the deviation scores for each country model
deviation_scores = {}
hawkish_scores = {}
dovish_scores = {}

# Calculate deviation scores for each updated country model
for country in other_country_models.keys():
    model = joblib.load(os.path.join(output_dir, f'{country}_rf_model_updated.joblib'))
    
    # Add missing variables to the testing data
    X_test = add_missing_variables(X_test, model)
    
    # Filter the input data to include only the features expected by the current model
    X_test_filtered = X_test[model.feature_names_in_]
    print(X_test_filtered)

    
    # Print statements to debug feature mismatch issue for each country model
    print(f"Shape of X_test_filtered for {country} model:", X_test_filtered.shape)
    print(f"Columns in X_test_filtered for {country} model:", X_test_filtered.columns.tolist())
    print(f"Expected number of features by {country} model:", model.n_features_in_)
    print(f"Feature names expected by {country} model:", model.feature_names_in_)
    
    country_predictions = model.predict(X_test_filtered)
    overall_score, hawkish_score, dovish_score = calculate_deviation_score(us_predictions, country_predictions)
    deviation_scores[country] = overall_score
    hawkish_scores[country] = hawkish_score
    dovish_scores[country] = dovish_score

# Print the deviation scores
for country, score in deviation_scores.items():
    print(f"Deviation score for {country}: {score}")
for country, score in hawkish_scores.items():
    print(f"Hawkish score for {country}: {score}")
for country, score in dovish_scores.items():
    print(f"Dovish score for {country}: {score}")


# Plot the deviation scores across countries
def plot_deviation_scores(deviation_scores, hawkish_scores, dovish_scores):
    countries = list(deviation_scores.keys())
    overall_scores = list(deviation_scores.values())
    hawkish_scores_list = list(hawkish_scores.values())
    dovish_scores_list = list(dovish_scores.values())

    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(len(countries))

    #plt.bar(index, overall_scores, bar_width, label='Overall', color='skyblue')
    plt.bar(index + bar_width, hawkish_scores_list, bar_width, label='Hawkish', color='red')
    plt.bar(index + 2 * bar_width, dovish_scores_list, bar_width, label='Dovish', color='blue')

    plt.xlabel('Country')
    plt.ylabel('Deviation Score')
    plt.title('Hawkish/Dovish Leanings versus the Fed')
    plt.xticks(index + bar_width, countries, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the deviation scores
plot_deviation_scores(deviation_scores, hawkish_scores, dovish_scores)

# Generate predictions using the Brazil model
brazil_model = joblib.load(os.path.join(output_dir, 'Brazil_rf_model_updated.joblib'))
X_test_filtered_brazil = X_test[brazil_model.feature_names_in_]
brazil_predictions = brazil_model.predict(X_test_filtered_brazil)

# Compare predictions
comparison_df = pd.DataFrame({
    'US Predictions': us_predictions,
    'Brazil Predictions': brazil_predictions
})
print(comparison_df.head())  # Print the first few rows for comparison

# Check the input data for the Brazil model
print("Shape of X_test_filtered for Brazil model:", X_test_filtered_brazil.shape)
print("Columns in X_test_filtered for Brazil model:", X_test_filtered_brazil.columns.tolist())
print("Expected number of features by Brazil model:", brazil_model.n_features_in_)
print("Feature names expected by Brazil model:", brazil_model.feature_names_in_)

# Check if the Brazil model has been updated correctly
print("Feature names in Brazil model:", brazil_model.feature_names_in_)