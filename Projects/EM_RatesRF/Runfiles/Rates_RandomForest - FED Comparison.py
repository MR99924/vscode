# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:22:46 2024

@author: MR99924
"""
import sys
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import os
import pandas as pd
import numpy as np
import math
from joblib import dump, load
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score,  precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from macrobond import Macrobond
import bloomberg
import traceback



mb = Macrobond()
label_encoder = LabelEncoder()

# We now move to the bloomberg function we have sorted out.

def get_bloomberg_date(tickers, date_from, date_to, field="PX_LAST", periodicity="DAILY"):
   bbg = bloomberg.Bloomberg()
   df = bbg.historicalRequest(tickers,
                              field,
                              date_from,
                              date_to,
                              periodicitySelection=periodicity,
                              nonTradingDayFillOption="ALL_CALENDAR_DAYS",
                              nonTradingDayFillMethod="PREVIOUS_VALUE",
                              )
   df = pd.pivot_table(df,
                       values='bbergvalue',
                       index=['bbergdate'],
                       columns=['bbergsymbol'],
                      aggfunc=np.max,
                      )
   df = df[tickers]
   return df

def apply_backfill_after_first_observation(df):
    for column in df.columns:
        first_valid_index = df[column].first_valid_index()
        if first_valid_index is not None:
            df.loc[first_valid_index:, column] = df.loc[first_valid_index:, column].bfill()
    return df


# Let's start by importing our data for each of the series we are interested in...

# We begin by looking at the policy rates we are trying to pull (Bloomberg)

tickers = ["FDTR Index"]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

rate_data = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

rate_data = rate_data.rename(columns={"FDTR Index": "rate_us"})

rate_data_pop = rate_data.diff()

print(rate_data_pop.head())

# Now we move on to inflation (Macrobond)

tickers = ["uscpi"]

h_inf = mb.FetchSeries(tickers)

new_column_names = {
        "uscpi": "h_inf_us"
        }

h_inf = h_inf.rename(columns = new_column_names)
h_inf_3mma = h_inf.rolling(window=3).mean()
h_inf_qoq_3mma = h_inf_3mma.pct_change(periods=3) * 100
h_inf_qoq_3mma_d = h_inf_qoq_3mma.resample('D').ffill()
h_inf_qoq_3mma_d = apply_backfill_after_first_observation(h_inf_qoq_3mma_d)
h_inf_qoq_3mma_d_lggd = h_inf_qoq_3mma_d.shift(30)

# Now we move on to core inflation (Macrobond)

tickers = ["usnaac1186"]

c_inf = mb.FetchSeries(tickers)

new_column_names = {"usnaac1186": "c_inf_us"
           }

c_inf = c_inf.rename(columns = new_column_names)
c_inf_3mma = c_inf.rolling(window=3).mean()
c_inf_qoq_3mma = c_inf_3mma.pct_change(periods=3) * 100
c_inf_qoq_3mma_d = c_inf_qoq_3mma.resample('D').ffill()
c_inf_qoq_3mma_d = apply_backfill_after_first_observation(c_inf_qoq_3mma_d)
c_inf_qoq_3mma_d_lggd = c_inf_qoq_3mma_d.shift(30)

# Now we move on to REER (Bloomberg)

tickers = ["JBDNUSD  Index"]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

neer_data = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

neer_data = neer_data.rename(columns={"JBDNUSD  Index": "neer_us"})

neer_data = neer_data.rename(columns = new_column_names)
neer_data_3mma = neer_data.rolling(window=90).mean()
neer_data_qoq_3mma = neer_data_3mma.pct_change(periods=90) * 100
neer_data_qoq_3mma_d = neer_data_qoq_3mma.resample('D').ffill()
neer_data_qoq_3mma_d = apply_backfill_after_first_observation(neer_data_qoq_3mma_d)

# Now we move to the activity section (this is a generalised pca of a basket of metrics - kept in-house)

tickers = ["GSUSCAI Index"]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

pca_act = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

new_column_names = {"GSUSCAI Index" : "pca_act_us"}

pca_act = pca_act.rename(columns = new_column_names)
pca_act_d = pca_act.resample('D').ffill()
pca_act_d = apply_backfill_after_first_observation(pca_act_d)
pca_act_d_lggd = pca_act_d.shift(30)

column_names = pca_act_d_lggd.columns.tolist()
print(column_names)
print(pca_act_d_lggd.index)
print(pca_act_d_lggd.columns)

# Maybe for old times' sake, in reference to inflation debates of old (and to extend the viable series over which we can train) we can include unemployemnt (Macrobond)

tickers = ["uslama1849"]

u_rat = mb.FetchSeries(tickers)

new_column_names = {"uslama1849" : "u_rat_us"}

u_rat = u_rat.rename(columns = new_column_names)
u_rat_12mma = u_rat.rolling(window=12).mean()
u_rat_12mma_sad = u_rat_12mma.diff(periods=6)
u_rat_12mma_sad_d = u_rat_12mma_sad.resample('D').ffill()
u_rat_12mma_sad_d = apply_backfill_after_first_observation(u_rat_12mma_sad_d)
u_rat_12mma_sad_d_lggd = u_rat_12mma_sad_d.shift(30)
print(u_rat_12mma_sad_d_lggd.head())

# Now we move to the external balance content (macrobond)

tickers = ["ih:mb:com:ca_usa:ca_usa"]

ca_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:ca_usa:ca_usa": "ca_rat_us"}

ca_rat = ca_rat.rename(columns = new_column_names)
ca_rat_12mma = ca_rat.rolling(window=12).mean()
ca_rat_12mma_sad = ca_rat_12mma.diff(periods=6)
ca_rat_12mma_sad_d = ca_rat_12mma_sad.resample('D').ffill()
ca_rat_12mma_sad_d = apply_backfill_after_first_observation(ca_rat_12mma_sad_d)
ca_rat_12mma_sad_d_lggd = ca_rat_12mma_sad_d.shift(60)
print(ca_rat_12mma_sad_d_lggd.head())


# Now we move to fiscal (macrobond)

tickers = ["ih:mb:com:fis_cg_usa:fis_cg_usa"]

fb_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:fis_cg_usa:fis_cg_usa" : "fb_rat_us"}

fb_rat = fb_rat.rename(columns = new_column_names)
fb_rat_12mma = fb_rat.rolling(window=12).mean()
fb_rat_12mma_sad = fb_rat_12mma.diff(periods=6)
fb_rat_12mma_sad_d = fb_rat_12mma_sad.resample('D').ffill()
fb_rat_12mma_sad_d = apply_backfill_after_first_observation(fb_rat_12mma_sad_d)
fb_rat_12mma_sad_d_lggd = fb_rat_12mma_sad_d.shift(60)
print(fb_rat_12mma_sad_d_lggd.head())


# Now we move on to the EMBIG credit spread data - organised by particular countries


tickers = ["USYC2Y10 Index"]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

embig_data = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

embig_data = embig_data.rename(columns={"USYC2Y10 Index" : "term_prem_us"})

print(embig_data.head())

# Now we move to currency volatility (Bloomberg)

tickers = ["VIX Index"]

dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

mkt_vol = get_bloomberg_date(tickers,dt_from,dt_to, field = 'PX_LAST', periodicity = 'DAILY')

mkt_vol = mkt_vol.rename(columns={"VIX Index" : "mkt_vol_us"})

print(mkt_vol.head())

# Now we move to external reserves as a share of short term debt (Macrobond)

tickers = ["ih:mb:com:reserve_metric_usa:reserve_metric_usa"]

res_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:reserve_metric_usa:reserve_metric_usa" : "res_rat_us"}

res_rat = res_rat.rename(columns = new_column_names)
res_rat_12mma = res_rat.rolling(window=12).mean()
res_rat_12mma_sad = res_rat_12mma.diff(periods=6)
res_rat_12mma_sad_d = res_rat_12mma_sad.resample('D').ffill()
res_rat_12mma_sad_d = apply_backfill_after_first_observation(res_rat_12mma_sad_d)
print(res_rat_12mma_sad_d.head())

# Lastly, we go to PPI inflation, we can either keep or remove, depending on personal preference (macrobond)

tickers = ["uspric0011"]

ppi_rat = mb.FetchSeries(tickers)

new_column_names = {"uspric0011" : "ppi_rat_us"}

ppi_rat = ppi_rat.rename(columns = new_column_names)
ppi_rat_3mma = ppi_rat.rolling(window=3).mean()
ppi_rat_qoq_3mma = ppi_rat_3mma.pct_change(periods=3) * 100
ppi_rat_qoq_3mma_d = ppi_rat_qoq_3mma.resample('D').ffill()
ppi_rat_qoq_3mma_d = apply_backfill_after_first_observation(ppi_rat_qoq_3mma_d)
ppi_rat_qoq_3mma_d_lggd = ppi_rat_qoq_3mma_d.shift(30)
print(ppi_rat_qoq_3mma_d_lggd.head())

# Ooookay, so we have a full suite of variables. In decending order, we have the following:
# EM Policy rates
# Headline inflation
# Core inflation
# REER
# DM policy rates
# Activity data
# Unemployment rates
# External balances (% of GDP)
# Fiscal balances (% of GDP)
# EMBIG credit spreads
# Currency volatility
# Reserves as a share of ST debt
# PPI inflation  

# What is left to do?
# We need to train a model based on the increments we have generated, using the backup data we have available.
# Train the random forest model to predict movements in the policy rate based on series available.
# Do so for each country individually - run tests to guague the predictive power of the model 
# Then do so for the EM complex as a whole  - run tests to guague the predictive power of the model
# Then do so for each DM country - run tests to guague the predictive power of the model
# Then 'port' the DM 'brain' into various EM countries  - give each a score based on how well they 'match' the DM policy rate (orthodoxy measure)

file_path = r'C:\\Users\\MR99924\\workspace\\vscode\\Projects\\EM_RatesRF\\CBDecisions_US.xlsx'
rate_decisions = pd.read_excel(file_path, sheet_name='RateDecisions', engine='openpyxl', usecols=range(1))
print(rate_decisions.head())

rate_decisions = pd.DataFrame(rate_decisions)
for country in rate_decisions.columns:
    rate_decisions[country] = pd.to_datetime(rate_decisions[country])

# Dictionary to store results for each country
results_dict = {}

country_code_mapping = {
    'United States': 'us'}

def round_to_nearest_25bps_except_10bps(rate_change):
    if math.isnan(rate_change):
        return float('nan')
    if abs(rate_change) < 0.10:
        return 0
    elif abs(rate_change) == 0.10:
        return 0.10
    return round(rate_change * 4) / 4

# Iterate over each country in the decision dates data

# Print the country_code_mapping dictionary to verify its contents
print(country_code_mapping)

# Ensure 'US' is in the dictionary
if 'US' not in country_code_mapping:
    print("Key 'US' is missing from country_code_mapping")
    country_code_mapping['US'] = 'us'  # Add the correct mapping for 'US'
else:
    print("Key 'US' is present in country_code_mapping")

for country in rate_decisions.columns:
    decision_dates = rate_decisions[country].dropna()
    decision_dates = decision_dates[decision_dates >= pd.Timestamp('1990-01-01')]
    print(decision_dates)
    
    # Get the policy rate data for the country using the correct code from the mapping
    policy_rate_column = 'rate_us'  # Adjusted to match the renamed columns
    policy_rates = rate_data[policy_rate_column]
    
    # Extract the policy rates on the decision dates
    rates_on_decision_days = policy_rates.loc[decision_dates]
    print(rates_on_decision_days)
    
    # Calculate the changes in policy rates on the decision dates
    rate_changes = rates_on_decision_days.diff()

    # Round the rate changes to the nearest 25bps
    rounded_rate_changes = rate_changes.apply(round_to_nearest_25bps_except_10bps)
    
    # Store the results in a DataFrame for the country
    target_df = pd.DataFrame({
        f'decision_date_{country_code_mapping[country]}': decision_dates.values,
        f'policy_rate_change_{country_code_mapping[country]}': rounded_rate_changes.values
    })
    # print(f"The target data for {country} have been successfully exported to '{country}_target.csv'.")
    
    # Add the DataFrame to the dictionary
    results_dict[country] = target_df
    print(target_df)

bins = [-float('inf'), -0.01, 0.01, float('inf')]
labels = ["Down", "Hold", "Up"]
label_mappings = {}

# Prepare features and target variable for the model
for country in rate_decisions.columns:
    print(country)
    
    # Get the country code from the mapping
    country_code = country_code_mapping[country]
    
    # Filter features to include only columns ending with the country code
    features = pd.concat([h_inf_qoq_3mma_d_lggd.filter(regex=f'_{country_code}$'),
                          c_inf_qoq_3mma_d_lggd.filter(regex=f'_{country_code}$'),
                          neer_data_qoq_3mma_d.filter(regex=f'_{country_code}$'),
                          pca_act_d_lggd.filter(regex=f'_{country_code}$'),
                          u_rat_12mma_sad_d_lggd.filter(regex=f'_{country_code}$'),
                          ca_rat_12mma_sad_d_lggd.filter(regex=f'_{country_code}$'),
                          fb_rat_12mma_sad_d_lggd.filter(regex=f'_{country_code}$'),
                          embig_data.filter(regex=f'_{country_code}$'),
                          mkt_vol.filter(regex=f'_{country_code}$'),
                          res_rat_12mma_sad_d.filter(regex=f'_{country_code}$'),
                          ppi_rat_qoq_3mma_d_lggd.filter(regex=f'_{country_code}$'),
                          ], axis=1)
    
    features.index = pd.to_datetime(features.index)
    features = features.loc[features.index >= '1990-01-01']
    print(features.columns)
     
    # Assuming target_df contains the necessary data
    target_df = results_dict[country]
    target = target_df
    print(target)
    
    try:
        target.set_index(f'decision_date_{country_code}', inplace=True)
        target.index = pd.to_datetime(target.index)
    except Exception as e:
        print(f"An error occurredin setting the index for {country}: {e}. Proceeding to the next one.")

    # Drop rows with NaN values in features or target
    features = features.dropna()
    target = target.dropna()

    # Apply binning - this is being done correctly
    target[f'policy_rate_change_{country_code}'] = pd.cut(target[f'policy_rate_change_{country_code}'], bins=bins, labels=labels, include_lowest=True)

    # This is the categorisation function, is this being applied correctly?
    try:
        #  Fit and transform the label encoder for each country separately
        target[f'policy_rate_change_{country_code}'] = label_encoder.fit_transform(target[f'policy_rate_change_{country_code}'])
    
        # Create a mapping from numerical labels to bin labels for each country
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
        # Store the label mapping in the separate dictionary
        label_mappings[country] = label_mapping
    
    except Exception as e:
        print(f"An error occurred in producing the target variable for {country}: {e}. We proceed to the next one.")

    # Replace infinite values with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    target.replace([np.inf, -np.inf], np.nan, inplace=True)

    #  Drop rows with NaN values in features or target
    features.dropna(inplace=True)
    target.dropna(inplace=True)
    
    # Ensure features and target have matching indices after dropping NaNs
    # Okay, there's a broader question here around what data points we will include and which we won't, and under what circumstances
    
    common_indices = features.index.intersection(target.index)
    features = features.loc[common_indices]
    target = target.loc[common_indices]
    features = features[~features.index.duplicated(keep='first')]
    target = target[~target.index.duplicated(keep='first')]
    
    base_target = target.values.ravel()
    print(base_target)

    base_rf_model = RandomForestClassifier(random_state=42, oob_score=True)

    previous_accuracy = 0
    for n_estimators in [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]:
        base_rf_model.set_params(n_estimators=n_estimators)
        base_rf_model.fit(features, base_target)
    
    model_filename = f'{country}_rf_model.joblib'
    dump(base_rf_model, model_filename)
        

    print(f"Length of features: {len(features)}")
    print(f"Length of target: {len(target)}")
    if len(features) != len(target):
        print(f"Skipping {country} due to inconsistent number of samples.")
        continue

    # Split the data into training and testing sets (80% train, 20% test)
    # It also seems like this is being done in a kind of random way, so we aren't predicting a model off a continuous chunk of the sample. 
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Print the shapes of the resulting arrays to verify the split
    print(f"{country} - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    if X_train.empty:
        print(f"No features training data available for {country}. Skipping this country.")
        continue
    if y_train.empty:
        print(f"No target training data available for {country}. Skipping this country.")
        continue
    
    # Reshape the target variable
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    # Initialize and train the Random Forest model - maybe we can adjust this so it's like a "smart estimator" finder
    rf_model = RandomForestClassifier(random_state=42, oob_score=True)

    # Train the initial Random Forest model
    initial_rf_model = RandomForestClassifier(random_state=42, oob_score=True)
    initial_rf_model.fit(X_train, y_train)

    # Get feature importances
    importances = initial_rf_model.feature_importances_

    n_top_features = 11  # Adjust this number based on your requirement
    indices = np.argsort(importances)[-n_top_features:]
    selected_features = X_train.columns[indices]

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    previous_accuracy = 0
    for n_estimators in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        rf_model.set_params(n_estimators=n_estimators)
        rf_model.fit(X_train_selected, y_train)
        y_val_pred = rf_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_val_pred)
        print(f'n_estimators: {n_estimators}, Validation Accuracy: {accuracy}')
        if accuracy <= previous_accuracy:
            break
        previous_accuracy = accuracy

    # Print the OOB error
    oob_error = 1 - rf_model.oob_score_
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test_selected)

   # Create a mapping from numerical labels to bin labels
    label_mapping = label_mappings[country]

    # Map the y_test and y_pred values back to the bin labels
    y_test_labels = pd.Series(y_test).map(label_mapping)
    y_pred_labels = pd.Series(y_pred).map(label_mapping)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels)
    validation_error = 1 - accuracy

    # Additional metrics for model quality
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

    
    
    # Store the results in a dictionary for each country
    results_dict[country] = {
        'model': rf_model,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'X_test_selected': X_test_selected,
        'y_test_labels': y_test_labels,
        'y_pred_labels': y_pred_labels,
        'oob_error' : oob_error,
        'valid_error': validation_error}

    print(results_dict)

output_dir = r'C:\Users\MR99924\workspace\vscode\Projects\EM_RatesRF'

# Save results to CSV files for each country
dfs = {}

for country, results in results_dict.items():
    try:
        if 'X_test_selected' not in results:
            raise KeyError('X_test key is missing')
        df_results = pd.DataFrame({
            'X_test_selected': results['X_test_selected'].index,
            'y_test_labels': results['y_test_labels'],
            'y_pred_labels': results['y_pred_labels'],
            'accuracy': [results['accuracy']] * len(results['y_test_labels']),
            'precision': [results['precision']] * len(results['y_test_labels']),
            'recall': [results['recall']] * len(results['y_test_labels']),
            'f1_score': [results['f1_score']] * len(results['y_test_labels']),
            'oob_error' :[results['oob_error']] * len(results['y_test_labels']),
            'valid_error' :[results['valid_error']] * len(results['y_test_labels'])        })
        dfs[country] = df_results
    except KeyError as e:
        print(f"KeyError for {country}: {e}. Proceeding to the next one.")
    except Exception as e:
        print(f"An error occurred for {country}: {e}. Proceeding to the next one.")
        traceback.print_exc()

# Save all DataFrames to a single Excel file with different sheets
with pd.ExcelWriter(os.path.join(output_dir, 'US_rf_model_results.xlsx')) as writer:
    for country, df in dfs.items():
        df.to_excel(writer, sheet_name=country, index=False)

print("The results have been successfully saved to an Excel file with different sheets.")


# Appendix:
# 1) For funzies, we can inspect the frequency distributions of each of the historical data periods.

# def plot_frequency_distribution(rate_changes, country):
#     plt.figure(figsize=(10, 6))
#     plt.hist(rate_changes, bins=50, edgecolor='k', alpha=0.7)
#     plt.title(f'Frequency Distribution of Rate Changes for {country}')
#     plt.xlabel('Rate Change')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()

# # Plot the frequency distribution for each country
# for country, df in results_dict.items():
#     rate_changes = df[f'{country}_policy_rate_change'].dropna()
#     rounded_rate_changes = rate_changes.apply(round_to_nearest_25bps_except_10bps)  # Apply the rounding function
#     plot_frequency_distribution(rounded_rate_changes, country)  # Use the rounded rate changes


# with pd.ExcelWriter('frequency_distribution_data.xlsx') as writer:
#     for country, df in results_dict.items():
#         rate_changes = df[f'{country}_policy_rate_change'].dropna()
#         rounded_rate_changes = rate_changes.apply(round_to_nearest_25bps_except_10bps)  # Apply the rounding function
#         rounded_rate_changes.to_frame(name=f'{country}_rounded_rate_change').to_excel(writer, sheet_name=country)

# print("The data used in the frequency distributions has been exported to 'frequency_distribution_data.xlsx'.")

# You need to produce a series of good tests to highlight the worthiness of the model, then expand it to show that it can be done to predict "next meeting" decisions.

def plot_precision_recall_curve(model, X_test, y_test, labels, country):
    y_pred_proba = model.predict_proba(X_test)
    
    # Find unique labels in y_test
    unique_labels = np.unique(y_test)
    filled_labels = [label for label in labels if label in unique_labels]
    filled_indices = [labels.index(label) for label in filled_labels]
    
    # Adjust y_pred_proba to include only columns for filled labels
    y_pred_proba_filled = y_pred_proba[:, filled_indices]
    num_filled_classes = len(filled_labels)
    
    if y_pred_proba_filled.shape[1] != num_filled_classes:
        print(f"An error occurred for {country}: Number of classes in y_pred_proba ({y_pred_proba_filled.shape[1]}) does not match number of filled labels ({num_filled_classes}). Proceeding to the next one.")
        return

    print(f"y_test: {y_test}")  # Debugging information
    print(f"unique_labels: {unique_labels}")  # Debugging information
    print(f"labels: {labels}")  # Debugging information
    print(f"filled_labels: {filled_labels}")  # Debugging information
    print(f"filled_indices: {filled_indices}")  # Debugging information
    print(f"y_pred_proba_filled: {y_pred_proba_filled}")  # Debugging information

    precision = {}
    recall = {}
    average_precision = {}

    for i in range(num_filled_classes):
        if np.sum(y_test == filled_labels[i]) == 0:
            print(f"No positive samples in y_true for class {filled_labels[i]}. Skipping this class.")
            continue
        precision[i], recall[i], _ = precision_recall_curve(y_test == filled_labels[i], y_pred_proba_filled[:, i])
        average_precision[i] = average_precision_score(y_test == filled_labels[i], y_pred_proba_filled[:, i])
        print(f"Class {filled_labels[i]}: precision={precision[i]}, recall={recall[i]}, average_precision={average_precision[i]}")  # Debugging information

    plt.figure(figsize=(10, 6))
    for i in range(num_filled_classes):
        if i in precision and len(precision[i]) > 0:
            plt.plot(recall[i], precision[i], label=f'Precision-Recall curve (AP = {average_precision[i]:.2f}) for class {filled_labels[i]}')
        else:
            print(f"No data to plot for class {filled_labels[i]}")  # Debugging information

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{country}: Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()


def plot_feature_importance(model, features, country):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'{country}: Feature Importance')
    plt.show()

def plot_calibration_curve(model, X_test, y_test, country, n_bins=10):
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title(f'{country}: Calibration Curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_learning_curve(model, X, y, country, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.title(f'Learning Curve for {country}')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_cumulative_gain_lift(model, X_test, y_test, country):
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Sort the predicted probabilities and true labels by the predicted probabilities in descending order
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_pred_proba_sorted = y_pred_proba[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    
    # Calculate cumulative gains and lift
    cumulative_gains = np.cumsum(y_test_sorted) / np.sum(y_test_sorted)
    lift = cumulative_gains / (np.arange(1, len(y_test_sorted) + 1) / len(y_test_sorted))
    
    # Plot cumulative gain chart
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(y_test_sorted) + 1) / len(y_test_sorted), cumulative_gains, marker='o', label='Cumulative Gain')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline')
    
    plt.xlabel('Percentage of Sample')
    plt.ylabel('Cumulative Gain')
    plt.title(f'{country}: Cumulative Gain Chart')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
    # Plot lift chart
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(y_test_sorted) + 1) / len(y_test_sorted), lift, marker='o', label='Lift')
    
    plt.xlabel('Percentage of Sample')
    plt.ylabel('Lift')
    plt.title(f'{country}: Lift Chart')
    plt.legend(loc='best')
    plt.grid()


def plot_roc_curve(model, X_test, y_test, labels, country):
    y_pred_proba = model.predict_proba(X_test)
    
    # Find unique labels in y_test
    unique_labels = np.unique(y_test)
    filled_labels = [label for label in labels if label in unique_labels]
    filled_indices = [labels.index(label) for label in filled_labels]
    
    # Adjust y_pred_proba to include only columns for filled labels
    y_pred_proba_filled = y_pred_proba[:, filled_indices]
    num_filled_classes = len(filled_labels)
    
    if y_pred_proba_filled.shape[1] != num_filled_classes:
        print(f"An error occurred for {country}: Number of classes in y_pred_proba ({y_pred_proba_filled.shape[1]}) does not match number of filled labels ({num_filled_classes}). Proceeding to the next one.")
        return

    print(f"y_test: {y_test}")  # Debugging information
    print(f"unique_labels: {unique_labels}")  # Debugging information
    print(f"labels: {labels}")  # Debugging information
    print(f"filled_labels: {filled_labels}")  # Debugging information
    print(f"filled_indices: {filled_indices}")  # Debugging information
    print(f"y_pred_proba_filled: {y_pred_proba_filled}")  # Debugging information

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_filled_classes):
        if np.sum(y_test == filled_labels[i]) == 0:
            print(f"No positive samples in y_true for class {filled_labels[i]}. Skipping this class.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test == filled_labels[i], y_pred_proba_filled[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"Class {filled_labels[i]}: fpr={fpr[i]}, tpr={tpr[i]}, roc_auc={roc_auc[i]}")  # Debugging information

    plt.figure(figsize=(10, 6))
    for i in range(num_filled_classes):
        if i in fpr and len(fpr[i]) > 0:
            plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {filled_labels[i]}')
        else:
            print(f"No data to plot for class {filled_labels[i]}")  # Debugging information

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{country}: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

## Iterate over each country in the results_dict
for country, results in results_dict.items():
    print(country)
    try:
        # Extract the model and test data
        model = results['model']
        X_test = results['X_test_selected']
        y_test = results['y_test_labels']

        # Ensure X_test and y_test have consistent lengths
        if len(X_test) != len(y_test):
            print(f"Inconsistent number of samples for {country}: X_test has {len(X_test)} samples, but y_test has {len(y_test)} samples. Skipping this country.")
            continue
        
        # Plot cumulative gain and life charts - this one  also produces an error
        #plot_cumulative_gain_lift(model, X_test, y_test, country)
        
        # Plot Calibration curves - this one produces an error
        #plot_calibration_curve(model, X_test, y_test, country)
        
        #Plot Precision-Recall curves
        plot_precision_recall_curve(model, X_test, y_test, labels, country)

        #Plot learning curves
        plot_learning_curve(model, X_test, y_test, country)

        # Plot feature importance
        plot_feature_importance(model, X_test, country)
        
        # Plot ROC curve
        plot_roc_curve(model, X_test, y_test, labels, country)
        
    except KeyError as e:
        print(f"KeyError for {country}: {e}. Proceeding to the next one.")
    except Exception as e:
        print(f"An error occurred for {country}: {e}. Proceeding to the next one.")

# # Now we move on to out of sample prediction for the future 

# Function to calculate implied probabilities for each bin
def calculate_implied_probabilities(model, X_test):
    # Get the predicted probabilities for each class
    predicted_probabilities = model.predict_proba(X_test)
    
    # Adjust labels if the number of classes is less than the number of bins
    if predicted_probabilities.shape[1] < len(labels):
        adjusted_labels = labels[:predicted_probabilities.shape[1]]
    else:
        adjusted_labels = labels
    
    # Create a DataFrame to store the probabilities with corresponding bins
    probabilities_df = pd.DataFrame(predicted_probabilities, columns=adjusted_labels)
    
    return probabilities_df

# Extend feature datasets to the current date
def extend_features_to_today(features):
    # Get the latest date in the existing data
    latest_date = pd.to_datetime(features.index.max())
    print(latest_date)
    
    # Generate a date range from the latest date to today
    today = pd.to_datetime('today').normalize()
    print(today)
    new_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), end=today, freq='D')
    print(new_dates)
    
    # Extend the feature dataset
    if not new_dates.empty:
        # Create a DataFrame with the new dates and fill with NaNs
        new_data = pd.DataFrame(index=new_dates, columns=features.columns)
        
        # Fill with the last available value
        new_data.fillna(method='ffill', inplace=True)
        
        # Append the new data to the existing features
        features = pd.concat([features, new_data])
    
    return features

# Assuming the necessary dataframes and variables are already defined in the provided code
# Example usage with one country (assuming 'results_dict' contains the trained models and test data for each country)

# Define the output directory and create it if it doesn't exist
output_dir = r'C:\Users\MR99924\workspace\vscode\Projects\EM_RatesRF'
os.makedirs(output_dir, exist_ok=True)

# Create a dictionary to store the implied probabilities DataFrames for each country
implied_probabilities_dict = {}

# Iterate over each country in the results_dict to calculate implied probabilities and store them in the dictionary
for country, results in results_dict.items():
    print(f"Processing country: {country}")
    
    # Load the trained model from the file
    model_filename = f'{country}_rf_model.joblib'
    rf_model = load(model_filename)
    
    # Extend the feature datasets to the current date
    features_extd = extend_features_to_today(results['X_test_selected'])
    
    # Replace infinite values with NaN and forward fill NaN values with the last available value
    features_extd.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_extd.ffill(inplace=True)
    
    # Get today's datapoint (closest to today)
    todays_features = features_extd.loc[pd.to_datetime('today').normalize()].to_frame().T
    print(f"Today's features for {country}: {todays_features}")
    
    # Ensure that todays_features has no NaN or infinite values before prediction
    if not todays_features.isnull().values.any() and np.isfinite(todays_features).all().all():
        try:
            implied_probabilities = calculate_implied_probabilities(rf_model, todays_features)
            
            # Add the index (date) to the implied probabilities DataFrame
            implied_probabilities.index = todays_features.index
            
            # Store the implied probabilities DataFrame in the dictionary
            implied_probabilities_dict[country] = implied_probabilities
            
            print(f"Implied probabilities for {country}: {implied_probabilities}")
        except ValueError as e:
            print(f"Error calculating implied probabilities for {country}: {e}")
    else:
        print(f"Today's features for {country} contains NaN or infinite values after cleaning.")

# Save all implied probabilities DataFrames to a single Excel file with different sheets
with pd.ExcelWriter(os.path.join(output_dir, 'implied_probabilities_results.xlsx')) as writer:
    for country, df in implied_probabilities_dict.items():
        df.to_excel(writer, sheet_name=country)

print("The implied probabilities have been successfully saved to an Excel file with different sheets.")