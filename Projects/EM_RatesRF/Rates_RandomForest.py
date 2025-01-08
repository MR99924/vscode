# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:22:46 2024

@author: MR99924
"""
import sys
sys.path.append(r'S:\Shared\Front Office\Asset Allocation\Analytics\libs')
import pandas as pd
#import bloomberg
import numpy as np
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

# Having these as optional add-ons for further use in the streamlining of our processes
#import xlwings as xw
#import bloomberg
# Also looking to get a macrobond API extention included also. This would remnove the need for us to operate on Excel at all.
# Means a one-button press is highly suitable for our needs


# We now move to the bloomberg function we are going to try to sort out.

#def get_bloomberg_date(tickers, date_from, date_to, field="PX_LAST", periodicity="DAILY"):
#    bbg = bloomberg.Bloomberg()
#    df = bbg.historicalRequest(tickers,
#                               field,
#                               date_from,
#                               date_to,
#                               periodicitySelection=periodicity,
#                               nonTradingDayFillOption="ALL_CALENDAR_DAYS",
#                               nonTradingDayFillMethod="PREVIOUS_VALUE",
#                               )
#    df = pd.pivot_table(df,
#                        values='bbergvalue',
#                        index=['bbergdate'],
#                        columns=['bbergsymbol'],
#                       aggfunc=np.max,
#                       )
#    df = df[tickers]
#    return df


# Let's start by importing our data for each of the series we are interested in...

# Path to folder containing the data files

folder_path = r"C:\Users\MR99924\.spyder-py3\workspace\Projects\EM_RatesRF\DataPull.xlsx"

#sheet1 = "Policy_Rates" # Making directional claims here is we want for our model to predict
#df1 = pd.read_excel(folder_path, sheet_name = sheet1)
#new_column_names1 = {
#        "Brazil": "brazil_pr",
#        "Chile": "chile_pr",
#        "Mexico": "mexico_pr",
#        "Czechia": "czechia_pr",
#        "Poland": "poland_pr",
#        "China": "china_pr",
#        "India": "india_pr",
#        "Korea": "korea_pr",
#        "Colombia": "colombia_pr",
#        "Hungary": "hungary_pr",
#        "Turkey" : "turkey_pr",
#        "Israel": "israel_pr",
#        "South Africa": "south_africa_pr",
#        "Malaysia": "malaysia_pr",
#        "Philippines": "philippines_pr",
#        "Thailand": "thailand_pr",
#        "Taiwan": "taiwan_pr",
#        "Peru": "peru_pr",
#        "Indonesia": "indonesia_pr"
#        }
#df1 = df1.rename(columns = new_column_names1).drop(index = "Date")

#column_names = df1.columns.tolist()
#print(column_names)
#print(df1.index)
#print(df1.columns)
#df1.to_csv("output.csv", index = False)

sheet2 = "Headline_Inflation" # Inflation factor - *you're going to have to fix the 0 error*
df2 = pd.read_excel(folder_path, sheet_name = sheet2)
#print(df2)
#start_date = df2["Date"].min()
#end_date = df2["Date"].max() + pd.offsets.MonthEnd(1)
#
#daily_dates = pd.date_range(start = start_date, end = end_date, freq = "D")
#
#d_df2 = pd.DataFrame({"Period": daily_dates})
#d_df2["Period"] = df2["Date"].dt.to_period("M")
#
#r_df2 = pd.merge(d_df2, df2, on = "Date", how = "left")
#
#print(r_df2)
#
new_column_names2 = {
        "Brazil": "brazil_inf",
        "Chile": "chile_inf",
        "Mexico": "mexico_inf",
        "Czechia": "czechia_inf",
        "Poland": "poland_inf",
        "China": "china_inf",
        "India": "india_inf",
        "Korea": "korea_inf",
        "Colombia": "colombia_inf",
        "Hungary": "hungary_inf",
        "Turkey" : "turkey_inf",
        "Israel": "israel_inf",
        "South Africa": "south_africa_inf",
        "Malaysia": "malaysia_inf",
        "Philippines": "philippines_inf",
        "Thailand": "thailand_inf",
        "Taiwan": "taiwan_inf",
        "Peru": "peru_inf",
        "Indonesia": "indonesia_inf"
        }

df2 = df2.rename(columns = new_column_names2)

column_names = df2.columns.tolist()
print(column_names)
print(df2.index)
print(df2.columns)
df2.to_csv("output.csv", index = False)


#sheet3 = "Core_inflation" # Inflation factor - *you're going to have to fix the 0 error*
#df3 = pd.read_excel(folder_path, sheet_name = sheet3)
#new_column_names3 = {
#        "Brazil": "brazil_c_inf",
#        "Chile": "chile_c_ind",
#        "Mexico": "mexico_c_inf",
#        "Czechia": "czechia_c_inf",
#        "Poland": "poland_c_inf",
#        "China": "china_c_inf",
#        "India": "india_c_inf",
#        "Korea": "korea_c_inf",
#        "Colombia": "colombia_c_inf",
#        "Hungary": "hungary_c_inf",
#        "Turkey" : "turkey_c_inf",
#        "Israel": "israel_c_inf",
#        "South Africa": "south_africa_c_inf",
#        "Malaysia": "malaysia_c_inf",
#        "Philippines": "philippines_c_inf",
#        "Thailand": "thailand_c_inf",
#        "Taiwan": "taiwan_c_inf",
#        "Peru": "peru_c_inf",
#        "Indonesia": "indonesia_c_inf"
#        }
#df3 = df3.rename(columns = new_column_names3).drop(index = "Date")

#column_names = df3.columns.tolist()
#print(column_names)
#print(df3.index)
#print(df3.columns)
#df3.to_csv("output.csv", index = False)


#
#sheet4 = "REER" # Currency factor - you'll need to adjust it back **need to fix***
#df4 = pd.read_excel(folder_path, sheet_name = sheet4)
#df4 = df4.drop(index=[0, 2, 3, 4, 5])
#print(df4)
#df4 = df4.drop(index = 1)
#df4 = df4.reset_index(drop = True)
#new_column_names4 = {
#        "1994-01-01 00:00:00" : "brazil_reer",
#        "Unnamed: 2": "chile_reer",
#        "Unnamed: 3": "mexico_reer",
#        "Unnamed: 4": "czechia_reer",
#        "Unnamed: 5": "poland_reer",
#        "Unnamed: 6": "china_reer",
#        "Unnamed: 7": "india_reer",
#        "Unnamed: 8": "korea_reer",
#        "Unnamed: 9": "colombia_reer",
#        "Unnamed: 10": "hungary_reer",
#        "Unnamed: 11" : "turkey_reer",
#        "Unnamed: 12": "israel_reer",
#        "Unnamed: 13": "south_africa_reer",
#        "Unnamed: 14": "malaysia_reer",
#        "Unnamed: 15": "philippines_reer",
#        "Unnamed: 16": "thailand_reer",
#        "Unnamed: 17": "taiwan_reer",
#        "Unnamed: 18": "peru_reer",
#        "Unnamed: 19": "indonesia_reer"
#        }
#df4 = df4.rename(columns = new_column_names4)
#print(df4.index)
#print(df4.columns)
#df4.to_csv("output.csv", index = False)
#


#sheet5 = "DM_Policy_Rates" # Rates factor - you'll need some hhelp here, this isn't working like the others.
#df5 = pd.read_excel(folder_path, sheet_name = sheet5)
#new_column_names5 = {
#        "Date" : "Date", 
#        "United States, Policy Rates, Federal Reserve, Target Rates, Federal Funds Target Rate": "US_Fed",
#        "Euro area": "ECB",
#        "Japan": "BoJ",
#        "United Kingdom, Policy Rates, Bank of England, Bank Rate": "BoE",
#        }
#df5.rename(columns = new_column_names5)
#try:
#    df5 = df5.drop(index = "Date")
#except:
#    print("Python can't find what's literally right in front of it")
#print(df5.index)
##
##column_names = df5.columns.tolist()
##print(column_names)
##
##print(df5.columns)
#df5.to_csv("output.csv", index = False)

#
#
#sheet6 = "GrowthTrackers" # Activity factor - extensions aside, this one works okay
#df6 = pd.read_excel(folder_path, sheet_name = sheet6)
#new_column_names6 = {
#        "Brazil": "brazil_act",
#        "Chile": "chile_act",
#        "Mexico": "mexico_act",
#        "Czechia": "czechia_act",
#        "Poland": "poland_act",
#        "China": "china_act",
#        "India": "india_act",
#        "Korea": "korea_act",
#        "Colombia": "colombia_act",
#        "Hungary": "hungary_act",
#        "Turkey" : "turkey_act",
#        "Israel": "israel_act",
#        "South Africa": "south_africa_act",
#        "Malaysia": "malaysia_act",
#        "Philippines": "philippines_act",
#        "Thailand": "thailand_act",
#        "Taiwan": "taiwan_act",
#        "Peru": "peru_act",
#        "Indonesia": "indonesia_act"
#        }
#df6 = df6.rename(columns = new_column_names6).drop(index = "Date")
#
#column_names = df6.columns.tolist()
#print(column_names)
#print(df6.index)
#print(df6.columns)
#df6.to_csv("output.csv", index = False)


#sheet7 = "CA" # External factor - 100% fine
#df7 = pd.read_excel(folder_path, sheet_name = sheet7)
#new_column_names7 = {
#        "Brazil": "brazil_ca",
#        "Chile": "chile_ca",
#        "Mexico": "mexico_ca",
#        "Czechia": "czechia_ca",
#        "Poland": "poland_ca",
#        "China": "china_ca",
#        "India": "india_ca",
#        "Korea": "korea_ca",
#        "Colombia": "colombia_ca",
#        "Hungary": "hungary_ca",
#        "Turkey" : "turkey_ca",
#        "Israel": "israel_ca",
#        "South Africa": "south_africa_ca",
#        "Malaysia": "malaysia_ca",
#        "Philippines": "philippines_ca",
#        "Thailand": "thailand_ca",
#        "Taiwan": "taiwan_ca",
#        "Peru": "peru_ca",
#        "Indonesia": "indonesia_ca"
#        }
#df7 = df7.rename(columns = new_column_names7).drop(index = "Date")
#
#column_names = df7.columns.tolist()
#print(column_names)
#print(df7.index)
#print(df7.columns)
#df7.to_csv("output.csv", index = False)

#
#sheet8 = "Fiscal" # Fiscal factor - 100% fine
#df8 = pd.read_excel(folder_path, sheet_name = sheet8)
#new_column_names8 = {
#        "Brazil": "brazil_fb",
#        "Chile": "chile_fb",
#        "Mexico": "mexico_fb",
#        "Czechia": "czechia_fb",
#        "Poland": "poland_fb",
#        "China": "china_fb",
#        "India": "india_fb",
#        "Korea": "korea_fb",
#        "Colombia": "colombia_fb",
#        "Hungary": "hungary_fb",
#        "Turkey" : "turkey_fb",
#        "Israel": "israel_fb",
#        "South Africa": "south_africa_fb",
#        "Malaysia": "malaysia_fb",
#        "Philippines": "philippines_fb",
#        "Thailand": "thailand_fb",
#        "Taiwan": "taiwan_fb",
#        "Peru": "peru_fb",
#        "Indonesia": "indonesia_fb"
#        }
#df8 = df8.rename(columns = new_column_names8).drop(index = "Date")
#
#column_names = df8.columns.tolist()
#print(column_names)
#print(df8.index)
#print(df8.columns)
#df8.to_csv("output.csv", index = False)

#sheet9 = "PrimaryBalance" # Fiscal factor - 100% fine (may need to adjust so it's not 1yma)
#df9 = pd.read_excel(folder_path, sheet_name = sheet9)
#new_column_names9 = {
#        "Brazil": "brazil_pfb",
#        "Chile": "chile_pfb",
#        "Mexico": "mexico_pfb",
#        "Czechia": "czechia_pfb",
#        "Poland": "poland_pfb",
#        "China": "china_pfb",
#        "India": "india_pfb",
#        "Korea": "korea_pfb",
#        "Colombia": "colombia_pfb",
#        "Hungary": "hungary_pfb",
#        "Turkey" : "turkey_pfb",
#        "Israel": "israel_pfb",
#        "South Africa": "south_africa_pfb",
#        "Malaysia": "malaysia_pfb",
#        "Philippines": "philippines_pfb",
#        "Thailand": "thailand_pfb",
#        "Taiwan": "taiwan_pfb",
#        "Peru": "peru_pfb",
#        "Indonesia": "indonesia_pfb"
#        }
#df9 = df9.rename(columns = new_column_names9).drop(index = "Date")
#
#column_names = df9.columns.tolist()
#print(column_names)
#print(df9.index)
#print(df9.columns)
#df9.to_csv("output.csv", index = False)

#sheet10 = "EMBIGSpreads" # Fiscal factor - BBG framework, might be better to use the function as specified.
#df10 = pd.read_excel(folder_path, sheet_name = sheet10)
#new_column_names10 = {
#        "Brazil": "brazil_spr",
#        "Chile": "chile_spr",
#        "Mexico": "mexico_spr",
#        "Czechia": "czechia_spr",
#        "Poland": "poland_spr",
#        "China": "china_spr",
#        "India": "india_spr",
#        "Korea": "korea_spr",
#        "Colombia": "colombia_spr",
#        "Hungary": "hungary_spr",
#        "Turkey" : "turkey_spr",
#        "Israel": "israel_spr",
#        "South Africa": "south_africa_spr",
#        "Malaysia": "malaysia_spr",
#        "Philippines": "philippines_spr",
#        "Thailand": "thailand_spr",
#        "Taiwan": "taiwan_spr",
#        "Peru": "peru_spr",
#        "Indonesia": "indonesia_spr"
#        }
#df10 = df10.rename(columns = new_column_names10)
#
#column_names = df10.columns.tolist()
#print(column_names)
#print(df10.index)
#print(df10.columns)
#df10.to_csv("output.csv", index = False)

#sheet11 = "Curr_Vol" # Currency factor - BBG framework, might be better to use the function as specified.
#df11 = pd.read_excel(folder_path, sheet_name = sheet11)
#new_column_names11 = {
#        "Brazil": "brazil_spr",
#        "Chile": "chile_spr",
#        "Mexico": "mexico_spr",
#        "Czechia": "czechia_spr",
#        "Poland": "poland_spr",
#        "China": "china_spr",
#        "India": "india_spr",
#        "Korea": "korea_spr",
#        "Colombia": "colombia_spr",
#        "Hungary": "hungary_spr",
#        "Turkey" : "turkey_spr",
#        "Israel": "israel_spr",
#        "South Africa": "south_africa_spr",
#        "Malaysia": "malaysia_spr",
#        "Philippines": "philippines_spr",
#        "Thailand": "thailand_spr",
#        "Taiwan": "taiwan_spr",
#        "Peru": "peru_spr",
#        "Indonesia": "indonesia_spr"
#        }
#df11 = df11.rename(columns = new_column_names11)
#
#column_names = df11.columns.tolist()
#print(column_names)
#print(df11.index)
#print(df11.columns)
#df11.to_csv("output.csv", index = False)




#sheet12 = "Reserves" # External factor - 100% fine
#df12 = pd.read_excel(folder_path, sheet_name = sheet12)
#new_column_names12 = {
#        "Brazil": "brazil_res",
#        "Chile": "chile_res",
#        "Mexico": "mexico_res",
#        "Czechia": "czechia_res",
#        "Poland": "poland_res",
#        "China": "china_res",
#        "India": "india_res",
#        "Korea": "korea_res",
#        "Colombia": "colombia_res",
#        "Hungary": "hungary_res",
#        "Turkey" : "turkey_res",
#        "Israel": "israel_res",
#        "South Africa": "south_africa_res",
#        "Malaysia": "malaysia_res",
#        "Philippines": "philippines_res",
#        "Thailand": "thailand_res",
#        "Taiwan": "taiwan_res",
#        "Peru": "peru_res",
#        "Indonesia": "indonesia_res"
#        }
#df12 = df12.rename(columns = new_column_names12).drop(index = "Date")
#
#column_names = df12.columns.tolist()
#print(column_names)
#print(df12.index)
#print(df12.columns)
#df12.to_csv("output.csv", index = False)


#
#sheet13 = "PPI" # Inflation factor - done by BBG frameworks. Best perhaps to exfil via the BBG library.
#df13 = pd.read_excel(folder_path, sheet_name = sheet13)
#new_column_names13 = {
#        "Brazil": "brazil_ppi",
#        "Chile": "chile_ppi",
#        "Mexico": "mexico_ppi",
#        "Czechia": "czechia_ppi",
#        "Poland": "poland_ppi",
#        "China": "china_ppi",
#        "India": "india_ppi",
#        "Korea": "korea_ppi",
#        "Colombia": "colombia_ppi",
#        "Hungary": "hungary_ppi",
#        "Turkey" : "turkey_ppi",
#        "Israel": "israel_ppi",
#        "South Africa": "south_africa_ppi",
#        "Malaysia": "malaysia_ppi",
#        "Philippines": "philippines_ppi",
#        "Thailand": "thailand_ppi",
#        "Taiwan": "taiwan_ppi",
#        "Peru": "peru_ppi",
#        "Indonesia": "indonesia_ppi"
#        }
#df13 = df13.rename(columns = new_column_names13)
#
#column_names = df13.columns.tolist()
#print(column_names)
#print(df13.index)
#print(df13.columns)
#df13.to_csv("output.csv", index = False)
#


# Okay, now the data for our series is imported, and the various categories flagged. 

#Combining the various indices
# inf_combined_df = pd.concat([dfs["Headline_Inflation"], dfs["Core_inflation"], dfs["PPI"]], axis=1)
# cur_combined_df = pd.concat([dfs["REER"], dfs["Curr_Vol"]], axis=1)
# rat_combined_df = dfs["DM_Policy_Rates"]
# act_combined_df = dfs["GrowthTrackers"]
# ext_combined_df = pd.concat([dfs["CA"], dfs["Reserves"]], axis=1)
# fis_combined_df = pd.concat([dfs["Fiscal"], dfs["PrimaryBalance"], dfs["EMBIGSpreads"]], axis=1)




#scaler = StandardScaler()

#inf_combined_df_scaled = scaler.fit_transform(inf_combined_df)
#cur_combined_df_scaled = scaler.fit_transform(cur_combined_df)
#rat_combined_df_scaled = scaler.fit_transform(rat_combined_df)
#act_combined_df_scaled = scaler.fit_transform(act_combined_df)
#ext_combined_df_scaled = scaler.fit_transform(ext_combined_df)
#fis_combined_df_scaled = scaler.fit_transform(fis_combined_df)

#inf_pca = PCA(n_components=3)
#cur_pca = PCA(n_components=2)
#rat_pca = PCA(n_components=1)  # Adjusted to avoid error
#act_pca = PCA(n_components=1)
#ext_pca = PCA(n_components=2)
#fis_pca = PCA(n_components=3)

#inf_principal_components = inf_pca.fit_transform(inf_combined_df_scaled)
#cur_principal_components = cur_pca.fit_transform(cur_combined_df_scaled)
#rat_principal_components = rat_pca.fit_transform(rat_combined_df_scaled)
#act_principal_components = act_pca.fit_transform(act_combined_df_scaled)
#ext_principal_components = ext_pca.fit_transform(ext_combined_df_scaled)
#fis_principal_components = fis_pca.fit_transform(fis_combined_df_scaled)


# Now we perform the PCA
#print("Explained variance ratio for Inflation factors PCA:", inf_pca.explained_variance_ratio_)
#print("Explained variance ratio for Currency factors PCA:", cur_pca.explained_variance_ratio_)
#print("Explained variance ratio for Rates factors PCA:", rat_pca.explained_variance_ratio_)
#print("Explained variance ratio for Activity factors PCA:", act_pca.explained_variance_ratio_)
#print("Explained variance ratio for External factors PCA:", ext_pca.explained_variance_ratio_)
#print("Explained variance ratio for Fiscal factors PCA:", fis_pca.explained_variance_ratio_)