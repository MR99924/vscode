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
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
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

tickers = ["BISPDHBR Index",
           "BISPDHCL Index",
           "BISPDHMX Index",
           "BISPDHCZ Index",
           "BISPDHPO Index",
           "BISPDHCH Index",
           "BISPDHIN Index",
           "BISPDHSK Index",
           "BISPDHCO Index",
           "BISPDHHU Index",
           "1866582 Index",
           "BISPDHIS Index",
           "BISPDHSA Index",
           "BISPDHMA Index",
           "BISPDHPH Index",
           "BISPDHTH Index",
           "TAREDSC Index",
           "BISPDHPE Index",
           "5366582 Index"]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

rate_data = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

rate_data = rate_data.rename(columns={"BISPDHBR Index": "rate_br",
           "BISPDHCL Index": "rate_cl",
           "BISPDHMX Index": "rate_mx",
           "BISPDHCZ Index": "rate_cz",
           "BISPDHPO Index": "rate_pl",
           "BISPDHCH Index": "rate_cn",
           "BISPDHIN Index": "rate_in",
           "BISPDHSK Index": "rate_kr",
           "BISPDHCO Index": "rate_co",
           "BISPDHHU Index": "rate_hu",
           "1866582 Index": "rate_tr",
           "BISPDHIS Index": "rate_il",
           "BISPDHSA Index": "rate_za",
           "BISPDHMA Index": "rate_my",
           "BISPDHPH Index": "rate_ph",
           "BISPDHTH Index": "rate_th",
           "TAREDSC Index": "rate_tw",
           "BISPDHPE Index": "rate_pe",
           "5366582 Index": "rate_id"})

rate_data_pop = rate_data.diff()

print(rate_data_pop.head())

# Now we move on to inflation (Macrobond)

tickers = ["brcpi",
           "clcpi",
           "mxcpi",
           "czcpi",
           "plcpi",
           "cncpi",
           "incpi",
           "krcpi",
           "cocpi",
           "hucpi",
           "trcpi",
           "ilcpi",
           "zacpi",
           "mycpi",
           "phcpi",
           "thcpi",
           "twcpi",
           "pecpi",
           "idcpi"]

h_inf = mb.FetchSeries(tickers)

new_column_names = {
        "brcpi": "h_inf_br",
        "clcpi": "h_inf_cl",
        "mxcpi": "h_inf_mx",
        "czcpi": "h_inf_cz",
        "plcpi": "h_inf_pl",
        "cncpi": "h_inf_cn",
        "incpi": "h_inf_in",
        "krcpi": "h_inf_kr",
        "cocpi": "h_inf_co",
        "hucpi": "h_inf_hu",
        "trcpi": "h_inf_tr",
        "ilcpi": "h_inf_il",
        "zacpi": "h_inf_za",
        "mycpi": "h_inf_my",
        "phcpi": "h_inf_ph",
        "thcpi": "h_inf_th",
        "twcpi": "h_inf_tw",
        "pecpi": "h_inf_pe",
        "idcpi": "h_inf_id"
        }

h_inf = h_inf.rename(columns = new_column_names)
h_inf_3mma = h_inf.rolling(window=3).mean()
h_inf_qoq_3mma = h_inf_3mma.pct_change(periods=3) * 100
h_inf_qoq_3mma_d = h_inf_qoq_3mma.resample('D').ffill()
h_inf_qoq_3mma_d = apply_backfill_after_first_observation(h_inf_qoq_3mma_d)
h_inf_qoq_3mma_d_lggd = h_inf_qoq_3mma_d.shift(30)

# Now we move on to core inflation (Macrobond)

tickers = ["brpric3002",
           "clpric0001",
           "mxpric1009",
           "ih:mb:com:cze_corepriceindex:cze_corepriceindex",
           "ih:mb:com:pol_coreprices_index:pol_coreprices_index",
           "cnpric4565",
           "inpric3225",
           "krpric2395",
           "copric1015",
           "hupric0143",
           "trpric0243",
           "ilpric0085",
           "zapric0381",
           "ih:mb:com:mys_corepriceindex:mys_corepriceindex",
           "ih:mb:com:phl_corepriceindex:phl_corepriceindex",  # Changed from "JoinMoreHistoryScaled(phpric0301, phpric0301_2012)",
           "thpric0035",
           "twpric0536",
           "pepric0288",
           "idpric0122"]

c_inf = mb.FetchSeries(tickers)

new_column_names = {"brpric3002": "c_inf_br",
           "clpric0001": "c_inf_cl",
           "mxpric1009": "c_inf_mx",
           "ih:mb:com:cze_corepriceindex:cze_corepriceindex": "c_inf_cz",
           "ih:mb:com:pol_coreprices_index:pol_coreprices_index": "c_inf_pl",
           "cnpric4565": "c_inf_cn",
           "inpric3225": "c_inf_in",
           "krpric2395": "c_inf_kr",
           "copric1015": "c_inf_co",
           "hupric0143": "c_inf_hu",
           "trpric0243": "c_inf_tr",
           "ilpric0085": "c_inf_il",
           "zapric0381": "c_inf_za",
           "ih:mb:com:mys_corepriceindex:mys_corepriceindex": "c_inf_my",
           "ih:mb:com:phl_corepriceindex:phl_corepriceindex": "c_inf_ph", # Changed from "JoinMoreHistoryScaled(phpric0301, phpric0301_2012)": "c_inf_ph",
           "thpric0035": "c_inf_th",
           "twpric0536": "c_inf_tw",
           "pepric0288": "c_inf_pe",
           "idpric0122": "c_inf_id"
           }

c_inf = c_inf.rename(columns = new_column_names)
c_inf_3mma = c_inf.rolling(window=3).mean()
c_inf_qoq_3mma = c_inf_3mma.pct_change(periods=3) * 100
c_inf_qoq_3mma_d = c_inf_qoq_3mma.resample('D').ffill()
c_inf_qoq_3mma_d = apply_backfill_after_first_observation(c_inf_qoq_3mma_d)
c_inf_qoq_3mma_d_lggd = c_inf_qoq_3mma_d.shift(30)

# Now we move on to REER (Bloomberg)

tickers = ["JBDNBRL  Index",
           "JBDNCLP  Index",
           "JBDNMXN  Index",
           "JBDNCZK  Index",
           "JBDNPLN  Index",
           "JBDNCNY  Index",
           "JBDNINR  Index",
           "JBDNKRW  Index",
           "JBDNCOP  Index",
           "JBDNHUF  Index",
           "JBDNTRY  Index",
           "JBDNILS  Index",
           "JBDNZAR  Index",
           "JBDNMYR  Index",
           "JBDNPHP  Index",
           "JBDNTHB  Index",
           "JBDNTWD  Index",
           "JBDNPEN  Index",
           "JBDNIDR  Index"
]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

neer_data = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

neer_data = neer_data.rename(columns={"JBDNBRL  Index": "neer_br",
           "JBDNCLP  Index": "neer_cl",
           "JBDNMXN  Index": "neer_mx",
           "JBDNCZK  Index": "neer_cz",
           "JBDNPLN  Index": "neer_pl",
           "JBDNCNY  Index": "neer_cn",
           "JBDNINR  Index": "neer_in",
           "JBDNKRW  Index": "neer_kr",
           "JBDNCOP  Index": "neer_co",
           "JBDNHUF  Index": "neer_hu",
           "JBDNTRY  Index": "neer_tr",
           "JBDNILS  Index": "neer_il",
           "JBDNZAR  Index": "neer_za",
           "JBDNMYR  Index": "neer_my",
           "JBDNPHP  Index": "neer_ph",
           "JBDNTHB  Index": "neer_th",
           "JBDNTWD  Index": "neer_tw",
           "JBDNPEN  Index": "neer_pe",
           "JBDNIDR  Index": "neer_id"})

neer_data = neer_data.rename(columns = new_column_names)
neer_data_3mma = neer_data.rolling(window=90).mean()
neer_data_qoq_3mma = neer_data_3mma.pct_change(periods=90) * 100
neer_data_qoq_3mma_d = neer_data_qoq_3mma.resample('D').ffill()
neer_data_qoq_3mma_d = apply_backfill_after_first_observation(neer_data_qoq_3mma_d)

# Now let's have a crack at the global rates variable, DM policy rates (Bloomberg)

tickers = ["FDTR Index", "EURR002W Index", "UKBRBASE Index", "BOJDTR Index"]
dt_from = dt.date(1990, 1, 1)
dt_to = dt.date.today()

# Get the data
dm_rate_data = get_bloomberg_date(tickers, dt_from, dt_to, periodicity='DAILY')

# Rename columns
dm_rate_data = dm_rate_data.rename(columns={"FDTR Index": "FED",
                                            "EURR002W Index": "ECB",
                                            "UKBRBASE Index": "BoE",
                                            "BOJDTR Index": "BoJ"})

weights = [0.7, 0.2, 0.05, 0.05]

# Define the weighted average function
def weighted_average(values, weights):
    return np.average(values, weights=weights)

# Calculate the rolling mean and difference
dm_rate_data_3mma = dm_rate_data.rolling(window=90).mean()
dm_rate_data_qd_3mma = dm_rate_data_3mma.diff(periods=90)

# Calculate the weighted average for each period and create a new column
dm_rate_data_qd_3mma['weighted_avg'] = dm_rate_data_qd_3mma.apply(lambda row: weighted_average(row, weights), axis=1)

print(dm_rate_data_qd_3mma.tail())

# Now we move to the activity section (this is a generalised pca of a basket of metrics - kept in-house)

tickers = ["ih:mb:com:growth_tracker_qoq_bra:growth_tracker_qoq_bra",
           "ih:mb:com:growth_tracker_qoq_chl:growth_tracker_qoq_chl",
           "ih:mb:com:growth_tracker_qoq_mex:growth_tracker_qoq_mex",
           "ih:mb:com:growth_tracker_qoq_cze:growth_tracker_qoq_cze",
           "ih:mb:com:growth_tracker_qoq_pol:growth_tracker_qoq_pol",
           "ih:mb:com:growthtracker_qoq_chn:growthtracker_qoq_chn",
           "ih:mb:com:growth_tracker_qoq_ind:growth_tracker_qoq_ind",
           "ih:mb:com:growth_tracker_qoq_kor:growth_tracker_qoq_kor",
           "ih:mb:com:growth_tracker_qoq_col:growth_tracker_qoq_col",
           "ih:mb:com:growth_tracker_qoq_hun:growth_tracker_qoq_hun",
           "ih:mb:com:growth_tracker_qoq_tur:growth_tracker_qoq_tur",
           "ih:mb:com:growth_tracker_qoq_isr:growth_tracker_qoq_isr",
           "ih:mb:com:growth_tracker_qoq_zaf:growth_tracker_qoq_zaf",
           "ih:mb:com:growth_tracker_qoq_mys:growth_tracker_qoq_mys",
           "ih:mb:com:growth_tracker_qoq_phl:growth_tracker_qoq_phl",
           "ih:mb:com:growth_tracker_qoq_tha:growth_tracker_qoq_tha",
           "ih:mb:com:growth_tracker_qoq_twn:growth_tracker_qoq_twn",
           "ih:mb:com:growth_tracker_qoq_per:growth_tracker_qoq_per",
           "ih:mb:com:growth_tracker_qoq_idn:growth_tracker_qoq_idn"
           ]

pca_act = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:growth_tracker_qoq_bra:growth_tracker_qoq_bra" : "pca_act_br",
           "ih:mb:com:growth_tracker_qoq_chl:growth_tracker_qoq_chl" : "pca_act_cl",
           "ih:mb:com:growth_tracker_qoq_mex:growth_tracker_qoq_mex" : "pca_act_mx",
           "ih:mb:com:growth_tracker_qoq_cze:growth_tracker_qoq_cze" : "pca_act_cz",
           "ih:mb:com:growth_tracker_qoq_pol:growth_tracker_qoq_pol" : "pca_act_pl",
           "ih:mb:com:growthtracker_qoq_chn:growthtracker_qoq_chn" : "pca_act_cn",
           "ih:mb:com:growth_tracker_qoq_ind:growth_tracker_qoq_ind" : "pca_act_in",
           "ih:mb:com:growth_tracker_qoq_kor:growth_tracker_qoq_kor" : "pca_act_kr",
           "ih:mb:com:growth_tracker_qoq_col:growth_tracker_qoq_col" : "pca_act_co",
           "ih:mb:com:growth_tracker_qoq_hun:growth_tracker_qoq_hun" : "pca_act_hu",
           "ih:mb:com:growth_tracker_qoq_tur:growth_tracker_qoq_tur" : "pca_act_tr",
           "ih:mb:com:growth_tracker_qoq_isr:growth_tracker_qoq_isr" : "pca_act_is",
           "ih:mb:com:growth_tracker_qoq_zaf:growth_tracker_qoq_zaf" : "pca_act_za",
           "ih:mb:com:growth_tracker_qoq_mys:growth_tracker_qoq_mys" : "pca_act_my",
           "ih:mb:com:growth_tracker_qoq_phl:growth_tracker_qoq_phl" : "pca_act_ph",
           "ih:mb:com:growth_tracker_qoq_tha:growth_tracker_qoq_tha" : "pca_act_th",
           "ih:mb:com:growth_tracker_qoq_twn:growth_tracker_qoq_twn" : "pca_act_tw",
           "ih:mb:com:growth_tracker_qoq_per:growth_tracker_qoq_per" : "pca_act_pe",
           "ih:mb:com:growth_tracker_qoq_idn:growth_tracker_qoq_idn" : "pca_act_id"
        }

pca_act = pca_act.rename(columns = new_column_names)
pca_act_d = pca_act.resample('D').ffill()
pca_act_d = apply_backfill_after_first_observation(pca_act_d)
pca_act_d_lggd = pca_act_d.shift(30)

column_names = pca_act_d_lggd.columns.tolist()
print(column_names)
print(pca_act_d_lggd.index)
print(pca_act_d_lggd.columns)

# Maybe for old times' sake, in reference to inflation debates of old (and to extend the viable series over which we can train) we can include unemployemnt (Macrobond)

tickers = ["brlama0041",
           "ih:mb:com:cl_unemployment_rate:cl_unemployment_rate",
           "mxlama7491",
           "czlama0051",
           "pllama0140",
           "cnunempsagemmonth",
           "ih:mb:com:in_unemployment_rate:in_unemployment_rate",
           "krlama0422",
           "colama0276",
           "hulama0288",
           "trlama0176",
           "ih:mb:com:il_unemployment_rate:il_unemployment_rate",
           "zaunempsagemmonth",
           "mylama0106",
           "phunempsagemmonth",
           "thlama0024",
           "twlama0237",
           "pelama0001",
           ]

u_rat = mb.FetchSeries(tickers)

new_column_names = {"brlama0041": "u_rat_br",
           "ih:mb:com:cl_unemployment_rate:cl_unemployment_rate": "u_rat_cl",
           "mxlama7491": "u_rat_mx",
           "czlama0051": "u_rat_cz",
           "pllama0140": "u_rat_pl",
           "cnunempsagemmonth": "u_rat_cn",
           "ih:mb:com:in_unemployment_rate:in_unemployment_rate": "u_rat_in",
           "krlama0422": "u_rat_kr",
           "colama0276": "u_rat_co",
           "hulama0288": "u_rat_hu",
           "trlama0176": "u_rat_tr",
           "ih:mb:com:il_unemployment_rate:il_unemployment_rate": "u_rat_il",
           "zaunempsagemmonth": "u_rat_za",
           "mylama0106": "u_rat_my",
           "phunempsagemmonth": "u_rat_ph",
           "thlama0024": "u_rat_th",
           "twlama0237": "u_rat_tw",
           "pelama0001": "u_rat_pe",
           }

u_rat = u_rat.rename(columns = new_column_names)
u_rat_12mma = u_rat.rolling(window=12).mean()
u_rat_12mma_sad = u_rat_12mma.diff(periods=6)
u_rat_12mma_sad_d = u_rat_12mma_sad.resample('D').ffill()
u_rat_12mma_sad_d = apply_backfill_after_first_observation(u_rat_12mma_sad_d)
u_rat_12mma_sad_d_lggd = u_rat_12mma_sad_d.shift(30)
print(u_rat_12mma_sad_d_lggd.head())

# Now we move to the external balance content (macrobond)

tickers = ["ih:mb:com:ca_bra:ca_bra",
           "ih:mb:com:ca_chl:ca_bra",
           "ih:mb:com:ca_mex:ca_mex",
           "ih:mb:com:ca_cze:ca_cze",
           "ih:mb:com:ca_pol:ca_pol",
           "ih:mb:com:ca_chn:ca_chn",
           "ih:mb:com:ca_ind:ca_ind",
           "ih:mb:com:ca_kor:ca_kor",
           "ih:mb:com:ca_col:ca_col",
           "ih:mb:com:ca_hun:ca_hun",
           "ih:mb:com:ca_tur:ca_tur",
           "ih:mb:com:ca_isr:ca_isr",
           "ih:mb:com:ca_zaf:ca_zaf",
           "ih:mb:com:ca_mys:ca_mys",
           "ih:mb:com:ca_phl:ca_phl",
           "ih:mb:com:ca_tha:ca_tha",
           "ih:mb:com:ca_twn:ca_twn",
           "ih:mb:com:ca_per:ca_per",
           "ih:mb:com:ca_idn:ca_idn"           ]

ca_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:ca_bra:ca_bra": "ca_rat_br",
           "ih:mb:com:ca_chl:ca_bra": "ca_rat_cl",
           "ih:mb:com:ca_mex:ca_mex": "ca_rat_mx",
           "ih:mb:com:ca_cze:ca_cze": "ca_rat_cz",
           "ih:mb:com:ca_pol:ca_pol": "ca_rat_pl",
           "ih:mb:com:ca_chn:ca_chn": "ca_rat_cn",
           "ih:mb:com:ca_ind:ca_ind": "ca_rat_in",
           "ih:mb:com:ca_kor:ca_kor": "ca_rat_kr",
           "ih:mb:com:ca_col:ca_col": "ca_rat_co",
           "ih:mb:com:ca_hun:ca_hun": "ca_rat_hu",
           "ih:mb:com:ca_tur:ca_tur": "ca_rat_tr",
           "ih:mb:com:ca_isr:ca_isr": "ca_rat_il",
           "ih:mb:com:ca_zaf:ca_zaf": "ca_rat_za",
           "ih:mb:com:ca_mys:ca_mys": "ca_rat_my",
           "ih:mb:com:ca_phl:ca_phl": "ca_rat_ph",
           "ih:mb:com:ca_tha:ca_tha": "ca_rat_th",
           "ih:mb:com:ca_twn:ca_twn": "ca_rat_tw",
           "ih:mb:com:ca_per:ca_per": "ca_rat_pe",
           "ih:mb:com:ca_idn:ca_idn": "ca_rat_id" 
           }

ca_rat = ca_rat.rename(columns = new_column_names)
ca_rat_12mma = ca_rat.rolling(window=12).mean()
ca_rat_12mma_sad = ca_rat_12mma.diff(periods=6)
ca_rat_12mma_sad_d = ca_rat_12mma_sad.resample('D').ffill()
ca_rat_12mma_sad_d = apply_backfill_after_first_observation(ca_rat_12mma_sad_d)
ca_rat_12mma_sad_d_lggd = ca_rat_12mma_sad_d.shift(60)
print(ca_rat_12mma_sad_d_lggd.head())


# Now we move to fiscal (macrobond)

tickers = ["ih:mb:com:fis_cg_BRA:fis_bal_cg_BRA",
           "ih:mb:com:fis_cg_CHL:fis_bal_cg_CHL",
           "ih:mb:com:fis_cg_MEX:fis_bal_cg_MEX",
           "ih:mb:com:fis_cg_CZE:fis_bal_cg_CZE",
           "ih:mb:com:fis_cg_POL:fis_bal_cg_POL",
           "ih:mb:com:fis_gg_CHN:fis_gg_CHN",
           "ih:mb:com:fis_cg_IND:fis_bal_cg_IND",
           "ih:mb:com:fis_cg_KOR:fis_bal_cg_KOR",
           "ih:mb:com:fis_cg_COL:fis_bal_cg_COL",
           "ih:mb:com:fis_cg_HUN:fis_bal_cg_HUN",
           "ih:mb:com:fis_cg_TUR:fis_bal_cg_TUR",
           "ih:mb:com:fis_cg_ISR:fis_bal_cg_ISR",
           "ih:mb:com:fis_cg_ZAF:fis_bal_cg_ZAF",
           "ih:mb:com:fis_cg_MYS:fis_bal_cg_MYS",
           "ih:mb:com:fis_gg_PHL:fis_bal_gg_PHL",
           "ih:mb:com:fis_cg_THA:fis_bal_cg_THA",
           "ih:mb:com:fis_cg_TWN:fis_bal_cg_TWN",
           "ih:mb:com:fis_cg_PER:fis_bal_cg_PER",
           "ih:mb:com:fis_cg_IDN:fis_bal_cg_IDN"           
           ]

fb_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:fis_cg_BRA:fis_bal_cg_BRA" : "fb_rat_br",
           "ih:mb:com:fis_cg_CHL:fis_bal_cg_CHL" : "fb_rat_cl",
           "ih:mb:com:fis_cg_MEX:fis_bal_cg_MEX" : "fb_rat_mx",
           "ih:mb:com:fis_cg_CZE:fis_bal_cg_CZE" : "fb_rat_cz",
           "ih:mb:com:fis_cg_POL:fis_bal_cg_POL" : "fb_rat_pl",
           "ih:mb:com:fis_gg_CHN:fis_gg_CHN" : "fb_rat_cn",
           "ih:mb:com:fis_cg_IND:fis_bal_cg_IND" : "fb_rat_in",
           "ih:mb:com:fis_cg_KOR:fis_bal_cg_KOR" : "fb_rat_kr",
           "ih:mb:com:fis_cg_COL:fis_bal_cg_COL" : "fb_rat_co",
           "ih:mb:com:fis_cg_HUN:fis_bal_cg_HUN" : "fb_rat_hu",
           "ih:mb:com:fis_cg_TUR:fis_bal_cg_TUR" : "fb_rat_tr",
           "ih:mb:com:fis_cg_ISR:fis_bal_cg_ISR" : "fb_rat_il",
           "ih:mb:com:fis_cg_ZAF:fis_bal_cg_ZAF" : "fb_rat_za",
           "ih:mb:com:fis_cg_MYS:fis_bal_cg_MYS" : "fb_rat_my",
           "ih:mb:com:fis_gg_PHL:fis_bal_gg_PHL" : "fb_rat_ph",
           "ih:mb:com:fis_cg_THA:fis_bal_cg_THA" : "fb_rat_th",
           "ih:mb:com:fis_cg_TWN:fis_bal_cg_TWN" : "fb_rat_tw",
           "ih:mb:com:fis_cg_PER:fis_bal_cg_PER" : "fb_rat_pe",
           "ih:mb:com:fis_cg_IDN:fis_bal_cg_IDN" : "fb_rat_id"    
           }

fb_rat = fb_rat.rename(columns = new_column_names)
fb_rat_12mma = fb_rat.rolling(window=12).mean()
fb_rat_12mma_sad = fb_rat_12mma.diff(periods=6)
fb_rat_12mma_sad_d = fb_rat_12mma_sad.resample('D').ffill()
fb_rat_12mma_sad_d = apply_backfill_after_first_observation(fb_rat_12mma_sad_d)
fb_rat_12mma_sad_d_lggd = fb_rat_12mma_sad_d.shift(60)
print(fb_rat_12mma_sad_d_lggd.head())


# Now we move on to the EMBIG credit spread data - organised by particular countries


tickers = ["JPSSGDBR Index",
           "JPSSGDCH Index",
           "JPSSGDMX Index",
           "JPSPCZEC Index",
           "JPSSGDPO Index",
           "JPSSGDCN Index",
           "JPSSGDIN Index",
           "JBMXKRST Index",
           "JPSSGDCL Index",
           "JPSSGDHN Index",
           "JPSSGDTR Index",
           "JEMBZAST Index",
           "JPSSGMAL Index",
           "JPSSGDPH Index",
           "JEMBTHST Index",
           "JBMXTWST Index",
           "JPSSGDPE Index",
           "JPSSGDID Index"    ]
dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

embig_data = get_bloomberg_date(tickers,dt_from,dt_to,periodicity = 'DAILY')

embig_data = embig_data.rename(columns={"JPSSGDBR Index" : "embig_data_br",
           "JPSSGDCH Index" : "embig_data_cl",
           "JPSSGDMX Index" : "embig_data_mx",
           "JPSPCZEC Index" : "embig_data_cz",
           "JPSSGDPO Index" : "embig_data_pl",
           "JPSSGDCN Index" : "embig_data_cn",
           "JPSSGDIN Index" : "embig_data_in",
           "JBMXKRST Index" : "embig_data_kr",
           "JPSSGDCL Index" : "embig_data_co",
           "JPSSGDHN Index" : "embig_data_hu",
           "JPSSGDTR Index" : "embig_data_tr",
           "JPSSGDSA Index" : "embig_data_za",
           "JPSSGMAL Index" : "embig_data_my",
           "JPSSGDPH Index" : "embig_data_ph",
           "JEMBTHST Index" : "embig_data_th",
           "JBMXTWST Index" : "embig_data_tw",
           "JPSSGDPE Index" : "embig_data_pe",
           "JPSSGDID Index" : "embig_data_id"
           })

print(embig_data.head())

# Now we move to currency volatility (Bloomberg)

tickers = ["BRL Curncy",
           "CLP Curncy",
           "MXN Curncy",
           "CZK Curncy",
           "PLN Curncy",
           "CNY Curncy",
           "INR Curncy",
           "KRW Curncy",
           "COP Curncy",
           "HUF Curncy",
           "TRY Curncy",
           "ILS Curncy",
           "ZAR Curncy",
           "MYR Curncy",
           "PHP Curncy",
           "THB Curncy",
           "TWD Curncy",
           "PEN Curncy",
           "IDR Curncy"    ]

dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

curr_vol = get_bloomberg_date(tickers,dt_from,dt_to, field = 'VOLATILITY_60D', periodicity = 'DAILY')

curr_vol = curr_vol.rename(columns={"BRL Curncy" : "cur_vol_br",
           "CLP Curncy" : "cur_vol_cl",
           "MXN Curncy" : "cur_vol_mx",
           "CZK Curncy" : "cur_vol_cz",
           "PLN Curncy" : "cur_vol_pl",
           "CNY Curncy" : "cur_vol_cn",
           "INR Curncy" : "cur_vol_in",
           "KRW Curncy" : "cur_vol_kr",
           "COP Curncy" : "cur_vol_co",
           "HUF Curncy" : "cur_vol_hu",
           "TRY Curncy" : "cur_vol_tr",
           "ILS Curncy" : "cur_vol_il",
           "ZAR Curncy" : "cur_vol_za",
           "MYR Curncy" : "cur_vol_my",
           "PHP Curncy" : "cur_vol_ph",
           "THB Curncy" : "cur_vol_th",
           "TWD Curncy" : "cur_vol_tw",
           "PEN Curncy" : "cur_vol_pe",
           "IDR Curncy" : "cur_vol_id" 
           })

print(curr_vol.head())

# Now we move to external reserves as a share of short term debt (Macrobond)

tickers = ["ih:mb:com:reserve_metric_bra:reserve_metric_bra",
           "ih:mb:com:reserve_metric_chl:reserve_metric_chl",
           "ih:mb:com:reserve_metric_mex:reserve_metric_mex",
           "ih:mb:com:reserve_metric_pol:reserve_metric_pol",
           "ih:mb:com:reserve_metric_chn:reserve_metric_chn",
           "ih:mb:com:reserve_metric_ind:reserve_metric_ind",
           "ih:mb:com:reserve_metric_kor:reserve_metric_kor",
           "ih:mb:com:reserve_metric_col:reserve_metric_col",
           "ih:mb:com:reserve_metric_hun:reserve_metric_hun",
           "ih:mb:com:reserve_metric_tur:reserve_metric_tur",
           "ih:mb:com:reserve_metric_isr:reserve_metric_isr",
           "ih:mb:com:reserve_metric_zaf:reserve_metric_zaf",
           "ih:mb:com:reserve_metric_mys:reserve_metric_mys",
           "ih:mb:com:reserve_metric_phl:reserve_metric_phl",
           "ih:mb:com:reserve_metric_tha:reserve_metric_tha",
           "ih:mb:com:reserve_metric_per:reserve_metric_per",
           "ih:mb:com:reserve_metric_idn:reserve_metric_idn",
                      ]

res_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:reserve_metric_bra:reserve_metric_bra" : "res_rat_br",
           "ih:mb:com:reserve_metric_chl:reserve_metric_chl" : "res_rat_cl",
           "ih:mb:com:reserve_metric_mex:reserve_metric_mex" : "res_rat_mx",
           "ih:mb:com:reserve_metric_pol:reserve_metric_pol" : "res_rat_pl",
           "ih:mb:com:reserve_metric_chn:reserve_metric_chn" : "res_rat_cn",
           "ih:mb:com:reserve_metric_ind:reserve_metric_ind" : "res_rat_in",
           "ih:mb:com:reserve_metric_kor:reserve_metric_kor" : "res_rat_kr",
           "ih:mb:com:reserve_metric_col:reserve_metric_col" : "res_rat_co",
           "ih:mb:com:reserve_metric_hun:reserve_metric_hun" : "res_rat_hu",
           "ih:mb:com:reserve_metric_tur:reserve_metric_tur" : "res_rat_tr",
           "ih:mb:com:reserve_metric_isr:reserve_metric_isr" : "res_rat_il",
           "ih:mb:com:reserve_metric_zaf:reserve_metric_zaf" : "res_rat_za",
           "ih:mb:com:reserve_metric_mys:reserve_metric_mys" : "res_rat_my",
           "ih:mb:com:reserve_metric_phl:reserve_metric_phl" : "res_rat_ph",
           "ih:mb:com:reserve_metric_tha:reserve_metric_tha" : "res_rat_th",
           "ih:mb:com:reserve_metric_per:reserve_metric_per" : "res_rat_pe",
           "ih:mb:com:reserve_metric_idn:reserve_metric_idn" : "res_rat_id"    
           }

res_rat = res_rat.rename(columns = new_column_names)
res_rat_12mma = res_rat.rolling(window=12).mean()
res_rat_12mma_sad = res_rat_12mma.diff(periods=6)
res_rat_12mma_sad_d = res_rat_12mma_sad.resample('D').ffill()
res_rat_12mma_sad_d = apply_backfill_after_first_observation(res_rat_12mma_sad_d)
print(res_rat_12mma_sad_d.head())

# Lastly, we go to PPI inflation, we can either keep or remove, depending on personal preference (macrobond)

tickers = ["ih:mb:com:br_ppi_joined:br_ppi_joined",
           "ih:mb:com:cl_ppi_joined:cl_ppi_joined",
           "mxpric7351",
           "czpric0076",
           "plpric0065",
           "cnpric4601",
           "inpric0881",
           "krpric0014",
           "copric1010",
           "hupric0066",
           "trpric0244",
           "ilpric0018",
           "ih:mb:com:za_ppi_joined:za_ppi_joined",
           "mypric0071",
           "ph_ppi_m_wbgdi",
           "thpric0036",
           "tw_ppi_m_wbgdi",
           "ih:mb:com:pe_ppi_joined:pe_ppi_joined",
           "id_ppi_m_wbgdi"
                      ]

ppi_rat = mb.FetchSeries(tickers)

new_column_names = {"ih:mb:com:br_ppi_joined:br_ppi_joined" : "ppi_rat_br",
           "ih:mb:com:cl_ppi_joined:cl_ppi_joined" : "ppi_rat_cl",
           "mxpric7351" : "ppi_rat_mx",
           "czpric0076" : "ppi_rat_cz",
           "plpric0065" : "ppi_rat_pl",
           "cnpric4601" : "ppi_rat_cn",
           "inpric0881" : "ppi_rat_in",
           "krpric0014" : "ppi_rat_kr",
           "copric1010" : "ppi_rat_co",
           "hupric0066" : "ppi_rat_hu",
           "trpric0244" : "ppi_rat_tr",
           "ilpric0018" : "ppi_rat_il",
           "ih:mb:com:za_ppi_joined:za_ppi_joined" : "ppi_rat_za",
           "mypric0071" : "ppi_rat_my",
           "phpric0019" : "ppi_rat_ph",
           "ph_ppi_m_wbgdi" : "ppi_rat_th",
           "tw_ppi_m_wbgdi" : "ppi_rat_tw",
           "ih:mb:com:pe_ppi_joined:pe_ppi_joined" : "ppi_rat_pe",
           "id_ppi_m_wbgdi" : "ppi_rat_id"    
           }

ppi_rat = ppi_rat.rename(columns = new_column_names)
ppi_rat_3mma = ppi_rat.rolling(window=3).mean()
ppi_rat_qoq_3mma = ppi_rat_3mma.pct_change(periods=3) * 100
ppi_rat_qoq_3mma_d = ppi_rat_qoq_3mma.resample('D').ffill()
ppi_rat_qoq_3mma_d = apply_backfill_after_first_observation(ppi_rat_qoq_3mma_d)
ppi_rat_qoq_3mma_d_lggd = ppi_rat_qoq_3mma_d.shift(30)
print(ppi_rat_qoq_3mma_d_lggd.head())

tickers = ["ECSRBR Q125 Index",
           "ECCBCL Q125 Index",
           "ECSRMX Q125 Index",
           "ECCBCZ Q125 Index",
           "ECCBPL Q125 Index",
           "ECCBCN Q125 Index",
           "ECCBIN Q125 Index",
           "ECCBKR Q125 Index",
           "ECCBCO Q125 Index",
           "ECCBHU Q125 Index",
           "ECCBTR Q125 Index",
           "ECCBIL Q125 Index",
           "ECCBZA Q125 Index",
           "ECCBMY Q125 Index",
           "ECCBPH Q125 Index",
           "ECCBTH Q125 Index",
           "ECCBTW Q125 Index",
           "ECCBPE Q125 Index",
           "ECCBID Q125 Index"    ]

dt_from = dt.date(1990,1,1)
dt_to = dt.date.today()

cons_rat = get_bloomberg_date(tickers,dt_from,dt_to, field = 'PX_LAST', periodicity = 'DAILY')

cons_rat = cons_rat.rename(columns={"ECSRBR Q125 Index" : "Brazil",
           "ECCBCL Q125 Index" : "Chile",
           "ECSRMX Q125 Index" : "Mexico",
           "ECCBCZ Q125 Index" : "Czechia",
           "ECCBPL Q125 Index" : "Poland",
           "ECCBCN Q125 Index" : "China",
           "ECCBIN Q125 Index" : "India",
           "ECCBKR Q125 Index" : "Korea",
           "ECCBCO Q125 Index" : "Colombia",
           "ECCBHU Q125 Index" : "Hungary",
           "ECCBTR Q125 Index" : "Turkey",
           "ECCBIL Q125 Index" : "Israel",
           "ECCBZA Q125 Index" : "South Africa",
           "ECCBMY Q125 Index" : "Malaysia",
           "ECCBPH Q125 Index" : "Philippines",
           "ECCBTH Q125 Index" : "Thailand",
           "ECCBTW Q125 Index" : "Taiwan",
           "ECCBPE Q125 Index" : "Peru",
           "ECCBID Q125 Index" : "Indonesia" 
           })

print(cons_rat.tail())

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

file_path = r'C:\\Users\\MR99924\\workspace\\vscode\\Projects\\EM_RatesRF\\CBDecisions.xlsx'
rate_decisions = pd.read_excel(file_path, sheet_name='RateDecisions', engine='openpyxl', usecols=range(19))
print(rate_decisions.head())

rate_decisions = pd.DataFrame(rate_decisions)
for country in rate_decisions.columns:
    rate_decisions[country] = pd.to_datetime(rate_decisions[country])

# Dictionary to store results for each country
results_dict = {}

country_code_mapping = {
    'Brazil': 'br',
    'Chile': 'cl',
    'Mexico': 'mx',
    'Czechia': 'cz',
    'Poland': 'pl',
    'China': 'cn',
    'India': 'in',
    'Turkey': 'tr',
    'Korea': 'kr',
    'Colombia': 'co',
    'Hungary': 'hu',
    'Israel': 'il',
    'South Africa': 'za',
    'Malaysia': 'my',
    'Philippines': 'ph',
    'Thailand': 'th',
    'Taiwan': 'tw',
    'Peru': 'pe',
    'Indonesia': 'id'
}

def round_to_nearest_25bps_except_10bps(rate_change):
    if math.isnan(rate_change):
        return float('nan')
    if abs(rate_change) < 0.10:
        return 0
    elif abs(rate_change) == 0.10:
        return 0.10
    return round(rate_change * 4) / 4

# Iterate over each country in the decision dates data

# Print the results for each country - just to check this lines up with everything
for country in rate_decisions.columns:
    decision_dates = rate_decisions[country].dropna()
    decision_dates = decision_dates[decision_dates >= pd.Timestamp('1990-01-01')]
    
    # Get the policy rate data for the country using the correct code from the mapping
    policy_rate_column = f'rate_{country_code_mapping[country]}'  # Adjusted to match the renamed columns
    policy_rates = rate_data[policy_rate_column]
    
    # Extract the policy rates on the decision dates
    rates_on_decision_days = policy_rates.loc[decision_dates]
    
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

for country in rate_decisions.columns:
    print(country)
    
    country_code = country_code_mapping[country]
    
    features = pd.concat([h_inf_qoq_3mma_d_lggd.filter(regex=f'_{country_code}$'),
                          c_inf_qoq_3mma_d_lggd.filter(regex=f'_{country_code}$'),
                          neer_data_qoq_3mma_d.filter(regex=f'_{country_code}$'),
                          pca_act_d_lggd.filter(regex=f'_{country_code}$'),
                          u_rat_12mma_sad_d_lggd.filter(regex=f'_{country_code}$'),
                          ca_rat_12mma_sad_d_lggd.filter(regex=f'_{country_code}$'),
                          fb_rat_12mma_sad_d_lggd.filter(regex=f'_{country_code}$'),
                          embig_data.filter(regex=f'_{country_code}$'),
                          curr_vol.filter(regex=f'_{country_code}$'),
                          res_rat_12mma_sad_d.filter(regex=f'_{country_code}$'),
                          ppi_rat_qoq_3mma_d_lggd.filter(regex=f'_{country_code}$'),
                          dm_rate_data_qd_3mma['FED'],
                          ], axis=1)
    
    features.index = pd.to_datetime(features.index)
    features = features.loc[features.index >= '1990-01-01']
     
    target_df = results_dict[country]
    target = target_df
    
    try:
        target.set_index(f'decision_date_{country_code}', inplace=True)
        target.index = pd.to_datetime(target.index)
    except Exception as e:
        print(f"An error occurred in setting the index for {country}: {e}. Proceeding to the next one.")

    features = features.dropna()
    target = target.dropna()

    # Remove binning and label encoding steps
    # target[f'policy_rate_change_{country_code}'] = pd.cut(target[f'policy_rate_change_{country_code}'], bins=bins, labels=labels, include_lowest=True)
    # target[f'policy_rate_change_{country_code}'] = label_encoder.fit_transform(target[f'policy_rate_change_{country_code}'])
    # label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    # label_mappings[country] = label_mapping

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    target.replace([np.inf, -np.inf], np.nan, inplace=True)

    features.dropna(inplace=True)
    target.dropna(inplace=True)
    
    common_indices = features.index.intersection(target.index)
    features = features.loc[common_indices]
    target = target.loc[common_indices]
    features = features[~features.index.duplicated(keep='first')]
    target = target[~target.index.duplicated(keep='first')]

    base_rf_model = RandomForestRegressor(random_state=42, oob_score=True)  # Change 1: Use RandomForestRegressor

    previous_accuracy = 0
    for n_estimators in [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]:
        base_rf_model.set_params(n_estimators=n_estimators)
        base_rf_model.fit(features, target)
    
    model_filename = f'{country}_rf_model.joblib'
    dump(base_rf_model, model_filename)
        
    print(f"Length of features: {len(features)}")
    print(f"Length of target: {len(target)}")
    if len(features) != len(target):
        print(f"Skipping {country} due to inconsistent number of samples.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    print(f"{country} - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    if X_train.empty:
        print(f"No features training data available for {country}. Skipping this country.")
        continue
    if y_train.empty:
        print(f"No target training data available for {country}. Skipping this country.")
        continue
    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    rf_model = RandomForestRegressor(random_state=42, oob_score=True)  # Change 1: Use RandomForestRegressor

    initial_rf_model = RandomForestRegressor(random_state=42, oob_score=True)  # Change 1: Use RandomForestRegressor
    initial_rf_model.fit(X_train, y_train)

    importances = initial_rf_model.feature_importances_

    n_top_features = 12
    indices = np.argsort(importances)[-n_top_features:]
    selected_features = X_train.columns[indices]

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    previous_accuracy = 0
    for n_estimators in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        rf_model.set_params(n_estimators=n_estimators)
        rf_model.fit(X_train_selected, y_train)
        y_val_pred = rf_model.predict(X_test_selected)
        accuracy = r2_score(y_test, y_val_pred)  # Change 2: Use r2_score
        print(f'n_estimators: {n_estimators}, Validation R^2: {accuracy}')
        if accuracy <= previous_accuracy:
            break
        previous_accuracy = accuracy

    oob_error = 1 - rf_model.oob_score_
    
    y_pred = rf_model.predict(X_test_selected)

    mse = mean_squared_error(y_test, y_pred)  # Change 2: Use mean_squared_error
    r2 = r2_score(y_test, y_pred)  # Change 2: Use r2_score
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    results_dict[country] = {
        'model': rf_model,
        'mse': mse,
        'r2': r2,
        'X_test_selected': X_test_selected,
        'y_test': y_test,
        'y_pred': y_pred,
        'oob_error': oob_error}

    print(results_dict)

output_dir = r'C:\Users\MR99924\workspace\vscode\Projects\EM_RatesRF'

dfs = {}

for country, results in results_dict.items():
    try:
        if 'X_test_selected' not in results:
            raise KeyError('X_test key is missing')
        df_results = pd.DataFrame({
            'X_test_selected': results['X_test_selected'].index,
            'y_test': results['y_test'],
            'y_pred': results['y_pred'],
            'mse': [results['mse']] * len(results['y_test']),
            'r2': [results['r2']] * len(results['y_test']),
            'oob_error': [results['oob_error']] * len(results['y_test'])})
        dfs[country] = df_results
    except KeyError as e:
        print(f"KeyError for {country}: {e}. Proceeding to the next one.")
    except Exception as e:
        print(f"An error occurred for {country}: {e}. Proceeding to the next one.")
        traceback.print_exc()

with pd.ExcelWriter(os.path.join(output_dir, 'merged_rf_model_results.xlsx')) as writer:
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



def plot_feature_importance(model, features, country):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'{country}: Feature Importance')
    plt.show()

def plot_prediction_error(model, X_test, y_test, country):
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{country}: Prediction Error Plot')
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


def plot_residuals(model, X_test, y_test, country):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{country}: Residual Plot')
    plt.grid()
    plt.show()

## Iterate over each country in the results_dict
for country, results in results_dict.items():
    print(country)
    try:
        model = results['model']
        X_test = results['X_test_selected']
        y_test = results['y_test']

        if len(X_test) != len(y_test):
            print(f"Inconsistent number of samples for {country}: X_test has {len(X_test)} samples, but y_test has {len(y_test)} samples. Skipping this country.")
            continue
        
        # Plot learning curves
        #plot_learning_curve(model, X_test, y_test, country)

        # Plot feature importance
        #plot_feature_importance(model, X_test, country)
        
        # Plot prediction error
        #plot_prediction_error(model, X_test, y_test, country)
        
        # Plot cumulative gain and lift charts
        #plot_cumulative_gain_lift(model, X_test, y_test, country)
        
        # Plot residuals
        #plot_residuals(model, X_test, y_test, country)
        
    except KeyError as e:
        print(f"KeyError for {country}: {e}. Proceeding to the next one.")
    except Exception as e:
        print(f"An error occurred for {country}: {e}. Proceeding to the next one.")

# Function to calculate the probability of a continuous variable being greater or less than a set value
def calculate_continuous_probabilities(model, X_test, threshold):
    # Get the predicted values
    predicted_values = model.predict(X_test)
    
    # Calculate the probabilities
    probabilities_greater = (predicted_values > threshold).mean()
    probabilities_less = (predicted_values <= threshold).mean()
    
    # Create a DataFrame to store the probabilities
    probabilities_df = pd.DataFrame({
        'Greater than threshold': [probabilities_greater],
        'Less than or equal to threshold': [probabilities_less]
    })
    
    return probabilities_df

# Extend feature datasets to the current date
def extend_features_to_today(features):
    # Get the latest date in the existing data
    latest_date = pd.to_datetime(features.index.max())
    print(f"Latest date in features: {latest_date}")
    
    # Generate a date range from the latest date to today
    today = pd.to_datetime('today').normalize()
    print(f"Today's date: {today}")
    new_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), end=today, freq='D')
    print(f"New dates to add: {new_dates}")
    
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

# Create a dictionary to store the probabilities DataFrames for each country
probabilities_dict = {}

def plot_normal_distribution(prediction, std_dev, country):
    # Generate data for the normal distribution
    x = np.linspace(prediction - 3*std_dev, prediction + 3*std_dev, 1000)
    y = norm.pdf(x, prediction, std_dev)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Normal Distribution')
    plt.axvline(prediction, color='r', linestyle='--', label='Prediction')
    plt.fill_between(x, y, alpha=0.2)
    
    # Add labels and title
    plt.xlabel('Rate Change')
    plt.ylabel('Probability Density')
    plt.title(f'Normal Distribution around Prediction for {country}')
    plt.legend()
    
    # Show the plot
    plt.show()


# Iterate over each country in the results_dict to calculate probabilities and store them in the dictionary
for country, results in results_dict.items():
    print(f"Processing country: {country}")

    if country in country_code_mapping:
        country_code = country_code_mapping[country].strip().lower()
        print(f"Country code for {country}: {country_code}")
    
    model_filename = f'{country}_rf_model.joblib'
    rf_model = load(model_filename)
    
    features_extd = extend_features_to_today(results['X_test_selected'])
    
    rate_data_filtered = rate_data.filter(regex=f'_{country_code}$')
    print("Filtered rate_data:")
    print(rate_data_filtered)
    rate_data_extd = extend_features_to_today(rate_data_filtered)
    print(f"Extended rate_data for {country}:")
    print(rate_data_extd)
    print(f"Shape of extended features: {features_extd.shape}")
    
    print("All column names in rate_data_extd:")
    print(rate_data_extd.columns.tolist())
    
    print("Data types of column names in rate_data_extd:")
    print(rate_data_extd.columns.map(type).tolist())
    
    features_extd.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_extd.ffill(inplace=True)
    print(features_extd)
    
    today_date = pd.to_datetime('today').normalize()
    print(today_date)
    todays_features = features_extd.loc[today_date].to_frame().T
    print(f"Today's features for {country} before reindexing:")
    print(todays_features)
    print(f"Shape of today's features before reindexing: {todays_features.shape}")

    expected_features = results['X_test_selected'].columns
    print(f"Expected features for {country}: {expected_features}")
    todays_features = todays_features.reindex(columns=expected_features, fill_value=0)
    print(f"Today's features for {country} after reindexing:")
    print(todays_features)
    print(f"Shape of today's features after reindexing: {todays_features.shape}")
    
    todays_rate = rate_data_extd.loc[today_date]
    todays_rate_value = todays_rate.values[0]
    print(todays_rate.values[0])

    print(f"Checking for date {today_date} and country {country} in cons_rat")
    print(f"cons_rat index: {cons_rat.index}")
    print(f"cons_rat columns: {cons_rat.columns}")
    
    if today_date in cons_rat.index and country in cons_rat.columns:
        print(country_code)
        todays_cons_rat = cons_rat.loc[today_date, country].squeeze()
        print(f"Today's consensus_rate for {country}: {todays_cons_rat}")
        print("Extended rate_data:")
        print(rate_data_extd)

        rate_data_extd.columns = rate_data_extd.columns.str.strip().str.lower()
        print("All column names in rate_data_extd after stripping and lowering:")
        print(rate_data_extd.columns.tolist())
        print("Data types of column names in rate_data_extd after stripping and lowering:")
        print(rate_data_extd.columns.map(type).tolist())
        
        expected_rate_change =  todays_cons_rat - todays_rate_value
        print(f"Expected rate change for {country}: {expected_rate_change}")
        
        if not todays_features.isnull().values.any() and np.isfinite(todays_features).all().all():
            try:
                threshold = expected_rate_change
                
                probabilities = calculate_continuous_probabilities(rf_model, todays_features, threshold)
                
                probabilities.index = todays_features.index
                
                probabilities_dict[country] = probabilities
                
                print(f"Probabilities for {country}: {probabilities}")

                # Generate predictions for the extended features
                predictions = rf_model.predict(features_extd)  # <-- Added this line
                
                # Calculate the standard deviation of the predictions
                std_dev = np.std(predictions)  # <-- Added this line
                print(f"Standard deviation of predictions: {std_dev}")  # <-- Added this line

                # Plot normal distribution around the prediction
                plot_normal_distribution(threshold, std_dev, country)  # <-- Modified this line to use std_dev

            except ValueError as e:
                print(f"Error calculating probabilities for {country}: {e}")
        else:
            print(f"Today's features for {country} contains NaN or infinite values after cleaning.")
    else:
        print(f"Date {today_date} or country {country} not found in cons_rat or country_code not in rate_data_extd columns.")

with pd.ExcelWriter(os.path.join(output_dir, 'probabilities_results.xlsx')) as writer:
    for country, df in probabilities_dict.items():
        df.to_excel(writer, sheet_name=country)

print("The probabilities have been successfully saved to an Excel file with different sheets.")