# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:51:46 2024

@author: MR99924
"""
import sys
sys.path.append(r'C:\\Users\\MR99924\\workspace\\vscode\\Projects\\assetallocation-research\\data_etl')
import pandas as pd
import datetime as dt
import bloomberg
import math
from macrobond import Macrobond
import numpy as np
import matplotlib.pyplot as plt

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

mb = Macrobond()

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

print(ca_rat_12mma_sad_d)