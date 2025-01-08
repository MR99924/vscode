# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:01:26 2025

@author: MR99924
"""
import sys
sys.path.append(r'S:\Shared\Front Office\Asset Allocation\Analytics\libs')
import pandas as pd
import bloomberg
import numpy as np
import datetime as dt

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
           "BISPDHPO Index": "rate_po",
           "BISPDHCH Index": "rate_ch",
           "BISPDHIN Index": "rate_in",
           "BISPDHSK Index": "rate_sk",
           "BISPDHCO Index": "rate_co",
           "BISPDHHU Index": "rate_hu",
           "1866582 Index": "rate_tu",
           "BISPDHIS Index": "rate_is",
           "BISPDHSA Index": "rate_sa",
           "BISPDHMA Index": "rate_ma",
           "BISPDHPH Index": "rate_ph",
           "BISPDHTH Index": "rate_th",
           "TAREDSC Index": "rate_tw",
           "BISPDHPE Index": "rate_pe",
           "5366582 Index": "rate_id"})

rate_data_pop = rate_data.diff()

print(rate_data_pop.head())
