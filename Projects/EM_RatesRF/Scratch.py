# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:51:46 2024

@author: MR99924
"""
import sys
sys.path.append(r'S:\Shared\Front Office\Asset Allocation\Analytics\libs')
import pandas as pd
import bloomberg
import numpy as np




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




get_bloomberg_date("GUKG10 Index", "01/07/2020", "19/12/2024")



