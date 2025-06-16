import sys
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import bloomberg
import traceback
import pandas as pd
import datetime as dt
import numpy as np
from macrobond import Macrobond
mb = Macrobond()

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