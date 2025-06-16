import numpy as np
import os
import sys
import os
import datetime as dt
import logging
import config
import traceback
import pandas as pd
sys.path.append(r'C:\Users\MR99924\workspace\vscode\Projects\assetallocation-research\data_etl')
import bloomberg
import logging
from macrobond import Macrobond
from sklearn.preprocessing import LabelEncoder

# Initialize API connectors
mb = Macrobond()
label_encoder = LabelEncoder()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA RETRIEVAL PARAMETERS
# =============================================================================

CPI_TICKERS = ["CPI YOY Index"]
ACT_TICKERS = ["GSUSCAI Index"]

# Default data fetch date range
DEFAULT_DATE_FROM = dt.date(1970, 1, 1)
DEFAULT_DATE_TO = dt.date.today()

# Path to folder containing text files
FOLDER_PATH = r"C:\Users\MR99924\workspace\vscode\Projects\US_BeigeBook\Text_files(1974-Present)"

# Bloomberg parameters
BLOOMBERG_FIELD = "PX_LAST"
BLOOMBERG_DAILY_PERIODICITY = "DAILY"
BLOOMBERG_MONTHLY_PERIODICITY = "MONTHLY"
BLOOMBERG_NON_TRADING_DAY_FILL = "ALL_CALENDAR_DAYS"
BLOOMBERG_NON_TRADING_DAY_METHOD = "PREVIOUS_VALUE"

def get_bloomberg_date(tickers, date_from, date_to, field="PX_LAST", periodicity="DAILY"):
    """
    Fetch data from Bloomberg for the given tickers and date range.
    
    Parameters:
        tickers: list - List of Bloomberg tickers
        date_from: datetime.date - Start date for data
        date_to: datetime.date - End date for data
        field: str - Bloomberg field (default: "PX_LAST")
        periodicity: str - Data frequency (default: "DAILY")
        
    Returns:
        DataFrame: Data for requested tickers and date range
    """
    try:
        bbg = bloomberg.Bloomberg()
        df = bbg.historicalRequest(tickers,
                                field,
                                date_from,
                                date_to,
                                periodicitySelection=periodicity,
                                nonTradingDayFillOption="ALL_CALENDAR_DAYS",
                                nonTradingDayFillMethod="PREVIOUS_VALUE",
                                )
        
        # Pivot the data to get a clean dataframe with dates as index
        df = pd.pivot_table(df,
                        values='bbergvalue',
                        index=['bbergdate'],
                        columns=['bbergsymbol'],
                        aggfunc=np.max,
                        )
        
        # Ensure all requested tickers are in the final DataFrame
        for ticker in tickers:
            if ticker not in df.columns:
                logger.warning(f"Ticker {ticker} not found in Bloomberg data")
        
        # Keep only the requested tickers that were found
        existing_tickers = [t for t in tickers if t in df.columns]
        if existing_tickers:
            df = df[existing_tickers]
        else:
            logger.warning("None of the requested tickers were found")
            df = pd.DataFrame(index=pd.date_range(date_from, date_to))
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching Bloomberg data: {e}")
        return pd.DataFrame(index=pd.date_range(date_from, date_to))

def classify_period(inflation, growth):
    if inflation > 2 and growth > 2:
        return 1  # High growth, high inflation
    elif inflation < 0 and growth < 0:
        return 2  # Low growth, deflationary
    elif inflation > 2 and growth < 2:
        return 3  # High inflation, low growth
    elif inflation < 2 and growth > 2:
        return 4  # Low inflation, high growth
    # Add more conditions as needed
    return 0  # Default label