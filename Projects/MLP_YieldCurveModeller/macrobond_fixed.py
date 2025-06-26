import pandas as pd
# import pytz
import datetime as dt
# import numpy as np
import win32com.client
# from typing import Tuple
from dataclasses import dataclass, field

from macrobond_api_constants import SeriesFrequency as sf
# from macrobond_api_constants import CalendarDateMode as cd
# from macrobond_api_constants import CalendarMergeMode as cm
# from macrobond_api_constants import MetadataValueType as md
from macrobond_api_constants import SeriesMissingValueMethod as mv
from macrobond_api_constants import SeriesPartialPeriodsMethod as pp
from macrobond_api_constants import SeriesToHigherFrequencyMethod as hf
from macrobond_api_constants import SeriesToLowerFrequencyMethod as lf
from macrobond_api_constants import SeriesWeekdays as sw


"""
pip install pywin32
python venv\Scripts\pywin32_postinstall.py -install
"""


class SeriesOptions(object):
    """Class to aggregate options into a single location."""
    series_frequency = sf
    missing_values = mv
    partial_periods = pp
    to_higher_frequency = hf
    to_lower_frequency = lf


@dataclass
class MbSeries:
    """Series object for unified series requests."""
    ticker: str
    missing_value_method: int = field(default=mv.AUTO)
    partial_periods_method: int = field(default=pp.NONE)
    to_higher_frequency_method: int = field(default=hf.AUTO)
    to_lower_frequency_method: int = field(default=lf.AUTO)


class Macrobond(object):
    def __init__(self):
        # Initiate win32com connection to Macrobond
        c = win32com.client.Dispatch('Macrobond.Connection')
        # Create Database connection
        db = c.Database

        # Create default attribute with all regions
        region_map = self.region_map()
        region_list = list(region_map.keys())

        # Save attributes
        self._mbdb = db
        self._region_list_all = region_list

    @property
    def region_list_all(self):
        return self._region_list_all

    def FetchOneSeries(self, ticker: str) -> pd.DataFrame:
        """
        Fetch one timeseries
        """
        try:
            # Initiate series
            series = self._mbdb.FetchOneSeries(ticker)

            # Check for errors
            if hasattr(series, 'IsError') and series.IsError:
                print(f'Error fetching ticker {ticker}: {series.ErrorMessage}')
                return pd.DataFrame()

            # Convert end dates
            p_end_dates = pd.to_datetime([date.strftime('%Y-%m-%d') for date in series.DatesAtEndOfPeriod])

            # Return DataFrame
            df = pd.DataFrame(series.Values, index=p_end_dates)
            df.columns = [series.Name]

            return df
            
        except Exception as e:
            print(f'Exception fetching ticker {ticker}: {str(e)}')
            return pd.DataFrame()

    def FetchSeries(self, ticker_list: [str]) -> pd.DataFrame:
        """
        Fetch several series and return a dataframe.
        """

        # Assert input type
        try:
            assert type(ticker_list) is list
        except AssertionError as ae:
            print('Input must be a list')
            raise ae

        if not ticker_list:
            print('Ticker list is empty')
            return pd.DataFrame()

        try:
            # Fetch all the series
            series = self._mbdb.FetchSeries(ticker_list)
            
            # Check for errors in individual series and filter out failed ones
            valid_tickers = []
            valid_series = []
            
            for i, ticker in enumerate(ticker_list):
                try:
                    if hasattr(series[i], 'IsError') and series[i].IsError:
                        print(f'Error fetching ticker {ticker}: {series[i].ErrorMessage}')
                        continue
                    else:
                        valid_tickers.append(ticker)
                        valid_series.append(series[i])
                except Exception as e:
                    print(f'Exception checking ticker {ticker}: {str(e)}')
                    continue
            
            if not valid_tickers:
                print("No valid tickers found")
                return pd.DataFrame()
            
            # Convert valid series to DataFrame
            df = self.__m_series_tuple_to_df(ticker_list=valid_tickers, series=valid_series)
            return df
            
        except Exception as e:
            print(f"Error in FetchSeries: {str(e)}")
            return pd.DataFrame()

    def CreateUnifiedSeriesRequest(self, series_list: [MbSeries], StartDate: str= "", EndDate: str = "", Currency: str = None, Frequency: int = sf.HIGHEST) -> pd.DataFrame:
        """
        Function to extract several series with merged properties e.g. currency, frequency
        Currency codes used in Macrobond: https://www.macrobond.com/currency-list/
        """

        try:
            # Create the request
            req = self._mbdb.CreateUnifiedSeriesRequest()
            req.StartDate = StartDate
            req.EndDate = EndDate
            req.Currency = Currency
            req.Frequency = Frequency

            # Add all tickers to the request
            for i, series in enumerate(series_list):
                req.AddSeries(series.ticker)
                req.AddedSeries[i].MissingValueMethod = series.missing_value_method
                req.AddedSeries[i].PartialPeriodsMethod = series.partial_periods_method
                req.AddedSeries[i].ToHigherFrequencyMethod = series.to_higher_frequency_method
                req.AddedSeries[i].ToLowerFrequencyMethod = series.to_lower_frequency_method

            # Fetch the data
            series_request = self._mbdb.FetchSeries(req)

            # Convert to pd.DataFrame
            df = self.__m_series_tuple_to_df(ticker_list=[s.ticker for s in series_list], series=series_request)
            return df
            
        except Exception as e:
            print(f"Error in CreateUnifiedSeriesRequest: {str(e)}")
            return pd.DataFrame()

    def UploadOneOrMoreSeries(self, df: pd.DataFrame, region_list: list, description_list: list, category_list: list, frequency):
        """
        Method to upload in-house series to Macrobond.
        """

        # Macrobond database cannot cast between types so cast everything to float before upload
        df = df.astype(float)

        # Extract tickers
        ticker_list = df.columns.to_list()

        # Create Empty Metadata container
        m = self._mbdb.CreateEmptyMetadata()

        # Check that region is valid
        for region in region_list:
            try:
                assert region.lower() in self.region_list_all
            except AssertionError as ae:
                print(f'Region {region} is not valid')
                raise ae

        # Example ticker to upload to private account: ih:mb:priv:series1_example
        # Example ticker to upload to department account: ih:mb:dept:series1_example
        # Example ticker to upload to company account: ih:mb:com:series1_example

        # Check ticker validity
        for ticker in ticker_list:
            self.f_check_inhouse_ticker_validity(ticker)

        # Convert index to list and set timezone
        dates = df.index.to_list()
        dates = [dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc) for date in dates]

        # Upload each series
        for i, (ticker, values) in enumerate(df.items()):
            s = self._mbdb.CreateSeriesObject(name=ticker, description=description_list[i], region=region_list[i],
                                              category=category_list[i], frequency=frequency,
                                              dayMask=sw.FULLWEEK, startDateOrDates=dates, values=values,
                                              metadata=m)
            self._mbdb.UploadOneOrMoreSeries(series=s)
            print(f'Ticker: {ticker} uploaded.')

        return None

    def AddRef(self):
        return self._mbdb.AddRef()

    def CreateDerivedMetadata(self):
        return self._mbdb.CreateDerivedMetadata()

    def CreateEmptyMetadata(self):
        return self._mbdb.CreateEmptyMetadata()

    def CreateSearchQuery(self):
        return self._mbdb.CreateSearchQuery()

    def CreateSeriesObject(self):
        return self._mbdb.CreateSeriesObject()

    def CreateSeriesObjectWithForecastFlags(self):
        return self._mbdb.CreateSeriesObjectWithForecastFlags()

    def DeleteOneOrMoreSeries(self):
        return self._mbdb.DeleteOneOrMoreSeries()

    def FetchEntities(self):
        return self._mbdb.FetchEntities()

    def FetchOneEntity(self):
        return self._mbdb.FetchOneEntity()

    def FetchOneSeriesWithRevisions(self):
        return self._mbdb.FetchOneSeriesWithRevisions()

    def FetchSeriesWithRevisions(self):
        return self._mbdb.FetchSeriesWithRevisions()

    def GetIDsOfNames(self):
        return self._mbdb.GetIDsOfNames()

    def GetMetadataInformation(self):
        return self._mbdb.GetMetadataInformation()

    def GetTypeInfo(self):
        return self._mbdb.GetTypeInfo()

    def GetTypeInfoCount(self):
        return self._mbdb.GetTypeInfoCount()

    def Invoke(self):
        return self._mbdb.Invoke()

    def QueryInterface(self):
        return self._mbdb.QueryInterface()

    def Release(self):
        return self._mbdb.Release()

    def Search(self):
        return self._mbdb.Search()

    def __m_series_tuple_to_df(self, ticker_list: [str], series) -> pd.DataFrame:
        """
        Convert series request to a pandas DataFrame
        """
        try:
            unpacked_series = []
            valid_tickers = []
            
            for i, ticker in enumerate(ticker_list):
                try:
                    unpacked = self.f_unpack_series(series[i])
                    if not unpacked.empty:
                        unpacked_series.append(unpacked)
                        valid_tickers.append(ticker)
                except Exception as e:
                    print(f"Error unpacking series for ticker {ticker}: {str(e)}")
                    continue
            
            if not unpacked_series:
                print("No valid series to convert to DataFrame")
                return pd.DataFrame()
            
            df = pd.concat(unpacked_series, axis=1)
            df.columns = valid_tickers
            return df
            
        except Exception as e:
            print(f"Error converting series to DataFrame: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def f_unpack_series(series) -> pd.Series:
        """
        Convert macrobond series request to pandas series
        """
        try:
            # Check if series has error
            if hasattr(series, 'IsError') and series.IsError:
                print(f'Series error: {series.ErrorMessage}')
                return pd.Series()
            
            # Convert dates
            p_end_dates = pd.to_datetime([date.strftime('%Y-%m-%d') for date in series.DatesAtEndOfPeriod])

            return pd.Series(series.Values, index=p_end_dates)
            
        except Exception as e:
            print(f"Error unpacking series: {str(e)}")
            return pd.Series()

    @staticmethod
    def f_check_inhouse_ticker_validity(ticker: str):
        """
        Function to check in-house tickers are valid
        """

        # Check that tickers are defined as in-house & macrobond
        try:
            assert ticker[0:5].lower() == 'ih:mb'
        except AssertionError as ae:
            print(f'Ticker: {ticker} is not valid. Has to start with "ih:mb"')
            raise ae

        # Check that ticker storage is ok
        try:
            assert ticker[6:10].lower() in ['priv', 'dept'] or ticker[6:9].lower() in ['com']
        except AssertionError as ae:
            print(f'Ticker: {ticker} is not valid. Storage has to be "priv", "dept", or "com"')
            raise ae

        return None

    @staticmethod
    def f_create_bbg_ticker(bbg_ticker: [str], **kwargs) -> [str]:
        """
        Function to format Bloomberg ticker as Macrobond tickers

        Example input:
        bbg_ticker = ['MXWO Index', 'MXWO000G index', 'ECSURPUS index']
        kwargs = {'BBG_Fields': ['PX_LAST', 'PX_OPEN', '']}
        """

        bbg_field_tf = False
        bbg_fields = []

        try:
            assert type(bbg_ticker) is list
        except AssertionError as ae:
            print(f"Ticker input must be a list. Error: {ae}")
            return []

        # Extract kwargs
        for key, val in kwargs.items():
            if key.lower() == 'bbg_fields':
                bbg_field_tf = True
                bbg_fields: list = kwargs.get('BBG_Fields')
            else:
                raise KeyError(f'Kwargs key: {key} not defined')

        # If we use fields then make sure lengths are correct
        if bbg_field_tf:
            try:
                assert len(bbg_ticker) == len(bbg_fields)
            except AssertionError as ae:
                print(f'Length of bbg_ticker must be the same as bbg_field. Error: {ae}')
                return []

        # Pre-allocate output array
        macrobond_tickers = []

        if bbg_fields:
            for ticker, field in zip(bbg_ticker, bbg_fields):

                # If a specific field entry is empty then don't add that one (some series don't need a field)
                if len(field) == 0:
                    macrobond_tickers.append(f'ih:bl:{ticker.lower()}')
                else:
                    macrobond_tickers.append(f'ih:bl:{ticker.lower()}:{field.lower()}')

        else:
            for ticker in bbg_ticker:
                macrobond_tickers.append(f'ih:bl:{ticker.lower()}')

        return macrobond_tickers

    @staticmethod
    def region_map():
        """
        Region shortnames provided by Macrobon at
        https://www.macrobond.com/region-list/

        Rather than hard coded could do:
        region_map = pd.read_csv(r'.\data_etl\regions.csv', encoding="windows-1252")
        region_map = region_map.set_index('Code').to_dict()['Description']
        Then you can have a function that updates the csv straight from the web.
        """
        region_map = {
            'advecos': 'World, Advanced',
            'af': 'Afghanistan',
            'africa': 'Africa',
            'al': 'Albania',
            'dz': 'Algeria',
            'as': 'American Samoa',
            'ad': 'Andorra',
            'ao': 'Angola',
            'ai': 'Anguilla',
            'aq': 'Antarctica',
            'ag': 'Antigua & Barbuda',
            'ar': 'Argentina',
            'am': 'Armenia',
            'aw': 'Aruba',
            'asean5': 'ASEAN-5',
            'asia': 'Asia',
            'asiapjp': 'Asia + Japan',
            'asiaxjp': 'Asia Excluding Japan',
            'asiaxcn': 'Asia Excluding China',
            'apac': 'Asia Pacific',
            'apacxjp': 'Asia Pacific Excluding Japan',
            'au': 'Australia',
            'auasia': 'Australasia',
            'at': 'Austria',
            'az': 'Azerbaijan',
            'bs': 'Bahamas',
            'bh': 'Bahrain',
            'balkan': 'Balkan Countries',
            'bd': 'Bangladesh',
            'bb': 'Barbados',
            'by': 'Belarus',
            'be': 'Belgium',
            'bz': 'Belize',
            'bj': 'Benin',
            'bm': 'Bermuda',
            'bt': 'Bhutan',
            'bo': 'Bolivia',
            'ba': 'Bosnia & Herzegovina',
            'bw': 'Botswana',
            'bv': 'Bouvet Island',
            'br': 'Brazil',
            'bric': 'BRIC Countries',
            'io': 'British Indian Ocean Territory',
            'bn': 'Brunei',
            'bg': 'Bulgaria',
            'bf': 'Burkina Faso',
            'mm': 'Myanmar (Burma)',
            'bi': 'Burundi',
            'kh': 'Cambodia',
            'cm': 'Cameroon',
            'ca': 'Canada',
            'cv': 'Cape Verde',
            'ky': 'Cayman Islands',
            'caeeuro': 'Central & Eastern Europe',
            'ssam': 'South & Central America',
            'cf': 'Central African Republic',
            'td': 'Chad',
            'cl': 'Chile',
            'cn': 'China',
            'cx': 'Christmas Island',
            'cc': 'Cocos (Keeling) Islands',
            'co': 'Colombia',
            'cis': 'CIS (Commonwealth of Independent States)',
            'cw': 'Curaçao',
            'km': 'Comoros',
            'cg': 'Congo',
            'cd': 'Congo (Democratic Republic)',
            'ck': 'Cook Islands',
            'cr': 'Costa Rica',
            'hr': 'Croatia',
            'cu': 'Cuba',
            'cy': 'Cyprus',
            'cz': 'Czech Republic',
            'csz': 'Czechoslovakia',
            'dk': 'Denmark',
            'devasia': 'Asia, Developing',
            'dvmkts': 'World, Developed',
            'dj': 'Djibouti',
            'dm': 'Dominica',
            'do': 'Dominican Republic',
            'dynasian': 'Dynamic Asian Economies',
            'dd': 'East Germany',
            'eeuro': 'Eastern Europe',
            'cemac': 'Central African Economic & Monetary Community (CEMAC)',
            'ec': 'Ecuador',
            'eg': 'Egypt',
            'sv': 'El Salvador',
            'emdeeco': 'World, Emerging & Developing',
            'emkts': 'World, Emerging',
            'gq': 'Equatorial Guinea',
            'er': 'Eritrea',
            'ee': 'Estonia',
            'et': 'Ethiopia',
            'eueu': 'EU',
            'eueu15': 'EU 15',
            'eu': 'Euro Area',
            'ea': 'Eurasia',
            'europe': 'Europe',
            'eurxez': 'Europe Excluding Euro Area',
            'eugb': 'Europe Excluding UK',
            'eumeafr': 'EMEA',
            'fk': 'Falkland Islands',
            'fo': 'Faroe Islands',
            'fj': 'Fiji',
            'fi': 'Finland',
            'wgr': 'West Germany',
            'fr': 'France',
            'gf': 'French Guyana',
            'pf': 'French Polynesia',
            'tf': 'French Southern Territories',
            'gseven': 'G7 Countries',
            'ga': 'Gabon',
            'gm': 'Gambia',
            'ge': 'Georgia',
            'de': 'Germany',
            'gh': 'Ghana',
            'gi': 'Gibraltar',
            'gcn': 'Greater China',
            'gr': 'Greece',
            'gl': 'Greenland',
            'gd': 'Grenada',
            'gp': 'Guadeloupe',
            'gu': 'Guam',
            'gt': 'Guatemala',
            'gg': 'Guernsey',
            'gn': 'Guinea',
            'gw': 'Guinea-Bissau',
            'gy': 'Guyana',
            'ht': 'Haiti',
            'hm': 'Heard Island & Mcdonald Islands',
            'va': 'Holy See (Vatican City State)',
            'hn': 'Honduras',
            'hk': 'Hong Kong',
            'hu': 'Hungary',
            'is': 'Iceland',
            'ieom': 'IEOM (French Overseas Emission Institute)',
            'in': 'India',
            'id': 'Indonesia',
            'ir': 'Iran',
            'iq': 'Iraq',
            'ie': 'Ireland',
            'im': 'Isle of Man',
            'il': 'Israel',
            'it': 'Italy',
            'ci': 'Ivory Coast',
            'jm': 'Jamaica',
            'jp': 'Japan',
            'je': 'Jersey',
            'jo': 'Jordan',
            'kz': 'Kazakhstan',
            'ke': 'Kenya',
            'ki': 'Kiribati',
            'apk': 'Kosovo',
            'kw': 'Kuwait',
            'kg': 'Kyrgyzstan',
            'la': 'Laos',
            'latam': 'Latin America',
            'lac': 'Latin America & the Caribbean',
            'lv': 'Latvia',
            'lb': 'Lebanon',
            'ls': 'Lesotho',
            'lr': 'Liberia',
            'ly': 'Libya',
            'li': 'Liechtenstein',
            'lt': 'Lithuania',
            'lu': 'Luxembourg',
            'mo': 'Macao',
            'mk': 'North Macedonia',
            'mg': 'Madagascar',
            'mfivasia': 'Major Five Asia',
            'mw': 'Malawi',
            'my': 'Malaysia',
            'mv': 'Maldives',
            'ml': 'Mali',
            'mt': 'Malta',
            'mh': 'Marshall Islands',
            'mq': 'Martinique',
            'mr': 'Mauritania',
            'mu': 'Mauritius',
            'yt': 'Mayotte',
            'mx': 'Mexico',
            'fm': 'Micronesia (Fed. States of)',
            'mideast': 'Middle East',
            'meafri': 'Middle East & Africa',
            'menaf': 'Middle East & North Africa',
            'md': 'Moldova',
            'mc': 'Monaco',
            'mn': 'Mongolia',
            'MS': 'Montenegro',
            'ms': 'Montserrat',
            'ma': 'Morocco',
            'mz': 'Mozambique',
            'nafta': 'NAFTA (North American Free Trade Agreement)',
            'na': 'Namibia',
            'nr': 'Nauru',
            'np': 'Nepal',
            'nl': 'Netherlands',
            'an': 'Netherlands Antilles',
            'nc': 'New Caledonia',
            'nz': 'New Zealand',
            'niae': 'Newly Industrialized Asian Economies',
            'ni': 'Nicaragua',
            'ne': 'Niger',
            'ng': 'Nigeria',
            'nu': 'Niue',
            'nonoecd': 'Non-OECD',
            'nordic': 'Nordic Countries',
            'nf': 'Norfolk Island',
            'americas': 'North & South America',
            'noram': 'North America',
            'kp': 'North Korea',
            'nsea': 'North Sea',
            'mp': 'Northern Mariana Islands',
            'no': 'Norway',
            'oceania': 'Oceania',
            'oecd': 'OECD Countries',
            'oecdthirt': 'OECD Euro Area 13',
            'eu14': 'OECD Euro Area 14',
            'oecdeu': 'OECD Europe',
            'oecdexhi': 'OECD Excluding High Inflation Countries',
            'oecdpacif': 'OECD Pacific',
            'om': 'Oman',
            'opec': 'OPEC Members',
            'oecs': 'Organization of Eastern Caribbean States (OECS)',
            'oae': 'World, Advanced, Excluding G7 & Euro Area, Other',
            'oilother': 'Oil Producers, Other',
            'pk': 'Pakistan',
            'pw': 'Palau',
            'ps': 'Palestine (West Bank & Gaza)',
            'pa': 'Panama',
            'pg': 'Papua New Guinea',
            'py': 'Paraguay',
            'pe': 'Peru',
            'ph': 'Philippines',
            'pn': 'Pitcairn',
            'pl': 'Poland',
            'pt': 'Portugal',
            'pr': 'Puerto Rico',
            'qa': 'Qatar',
            'worldrest': 'Rest of the World',
            're': 'Réunion',
            'ro': 'Romania',
            'ru': 'Russia',
            'rw': 'Rwanda',
            'bl': 'Saint Barthélemy',
            'sh': 'Saint Helena',
            'kn': 'Saint Kitts & Nevis',
            'lc': 'Saint Lucia',
            'mf': 'Saint Martin',
            'pm': 'Saint Pierre & Miquelon',
            'vc': 'Saint Vincent & The Grenadines',
            'ws': 'Samoa',
            'sm': 'San Marino',
            'st': 'Sao Tome & Principe',
            'sa': 'Saudi Arabia',
            'sn': 'Senegal',
            'rs': 'Serbia',
            'csg': 'Serbia & Montenegro',
            'sc': 'Seychelles',
            'sl': 'Sierra Leone',
            'sg': 'Singapore',
            'sk': 'Slovakia', 'si': 'Slovenia', 'sb': 'Solomon Islands',
            'so': 'Somalia', 'za': 'South Africa', 'souam': 'South America',
            'gs': 'South Georgia & The South Sandwich Islands', 'kr': 'South Korea', 'ss': 'South Sudan',
            'es': 'Spain', 'lk': 'Sri Lanka', 'subsahaf': 'Sub-Saharan Africa', 'sd': 'Sudan',
            'sr': 'Suriname', 'sj': 'Svalbard & Jan Mayen', 'sz': 'Eswatini', 'se': 'Sweden',
            'ch': 'Switzerland', 'sy': 'Syria', 'tw': 'Taiwan', 'tj': 'Tajikistan', 'tz': 'Tanzania',
            'th': 'Thailand', 'tl': 'Timor-Leste', 'tg': 'Togo', 'tk': 'Tokelau', 'to': 'Tonga',
            'tt': 'Trinidad & Tobago', 'tn': 'Tunisia', 'tr': 'Turkey', 'tm': 'Turkmenistan',
            'tc': 'Turks & Caicos Islands', 'tv': 'Tuvalu', 'ug': 'Uganda', 'ua': 'Ukraine',
            'ae': 'United Arab Emirates', 'gb': 'United Kingdom', 'us': 'United States',
            'um': 'United States Minor Outlying Islands', 'uy': 'Uruguay', 'uz': 'Uzbekistan',
            'wf': 'Wallis & Futuna', 'vu': 'Vanuatu', 've': 'Venezuela', 'wehemi': 'Western Hemisphere',
            'eh': 'Western Sahara', 'vn': 'Vietnam', 'vg': 'Virgin Islands, British',
            'vi': 'Virgin Islands, U.S.', 'world': 'World', 'wicn': 'World Excluding China',
            'wius': 'World Excluding USA', 'ye': 'Yemen', 'year': 'Yemen, Arab Republic',
            'yepd': "Yemen, People's Democratic Republic", 'yu': 'Yugoslavia', 'zm': 'Zambia',
            'zw': 'Zimbabwe', 'ax': 'Åland Islands', 'g1': 'G10 Countries '
        }

        return region_map


if __name__ == "__main__":
    # initiate macrobond
    mb = Macrobond()

    # fetch one series
    cpi = mb.FetchOneSeries('uscpi')
    print(cpi)

    # fetch many series
    tickers = ['uscpi', 'usgdp', 'gb3mgov']
    series = [MbSeries(t) for t in tickers]
    df = mb.CreateUnifiedSeriesRequest(series)
    print(df)

    # fetch unified data set
    pi = MbSeries(
        'uscpi',
        mv.AUTO,
        pp.NONE,
        hf.AUTO,
        lf.AVERAGE
    )

    y = MbSeries(
        'usgdp',
        mv.AUTO,
        pp.NONE,
        hf.DISTRIBUTE,
        lf.AUTO
    )

    i = MbSeries(
        'gb3mgov',
        mv.AUTO,
        pp.NONE,
        hf.AUTO,
        lf.AVERAGE
    )

    series2 = [pi, y, i]

    df2 = mb.CreateUnifiedSeriesRequest(series2, Frequency=sf.QUARTERLY)
    print(df2)