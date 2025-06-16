"""
Configuration file for the economic forecasting and yield modeling system.
This centralizes all constants, mappings, and settings used across different modules.
"""

import os
import datetime as dt
import logging

# =============================================================================
# COUNTRY DEFINITIONS
# =============================================================================

# List of countries with their full names
country_list = [
    'United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Poland',
    'Hungary', 'Czechia', 'South Africa', 'Canada', 'Australia', 'South Korea'
]

# Mapping from country names to country codes
country_list_mapping = {
    'United States': 'us',
    'United Kingdom': 'gb',
    'France': 'fr',
    'Germany': 'de',
    'Italy': 'it',
    'Poland': 'pl',
    'Hungary': 'hu',
    'Czechia': 'cz',
    'South Africa': 'za',
    'Canada': 'ca',
    'Australia': 'au',
    'South Korea': 'kr'
}

# Regional groupings
eurozone_countries = ['France', 'Germany', 'Italy']
emerging_markets = ['Poland', 'Hungary', 'Czechia', 'South Africa', "South Korea"]
developed_markets = ['United States', 'United Kingdom', 'Canada', 'Australia']

# =============================================================================
# DATA SOURCE TICKERS AND MAPPINGS
# =============================================================================

# GDP data tickers (Macrobond)
gdp_tickers = {
    "usgdp": "United States",
    "gbgdp": "United Kingdom",
    "frgdp": "France",
    "degdp": "Germany",
    "oecd_qna_00011061": "Italy",
    "plnaac0197": "Poland",
    "hugdp": "Hungary",
    "czgdp": "Czechia",
    "zagdp": "South Africa",
    "cagdp": "Canada",
    "augdp": "Australia",
    "krgdp": "South Korea"
}

# CPI inflation data tickers (Macrobond)
cpi_inf_tickers = {
    "uscpi": "United States",
    "gbcpi": "United Kingdom",
    "frcpi": "France",
    "decpi": "Germany",
    "itcpi": "Italy",
    "plcpi": "Poland",
    "hucpi": "Hungary",
    "czcpi": "Czechia",
    "zapric0000": "South Africa",
    "cacpi": "Canada",
    "aucpi": "Australia",
    "krcpi": "South Korea"
}

# Long-term GDP forecast tickers (Macrobond)
growth_forecast_lt_tickers = {
    "ih:mb:com:gdp_lt_us:gdp_lt_us": "United States",
    "ih:mb:com:gdp_lt_gb:gdp_lt_gb": "United Kingdom",
    "ih:mb:com:gdp_lt_fr:gdp_lt_fr": "France",
    "ih:mb:com:gdp_lt_de:gdp_lt_de": "Germany",
    "ih:mb:com:gdp_lt_it:gdp_lt_it": "Italy",
    "ih:mb:com:gdp_lt_pl:gdp_lt_pl": "Poland",
    "ih:mb:com:gdp_lt_hu:gdp_lt_hu": "Hungary",
    "ih:mb:com:gdp_lt_cz:gdp_lt_cz": "Czechia",
    "ih:mb:com:gdp_lt_za:gdp_lt_za": "South Africa",
    "ih:mb:com:gdp_lt_ca:gdp_lt_ca": "Canada",
    "ih:mb:com:gdp_lt_au:gdp_lt_au": "Australia",
    "ih:mb:com:gdp_lt_kr:gdp_lt_kr": "South Korea"
}

# CPI target tickers (Macrobond)
cpi_target_tickers = {
    "usrate1950": "United States",
    "gbrate0237": "United Kingdom",
    "eurate0022": "Eurozone",  # Used for France, Germany, Italy
    "carate0093": "Canada",
    "aurate0097": "Australia",
    "plrate0043": "Poland",
    "zapric0688": "South Africa",
    "hurate0005": "Hungary",
    "krrate0161": "South Korea",
    "czrate0064": "Czechia"
}

# Bond yield tickers for different tenors (Bloomberg)
bond_yield_tickers = {
    "2yr": {
        "USGG2YR Index": "United States",
        "GTGBP2YR Corp": "United Kingdom",
        "GTFRF2YR Corp": "France",
        "GTDEM2YR Corp": "Germany",
        "GTITL2YR Corp": "Italy",
        "GTPLN2YR Corp": "Poland",
        "GTHUF3YR Corp": "Hungary",  # Note: 3yr for Hungary
        "GTCZK2YR Corp": "Czechia",
        "GTZAR2YR Corp": "South Africa",
        "GTCAD2YR Corp": "Canada",
        "GTAUD2YR Corp": "Australia",
        "GTKRW2YR Corp": "South Korea"
    },
    "5yr": {
        "USGG5YR Index": "United States",
        "GTGBP5YR Corp": "United Kingdom",
        "GTFRF5YR Corp": "France",
        "GTDEM5YR Corp": "Germany",
        "GTITL5YR Corp": "Italy",
        "GTPLN5YR Corp": "Poland",
        "GTHUF5YR Corp": "Hungary",
        "GTCZK5YR Corp": "Czechia",
        "GTZAR5YR Corp": "South Africa",
        "GTCAD5YR Corp": "Canada",
        "GTAUD5YR Corp": "Australia",
        "GTKRW5YR Corp": "South Korea"
    },
    "10yr": {
        "USGG10YR Index": "United States",
        "GTGBP10YR Corp": "United Kingdom",
        "GTFRF10YR Corp": "France",
        "GTDEM10YR Corp": "Germany",
        "GTITL10YR Corp": "Italy",
        "GTPLN10YR Corp": "Poland",
        "GTHUF10YR Corp": "Hungary",
        "GTCZK10YR Corp": "Czechia",
        "GTZAR10YR Corp": "South Africa",
        "GTCAD10YR Corp": "Canada",
        "GTAUD10YR Corp": "Australia",
        "GTKRW10YR Corp": "South Korea"
    },
    "30yr": {
        "USGG30YR Index": "United States",
        "GTGBP30YR Corp": "United Kingdom",
        "GTFRF30YR Corp": "France",
        "GTDEM30YR Corp": "Germany",
        "GTITL30YR Corp": "Italy",
        "GTCAD30YR Corp": "Canada",
        "GTAUD30YR Corp": "Australia",
        "GTKRW30YR Corp": "South Korea"
        # Note: Some countries don't have 30yr bonds
    }
}

# Policy rate tickers (Bloomberg)
pol_rat_tickers = {
    "BISPDHUS Index": "United States",
    "BISPDHUK Index": "United Kingdom",
    "BISPDHEA Index": "France",  # Eurozone
    "EURR002W Index": "Germany",  # Eurozone
    "EUORDEPO Index": "Italy",    # Eurozone
    "BISPDHPO Index": "Poland",
    "BISPDHHU Index": "Hungary",
    "BISPDHCZ Index": "Czechia",
    "BISPDHSA Index": "South Africa",
    "BISPDHCA Index": "Canada",
    "BISPDHAU Index": "Australia",
    "BISPDHSK Index": "South Korea"
}

# Economic activity tracker tickers (Bloomberg)
act_track_tickers = {
    "GSUSCAI Index": "United States",
    "GSGBCAI Index": "United Kingdom",
    "GSFRCAI Index": "France",
    "GSDECAI Index": "Germany",
    "GSITCAI Index": "Italy",
    "GSPLCAI Index": "Poland",
    "GSHUCAI Index": "Hungary",
    "GSCZCAI Index": "Czechia",
    "GSZACAI Index": "South Africa",
    "GSCACAI Index": "Canada",
    "GSAUCAI Index": "Australia",
    "GSKRCAI Index": "South Korea"
}

# Credit rating tickers (Macrobond)
moodys_rating_tickers = {
    "ih:mb:com:m_rating_usa": "United States",
    "ih:mb:com:m_rating_gbr": "United Kingdom",
    "ih:mb:com:m_rating_fra": "France",
    "ih:mb:com:m_rating_deu": "Germany",
    "ih:mb:com:m_rating_ita": "Italy",
    "ih:mb:com:m_rating_pol": "Poland",
    "ih:mb:com:m_rating_hun": "Hungary",
    "ih:mb:com:m_rating_cze": "Czechia",
    "ih:mb:com:m_rating_zaf": "South Africa",
    "ih:mb:com:m_rating_can": "Canada",
    "ih:mb:com:m_rating_aus": "Australia",
    "ih:mb:com:m_rating_kor": "South Korea"
}

fitch_rating_tickers = {
    "ih:mb:com:f_rating_usa": "United States",
    "ih:mb:com:f_rating_gbr": "United Kingdom",
    "ih:mb:com:f_rating_fra": "France",
    "ih:mb:com:f_rating_deu": "Germany",
    "ih:mb:com:f_rating_ita": "Italy",
    "ih:mb:com:f_rating_pol": "Poland",
    "ih:mb:com:f_rating_hun": "Hungary",
    "ih:mb:com:f_rating_cze": "Czechia",
    "ih:mb:com:f_rating_zaf": "South Africa",
    "ih:mb:com:f_rating_can": "Canada",
    "ih:mb:com:f_rating_aus": "Australia",
    "ih:mb:com:f_rating_kor": "South Korea"
}

sp_rating_tickers = {
    "ih:mb:com:s_rating_usa": "United States",
    "ih:mb:com:s_rating_gbr": "United Kingdom",
    "ih:mb:com:s_rating_fra": "France",
    "ih:mb:com:s_rating_deu": "Germany",
    "ih:mb:com:s_rating_ita": "Italy",
    "ih:mb:com:s_rating_pol": "Poland",
    "ih:mb:com:s_rating_hun": "Hungary",
    "ih:mb:com:s_rating_cze": "Czechia",
    "ih:mb:com:s_rating_zaf": "South Africa",
    "ih:mb:com:s_rating_can": "Canada",
    "ih:mb:com:s_rating_aus": "Australia",
    "ih:mb:com:s_rating_kor": "South Korea"
}

iip_gdp_tickers = {
    "ih:mb:com:iip_ppt_gdp_us:iip_ppt_gdp_us"	: "United States",
    "ih:mb:com:iip_ppt_gdp_gb:iip_ppt_gdp_gb"	: "United Kingdom",
    "ih:mb:com:iip_ppt_gdp_fr:iip_ppt_gdp_fr"	: "France",
    "ih:mb:com:iip_ppt_gdp_de:iip_ppt_gdp_de"	: "Germany",
    "ih:mb:com:iip_ppt_gdp_it:iip_ppt_gdp_it"	: "Italy",
    "ih:mb:com:iip_ppt_gdp_pl:iip_ppt_gdp_pl"	: "Poland",
    "ih:mb:com:iip_ppt_gdp_hu:iip_ppt_gdp_hu"	: "Hungary",
    "ih:mb:com:iip_ppt_gdp_cz:iip_ppt_gdp_cz"	: "Czechia",
    "ih:mb:com:iip_ppt_gdp_za:iip_ppt_gdp_za"	: "South Africa",
    "ih:mb:com:iip_ppt_gdp_ca:iip_ppt_gdp_ca"	: "Canada",
    "ih:mb:com:iip_ppt_gdp_au:iip_ppt_gdp_au"	: "Australia",
    "ih:mb:com:iip_ppt_gdp_kr:iip_ppt_gdp_kr"	: "South Korea"   
}

unemployment_tickers = {
    
    "uslama1849" :"United States",
	"gblama00081" : "United Kingdom",
    "frlama0193" : "France",
    "delama0822" : "Germany",
    "itlama0497" : "Italy",
    "pllama0140" : "Poland",
    "hulama0288" : "Hungary",
    "czlama0051" : "Czechia",
    "zalama0003" : "South Africa",
    "calama1124" : "Canada",
    "aulama0227" : "Australia",
    "krlama0422" : "South Korea"
}

# Column name mappings for data sources
COLUMN_MAPPINGS = {
    "gdp": {
        "usgdp": "gdp_us",
        "gbgdp": "gdp_gb",
        "frgdp": "gdp_fr",
        "degdp": "gdp_de",
        "oecd_qna_00011061": "gdp_it",
        "plnaac0197": "gdp_pl",
        "hugdp": "gdp_hu",
        "czgdp": "gdp_cz",
        "zagdp": "gdp_za",
        "cagdp": "gdp_ca",
        "augdp": "gdp_au",
        "krgdp": "gdp_kr"
    },
    "cpi_inf": {
        "uscpi": "cpi_inf_us",
        "gbcpi": "cpi_inf_gb",
        "frcpi": "cpi_inf_fr",
        "decpi": "cpi_inf_de",
        "itcpi": "cpi_inf_it",
        "plcpi": "cpi_inf_pl",
        "hucpi": "cpi_inf_hu",
        "czcpi": "cpi_inf_cz",
        "zapric0000": "cpi_inf_za",
        "cacpi": "cpi_inf_ca",
        "aucpi": "cpi_inf_au",
        "krcpi": "cpi_inf_kr"
    },
    "growth_forecast_lt": {
        "ih:mb:com:gdp_lt_us:gdp_lt_us": "gdp_lt_us",
        "ih:mb:com:gdp_lt_gb:gdp_lt_gb": "gdp_lt_gb",
        "ih:mb:com:gdp_lt_fr:gdp_lt_fr": "gdp_lt_fr",
        "ih:mb:com:gdp_lt_de:gdp_lt_de": "gdp_lt_de",
        "ih:mb:com:gdp_lt_it:gdp_lt_it": "gdp_lt_it",
        "ih:mb:com:gdp_lt_pl:gdp_lt_pl": "gdp_lt_pl",
        "ih:mb:com:gdp_lt_hu:gdp_lt_hu": "gdp_lt_hu",
        "ih:mb:com:gdp_lt_cz:gdp_lt_cz": "gdp_lt_cz",
        "ih:mb:com:gdp_lt_za:gdp_lt_za": "gdp_lt_za",
        "ih:mb:com:gdp_lt_ca:gdp_lt_ca": "gdp_lt_ca",
        "ih:mb:com:gdp_lt_au:gdp_lt_au": "gdp_lt_au",
        "ih:mb:com:gdp_lt_kr:gdp_lt_kr": "gdp_lt_kr"
    },
    "cpi_target": {
        "usrate1950": "cpi_target_us",
        "gbrate0237": "cpi_target_gb",
        "eurate0022_3": "cpi_target_fr",
        "eurate0022_4": "cpi_target_de",
        "eurate0022_5": "cpi_target_it",
        "carate0093": "cpi_target_ca",
        "aurate0097": "cpi_target_au",
        "plrate0043": "cpi_target_pl",
        "zapric0688": "cpi_target_za",
        "hurate0005": "cpi_target_hu",
        "krrate0161": "cpi_target_kr",
        "czrate0064": "cpi_target_cz"
    },
    "bond_yield_2yr": {
        "USGG2YR Index": "yld_2yr_us",
        "GTGBP2YR Corp": "yld_2yr_gb",
        "GTFRF2YR Corp": "yld_2yr_fr",
        "GTDEM2YR Corp": "yld_2yr_de",
        "GTITL2YR Corp": "yld_2yr_it",
        "GTPLN2YR Corp": "yld_2yr_pl",
        "GTHUF3YR Corp": "yld_2yr_hu",  # Using 3yr for Hungary
        "GTCZK2YR Corp": "yld_2yr_cz",
        "GTZAR2YR Corp": "yld_2yr_za",
        "GTCAD2YR Corp": "yld_2yr_ca",
        "GTAUD2YR Corp": "yld_2yr_au",
        "GTKRW2YR Corp": "yld_2yr_kr"
    },
    "bond_yield_5yr": {
        "USGG5YR Index": "yld_5yr_us",
        "GTGBP5YR Corp": "yld_5yr_gb",
        "GTFRF5YR Corp": "yld_5yr_fr",
        "GTDEM5YR Corp": "yld_5yr_de",
        "GTITL5YR Corp": "yld_5yr_it",
        "GTPLN5YR Corp": "yld_5yr_pl",
        "GTHUF5YR Corp": "yld_5yr_hu",
        "GTCZK5YR Corp": "yld_5yr_cz",
        "GTZAR5YR Corp": "yld_5yr_za",
        "GTCAD5YR Corp": "yld_5yr_ca",
        "GTAUD5YR Corp": "yld_5yr_au",
        "GTKRW5YR Corp": "yld_5yr_kr"
    },
    "bond_yield_10yr": {
        "USGG10YR Index": "yld_10yr_us",
        "GTGBP10YR Corp": "yld_10yr_gb",
        "GTFRF10YR Corp": "yld_10yr_fr",
        "GTDEM10YR Corp": "yld_10yr_de",
        "GTITL10YR Corp": "yld_10yr_it",
        "GTPLN10YR Corp": "yld_10yr_pl",
        "GTHUF10YR Corp": "yld_10yr_hu",
        "GTCZK10YR Corp": "yld_10yr_cz",
        "GTZAR10YR Corp": "yld_10yr_za",
        "GTCAD10YR Corp": "yld_10yr_ca",
        "GTAUD10YR Corp": "yld_10yr_au",
        "GTKRW10YR Corp": "yld_10yr_kr"
    },
    "bond_yield_30yr": {
        "USGG30YR Index": "yld_30yr_us",
        "GTGBP30YR Corp": "yld_30yr_gb",
        "GTFRF30YR Corp": "yld_30yr_fr",
        "GTDEM30YR Corp": "yld_30yr_de",
        "GTITL30YR Corp": "yld_30yr_it",
        "GTCAD30YR Corp": "yld_30yr_ca",
        "GTAUD30YR Corp": "yld_30yr_au",
        "GTKRW30YR Corp": "yld_30yr_kr"
    },
    "policy_rates": {
        "BISPDHUS Index": "pol_rat_us",
        "BISPDHUK Index": "pol_rat_gb",
        "BISPDHEA Index": "pol_rat_fr",
        "EURR002W Index": "pol_rat_de",
        "EUORDEPO Index": "pol_rat_it",
        "BISPDHPO Index": "pol_rat_pl",
        "BISPDHHU Index": "pol_rat_hu",
        "BISPDHCZ Index": "pol_rat_cz",
        "BISPDHSA Index": "pol_rat_za",
        "BISPDHCA Index": "pol_rat_ca",
        "BISPDHAU Index": "pol_rat_au",
        "BISPDHSK Index": "pol_rat_kr"
    },
    "activity": {
        "GSUSCAI Index": "act_track_us",
        "GSGBCAI Index": "act_track_gb",
        "GSFRCAI Index": "act_track_fr",
        "GSDECAI Index": "act_track_de",
        "GSITCAI Index": "act_track_it",
        "GSPLCAI Index": "act_track_pl",
        "GSHUCAI Index": "act_track_hu",
        "GSCZCAI Index": "act_track_cz",
        "GSZACAI Index": "act_track_za",
        "GSCACAI Index": "act_track_ca",
        "GSAUCAI Index": "act_track_au",
        "GSKRCAI Index": "act_track_kr"
    },
    "moody_ratings": {
        "ih:mb:com:m_rating_usa": "m_rating_us",
        "ih:mb:com:m_rating_gbr": "m_rating_gb",
        "ih:mb:com:m_rating_fra": "m_rating_fr",
        "ih:mb:com:m_rating_deu": "m_rating_de",
        "ih:mb:com:m_rating_ita": "m_rating_it",
        "ih:mb:com:m_rating_pol": "m_rating_pl",
        "ih:mb:com:m_rating_hun": "m_rating_hu",
        "ih:mb:com:m_rating_cze": "m_rating_cz",
        "ih:mb:com:m_rating_zaf": "m_rating_za",
        "ih:mb:com:m_rating_can": "m_rating_ca",
        "ih:mb:com:m_rating_aus": "m_rating_au",
        "ih:mb:com:m_rating_kor": "m_rating_kr"
    },
    "fitch_ratings": {
        "ih:mb:com:f_rating_usa": "f_rating_us",
        "ih:mb:com:f_rating_gbr": "f_rating_gb",
        "ih:mb:com:f_rating_fra": "f_rating_fr",
        "ih:mb:com:f_rating_deu": "f_rating_de",
        "ih:mb:com:f_rating_ita": "f_rating_it",
        "ih:mb:com:f_rating_pol": "f_rating_pl",
        "ih:mb:com:f_rating_hun": "f_rating_hu",
        "ih:mb:com:f_rating_cze": "f_rating_cz",
        "ih:mb:com:f_rating_zaf": "f_rating_za",
        "ih:mb:com:f_rating_can": "f_rating_ca",
        "ih:mb:com:f_rating_aus": "f_rating_au",
        "ih:mb:com:f_rating_kor": "f_rating_kr"
    },
    "sp_ratings": {
        "ih:mb:com:s_rating_usa": "s_rating_us",
        "ih:mb:com:s_rating_gbr": "s_rating_gb",
        "ih:mb:com:s_rating_fra": "s_rating_fr",
        "ih:mb:com:s_rating_deu": "s_rating_de",
        "ih:mb:com:s_rating_ita": "s_rating_it",
        "ih:mb:com:s_rating_pol": "s_rating_pl",
        "ih:mb:com:s_rating_hun": "s_rating_hu",
        "ih:mb:com:s_rating_cze": "s_rating_cz",
        "ih:mb:com:s_rating_zaf": "s_rating_za",
        "ih:mb:com:s_rating_can": "s_rating_ca",
        "ih:mb:com:s_rating_aus": "s_rating_au",
        "ih:mb:com:s_rating_kor": "s_rating_kr"
    },
    "consolidated_ratings": {
        "us": "rating_us",
        "gb": "rating_gb",
        "fr": "rating_fr",
        "de": "rating_de",
        "it": "rating_it",
        "pl": "rating_pl",
        "hu": "rating_hu",
        "cz": "rating_cz",
        "za": "rating_za",
        "ca": "rating_ca",
        "au": "rating_au",
        "kr": "rating_kr"
    },
    "iip_gdp" : {
    "ih:mb:com:iip_ppt_gdp_us:iip_ppt_gdp_us"	: "iip_gdp_us",
    "ih:mb:com:iip_ppt_gdp_gb:iip_ppt_gdp_gb"	: "iip_gdp_gb",
    "ih:mb:com:iip_ppt_gdp_fr:iip_ppt_gdp_fr"	: "iip_gdp_fr",
    "ih:mb:com:iip_ppt_gdp_de:iip_ppt_gdp_de"	: "iip_gdp_de",
    "ih:mb:com:iip_ppt_gdp_it:iip_ppt_gdp_it"	: "iip_gdp_it",
    "ih:mb:com:iip_ppt_gdp_pl:iip_ppt_gdp_pl"	: "iip_gdp_pl",
    "ih:mb:com:iip_ppt_gdp_hu:iip_ppt_gdp_hu"	: "iip_gdp_hu",
    "ih:mb:com:iip_ppt_gdp_cz:iip_ppt_gdp_cz"	: "iip_gdp_cz",
    "ih:mb:com:iip_ppt_gdp_za:iip_ppt_gdp_za"	: "iip_gdp_za",
    "ih:mb:com:iip_ppt_gdp_ca:iip_ppt_gdp_ca"	: "iip_gdp_ca",
    "ih:mb:com:iip_ppt_gdp_au:iip_ppt_gdp_au"	: "iip_gdp_au",
    "ih:mb:com:iip_ppt_gdp_kr:iip_ppt_gdp_kr"	: "iip_gdp_kr",   
    },
    "unemployment_rate" : {
    "uslama1849" :"u_rat_us",
	"gblama00081" : "u_rat_gb",
    "frlama0193" : "u_rat_fr",
    "delama0822" : "u_rat_de",
    "itlama0497" : "u_rat_it",
    "pllama0140" : "u_rat_pl",
    "hulama0288" : "u_rat_hu",
    "czlama0051" : "u_rat_cz",
    "zalama0003" : "u_rat_za",
    "calama1124" : "u_rat_ca",
    "aulama0227" : "u_rat_au",
    "krlama0422" : "u_rat_kr"
}
}

# =============================================================================
# FORECAST PARAMETERS
# =============================================================================

# Date ranges
HISTORICAL_START_DATE = '1947-01-01'  # Start date for historical data
DEFAULT_FORECAST_END_DATE = '2060-12-31'  # End date for forecasts
DEFAULT_HISTORICAL_FORECAST_START = '1990-01-01'  # Start date for historical forecasts

# Forecast horizons (in months)
FORECAST_HORIZON = [24, 60, 120, 360]  # 2, 5, 10, and 30 years

# Forecast transition parameters
DEFAULT_FORECAST_HORIZON = 60  # Near-term forecast horizon (5 years)
FULL_FORECAST_HORIZON = 432  # Full forecast horizon (36 years)
GROWTH_DECAY_PERIOD = 60  # Transition period for growth (5 years)
INFLATION_DECAY_PERIOD = 36  # Transition period for inflation (3 years)
DEFAULT_WIND_BACK_YEARS = 5  # Years to wind back for transition
GENERATE_PLOTS = True
# =============================================================================
# YIELD MODEL CONFIGURATION
# =============================================================================

# Tenor-specific feature sets
TENOR_FEATURES = {
    'yld_2yr': ['policy_rates', 'inflation', 'activity', 'historical_forecasts_2yr'],
    'yld_5yr': ['policy_rates', 'inflation', 'activity', 'historical_forecasts_5yr'],
    'yld_10yr': ['policy_rates', 'inflation', 'activity', 'risk_rating', 'historical_forecasts_10yr'],
    'yld_30yr': ['policy_rates', 'inflation', 'activity', 'risk_rating', 'historical_forecasts_30yr']
}

# Neural network parameters
MLP_CONFIGS = [
    {'hidden_layer_sizes': (10,), 'max_iter': 1000, 'random_state': 42},
    {'hidden_layer_sizes': (20, 10), 'max_iter': 1000, 'random_state': 42},
    {'hidden_layer_sizes': (50, 25), 'max_iter': 1000, 'random_state': 42},
    {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42}
]

# Default training parameters
DEFAULT_TRAIN_TEST_SPLIT = 0.6  # 60% training, 40% testing
MIN_DATA_POINTS_FOR_MODEL = 30  # Minimum data points needed for modeling

# =============================================================================
# DATA RETRIEVAL PARAMETERS
# =============================================================================

# Default data fetch date range
DEFAULT_DATE_FROM = dt.date(1990, 1, 1)
DEFAULT_DATE_TO = dt.date.today()

# Bloomberg parameters
BLOOMBERG_FIELD = "PX_LAST"
BLOOMBERG_DAILY_PERIODICITY = "DAILY"
BLOOMBERG_MONTHLY_PERIODICITY = "MONTHLY"
BLOOMBERG_NON_TRADING_DAY_FILL = "ALL_CALENDAR_DAYS"
BLOOMBERG_NON_TRADING_DAY_METHOD = "PREVIOUS_VALUE"

# Default values for missing data
DEFAULT_GROWTH_RATE = 2.0
DEFAULT_INFLATION_RATE = 2.0

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = BASE_DIR

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, VISUALIZATION_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
DEFAULT_MODEL_PATH_TEMPLATE = os.path.join(MODEL_DIR, "{country}_{tenor}_mlp_model.pkl")
DEFAULT_FORECAST_CSV = os.path.join(OUTPUT_DIR, "unified_forecasts.csv")
DEFAULT_HISTORICAL_FORECAST_TEMPLATE = os.path.join(OUTPUT_DIR, "{country}_historical_forecasts.csv")
LOG_FILE = os.path.join(LOG_DIR, "forecast_model.log")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log levels
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create logging configuration function
def configure_logging(level=DEFAULT_LOG_LEVEL, log_file=LOG_FILE):
    """
    Configure logging for the application.
    
    Parameters:
        level: Log level (default: INFO)
        log_file: Path to log file
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()