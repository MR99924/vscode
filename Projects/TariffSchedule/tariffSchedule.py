import os
HOME_DIR = r"C:\Users\MR99924\workspace\vscode\Projects\TariffSchedule"
os.chdir(HOME_DIR)
print("Current working directory set to:", os.getcwd())

import json
import datetime as dt
import pandas as pd
import requests
import certifi
from dateutil.rrule import rrule, DAILY
from tqdm import tqdm

# ------------------ USER SETTINGS ------------------ #
API_KEY = "e727b6ccea654f9cc4bb0cf7a91cfe1a063ace8f"
YEAR = 2024
# Supply *numeric* Census CTY_CODE values (see note ↓). Leave blank to pull **all** but risk slow downloads.
COUNTRIES = [1010,1220,1610, 2010,2110,2150,2190,2230,2250,2320,2360,2390,2410,2430,2440,2450,2470,2481,2482,2483,
             2484,2485,2486,2487,2488,2489,2720,2740,2774,2777,2779,2831,2839,3010,3070,3120,3150,3170,3310,3330,3350,3370,
             3510,3530,3550,3570,3720,4000,4010,4031,4039,4050,4091,4099,4120,4190,4210,4231,4239,4271,4272,4279,4280,4330,
             4351,4359,4370,4411,4419,4470,4490,4510,4550,4621,4622,4623,4631,4632,4633,4634,4635,4641,4642,4643,4644,4700,
             4710,4720,4730,4751,4752,4759,4791,4792,4793,4794,4801,4803,4804,4810,4840,4850,4870,4890,4910,5020,5040,5050,
             5070,5081,5082,5083,5110,5130,5170,5180,5200,5210,5230,5250,5310,5330,5350,5360,5380,5420,5460,5490,5520,5530,
             5550,5570,5590,5600,5601,5610,5650,5660,5682,5683,5700,5740,5800,5820,5830,5880,6021,6022,6023,6024,6029,
             6040,6141,6142,6143,6144,6150,6223,6224,6225,6226,6227,6412,6413,6414,6810,6820,6830,6862,6863,6864,7140,7210,
             7230,7250,7290,7321,7323,7380,7410,7420,7440,7450,7460,7470,7480,7490,7500,7510,7520,7530,7540,7550,7560,7580,
             7600,7610,7620,7642,7643,7644,7650,7670,7690,7700,7741,7749,7770,7780,7790,7800,7810,7830,7850,7870,
             7880,7881,7890,7904,7905,7910,7920,7930,7940,7950,7960,7970,7990]
BATCH = 1 # how many country codes per API call
TIMEOUT = 300 # seconds to wait for each call
POLICY_CSV = "policy_actions_master.csv"
OUT_CSV = "daily_avg_tariff.csv"

# --------------------------------------------------- #

## 1 ── Pull constant import weights (HS10 × country) ##
def fetch_imports(year: int, countries: list[int] | None = None, batch: int = 1, timeout: int = 300) -> pd.DataFrame:
    """
    Pull 2024 import weights in manageable chunks.
    If 'countries' is None or empty, fetches ALL countries
    in batches of 'batch' CTY_CODEs per request.
    """
    base_url = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
    common = {
        "get": "CTY_CODE,I_COMMODITY,GEN_VAL_YR",
        "time": year,
        "COMM_LVL": "HS10",
        "key": API_KEY
    }

    # Helper to hit the API once
    def _one_call(codes: str):
        params = common | {"CTY_CODE": codes}
        print("Sending request to:", base_url)
        print("With params:", params)

        try:
            r = requests.get(base_url, params=params, timeout=timeout, verify=False)
            print("Status Code:", r.status_code)
            print("First 200 characters of response:", r.text[:200])
            r.raise_for_status()

            if r.status_code == 204 or not r.text.strip():
                print(f"⚠️ No data returned for CTY_CODE(s): {codes}")
                return None

            if not r.headers.get("Content-Type", "").startswith("application/json"):
                print(f"⚠️ Non-JSON response received for {codes}")
                return None

            return r.json()

        except Exception as e:
            print(f"❌ Error fetching CTY_CODE(s) {codes}: {e}")
            return None

    # Decide which CTY_CODE strings to feed the helper
    if not countries:
        print("fetching full list of CTY_CODEs from Census...")
        all_codes = [row[0] for row in _one_call("*")[1:]]  # first col is CTY_CODE
        target_groups = [all_codes[i:i+batch] for i in range(0, len(all_codes), batch)]
    else:
        target_groups = [countries[i:i+batch] for i in range(0, len(countries), batch)]

    records = []
    for group in target_groups:
        codestr = ",".join(map(str, group))  # e.g. "1010,5700,8260"
        data = _one_call(codestr)
        if not data:
            continue  # Skip this batch if nothing came back

        cols, *rows = data
        records.extend(rows)
        print(f"✔ Fetched {len(rows):>6} rows for CTY_CODE(s): {codestr}")


    df = pd.DataFrame(records, columns=cols)
    df['GEN_VAL_YR'] = pd.to_numeric(df['GEN_VAL_YR'], errors="coerce")
    df["I_COMMODITY"] = df["I_COMMODITY"].astype(str).str.zfill(10)

    annual_totals = df.groupby(["CTY_CODE", "I_COMMODITY"], as_index=False)["GEN_VAL_YR"].sum().rename(columns={"I_COMMODITY": "hs10", "GEN_VAL_YR": "value"})
        
    return annual_totals

weights = fetch_imports(YEAR, COUNTRIES)
weights.to_csv("import_data.csv")

## 2 ── Load tariff-action table ######################
policy = pd.read_csv(POLICY_CSV, parse_dates=['start', 'end'])
policy['country'] = policy['country'].fillna('*')

## 3 ── Expand tariff table to daily granularity ######
dates = pd.date_range(policy['start'].min(), policy['end'].max(), freq='D')
daily = (policy
         .apply(lambda r: pd.Series(index=pd.date_range(r.start, r.end, 'D'), data=r.rate), axis=1)
         .T.fillna(0).groupby(level=0).max())  # max if overlaps
daily.index = pd.to_datetime(daily.index)

## 4 ── Match duties to HS10 × country, weight, aggregate ##
def effective_rate(row, date):
    """Return the duty rate applicable to a single import row on 'date'."""
    key = date
    if key not in daily.index:  # before first tariff action
        return 0.0
    series = daily.loc[key]
    # fallback hierarchy: exact country-match > '*' wildcard
    for c in (row['CTY_CODE'], '*'):
        # match by HS prefix length
        for pref, rate in series.items():
            if (pref == '*' or row['HS10'].startswith(str(pref))) and \
                    (policy.loc[policy['hs_prefix'] == pref, 'country']
                     .str.contains(c).any()):
                return rate
    return 0.0

records = []
for d in tqdm(dates, desc="Computing daily series"):
    tmp = weights.copy()
    tmp['rate'] = tmp.apply(lambda r: effective_rate(r, d), axis=1)
    tmp['weighted'] = tmp['value'] * tmp['rate']
    avg = tmp['weighted'].sum() / tmp['value'].sum()
    records.append({'date': d, 'avg_rate': avg})

pd.DataFrame(records).to_csv(OUT_CSV, index=False)
print(f"Saved daily series to {OUT_CSV}")
