import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import os

# Minutely imbalance data
ODS133_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods133/records"
# Quarter-hourly imbalance data
ODS134_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"
# Photovoltaic power production estimation and forecast on Belgian grid (Historical)
ODS032_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/records"
# Wind power production estimation and forecast on Belgian grid (Historical)
ODS031_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031/records"
# Measured and forecasted total load on the Belgian grid (Historical data)
ODS001_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/records"
# 

CACHE_DIR = 'data_cache'
CACHE_DAYS = 30  # Remove cache files older than this
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_cache():
    now = datetime.now()
    for fname in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.isfile(fpath):
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if (now - mtime).days > CACHE_DAYS:
                os.remove(fpath)

def fetch_with_cache(name, start_datetime, end_datetime, url, chunk_minutes=None):
    """
    Generalized fetch function with caching for all data types.
    - name: short string for file prefix (e.g. 'imbalance_1min', 'pv', 'wind', 'load')
    - start_datetime, end_datetime: ISO strings
    - url: API endpoint
    - chunk_minutes: for chunked APIs (None for no chunking)
    """
    fname = f"{CACHE_DIR}/{name}_{start_datetime[:10]}_{end_datetime[:10]}.csv"
    if os.path.exists(fname):
        return pd.read_csv(fname, parse_dates=['datetime'])
    all_records = []
    if chunk_minutes:
        start_dt = pd.to_datetime(start_datetime)
        end_dt = pd.to_datetime(end_datetime)
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + pd.Timedelta(minutes=chunk_minutes), end_dt)
            offset = 0
            limit = 100
            max_offset = 9900
            while offset <= max_offset:
                api_url = (
                    f"{url}?where=datetime >= \"{current_start.strftime('%Y-%m-%dT%H:%M:%S')}\" AND datetime < \"{current_end.strftime('%Y-%m-%dT%H:%M:%S')}\""
                    f"&order_by=datetime"
                    f"&limit={limit}"
                    f"&offset={offset}"
                    "&timezone=Europe/Brussels"
                    "&use_labels=true"
                )
                response = requests.get(api_url)
                response.raise_for_status()
                data = response.json()
                records = data.get("results", [])
                if not records:
                    break
                all_records.extend(records)
                if len(records) < limit:
                    break
                offset += limit
            current_start = current_end
    else:
        offset = 0
        limit = 100
        max_offset = 9900
        while offset <= max_offset:
            api_url = (
                f"{url}?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
                f"&order_by=datetime"
                f"&limit={limit}"
                f"&offset={offset}"
                "&timezone=Europe/Brussels"
                "&use_labels=true"
            )
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            records = data.get("results", [])
            if not records:
                break
            all_records.extend(records)
            if len(records) < limit:
                break
            offset += limit
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        df.to_csv(fname, index=False)
        return df
    else:
        return pd.DataFrame([])

def fetch_minutely_imbalance(start_datetime, end_datetime):
    return fetch_with_cache('imbalance_1min', start_datetime, end_datetime, ODS133_URL, chunk_minutes=7000)

def fetch_quarterhourly_imbalance(start_datetime, end_datetime):
    return fetch_with_cache('imbalance_15min', start_datetime, end_datetime, ODS134_URL, chunk_minutes=10000)

def fetch_photovoltaic_production(start_datetime, end_datetime):
    return fetch_with_cache('pv', start_datetime, end_datetime, ODS032_URL)

def fetch_wind_production(start_datetime, end_datetime):
    return fetch_with_cache('wind', start_datetime, end_datetime, ODS031_URL)

def fetch_total_load(start_datetime, end_datetime):
    return fetch_with_cache('load', start_datetime, end_datetime, ODS001_URL)