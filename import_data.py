import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

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

def fetch_minutely_imbalance(start_datetime, end_datetime):
    chunk_minutes = 7000
    all_records = []
    start_dt = pd.to_datetime(start_datetime)
    end_dt = pd.to_datetime(end_datetime)
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + pd.Timedelta(minutes=chunk_minutes), end_dt)
        offset = 0
        limit = 100
        max_offset = 9900
        while offset <= max_offset:
            url = (
                f"{ODS133_URL}?where=datetime >= \"{current_start.strftime('%Y-%m-%dT%H:%M:%S')}\" AND datetime < \"{current_end.strftime('%Y-%m-%dT%H:%M:%S')}\""
                f"&order_by=datetime"
                f"&limit={limit}"
                f"&offset={offset}"
                "&timezone=Europe/Brussels"
                "&use_labels=true"
            )
            response = requests.get(url)
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
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        return df
    else:
        return pd.DataFrame([])

def fetch_quarterhourly_imbalance(start_datetime, end_datetime):
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900
    while offset <= max_offset:
        url = (
            f"{ODS134_URL}?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
            f"&order_by=datetime"
            f"&limit={limit}"
            f"&offset={offset}"
            "&timezone=Europe/Brussels"
            "&use_labels=true"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        records = data.get("results", [])
        if not records:
            break
        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit
        all_records.sort(key=lambda x: x['datetime'])
    return pd.DataFrame(all_records)

def fetch_photovoltaic_production(start_datetime, end_datetime):
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900
    while offset <= max_offset:
        url = (
            f"{ODS032_URL}?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
            f"&order_by=datetime"
            f"&limit={limit}"
            f"&offset={offset}"
            "&timezone=Europe/Brussels"
            "&use_labels=true"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        records = data.get("results", [])
        if not records:
            break
        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit
        all_records.sort(key=lambda x: x['datetime'])
    return pd.DataFrame(all_records)

def fetch_wind_production(start_datetime, end_datetime):
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900
    while offset <= max_offset:
        url = (
            f"{ODS031_URL}?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
            f"&order_by=datetime"
            f"&limit={limit}"
            f"&offset={offset}"
            "&timezone=Europe/Brussels"
            "&use_labels=true"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        records = data.get("results", [])
        if not records:
            break
        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit
        all_records.sort(key=lambda x: x['datetime'])
    return pd.DataFrame(all_records)

def fetch_total_load(start_datetime, end_datetime):
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900
    while offset <= max_offset:
        url = (
            f"{ODS001_URL}?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
            f"&order_by=datetime"
            f"&limit={limit}"
            f"&offset={offset}"
            "&timezone=Europe/Brussels"
            "&use_labels=true"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        records = data.get("results", [])
        if not records:
            break
        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit
        all_records.sort(key=lambda x: x['datetime'])
    return pd.DataFrame(all_records)