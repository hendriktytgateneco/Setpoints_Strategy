"""
main_1min_parallel_grid.py

Parallelized battery setpoint optimization using Elia imbalance price data (1-min and 15-min resolution).

- Fetches 1-min and 15-min imbalance price data for a given week from Elia's ODS API, handling API row limits by chunking requests.
- Simulates battery operation (with delay) for a grid of charge/discharge setpoints.
- Uses multiprocessing (ProcessPoolExecutor) to parallelize the grid search over setpoints for efficient CPU utilization.
- Applies a cycle constraint (max average cycles/day) to filter feasible solutions.
- Selects the best setpoint pair based on total revenue, and visualizes the results.
- Prints economic metrics: best setpoints, revenue, cycles, average spread, and revenue per MWh injected.

Functions:
- fetch_minutely_imbalance: Fetches 1-min data, chunked to avoid API row limits.
- fetch_quarterhourly_imbalance: Fetches 15-min data, chunked to avoid API row limits.
- simulate_vectorized_setpoints_worker: Simulates battery operation for a given setpoint pair (worker for parallel execution).
- optimize_battery_with_delay_parallel: Orchestrates data fetching, parallel grid search, result selection, and plotting.

Run as a script to optimize for the previous full week.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ODS133_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods133/records"
ODS134_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"

def fetch_minutely_imbalance(start_datetime, end_datetime):
    """
    Fetches 1-min imbalance price data from Elia ODS133 API for the specified period.
    Handles the API's 10,000 row limit by splitting the request into chunks (default 7,000 minutes per chunk).
    Returns a DataFrame with all 1-min data for the requested period.
    Args:
        start_datetime (str): Start datetime in ISO format (e.g., '2024-01-01T00:00:00').
        end_datetime (str): End datetime in ISO format (exclusive).
    Returns:
        pd.DataFrame: DataFrame with columns including 'datetime' and 'imbalanceprice'.
    """
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
    """
    Fetches 15-min imbalance price data from Elia ODS134 API for the specified period.
    Handles the API's 10,000 row limit by splitting the request into chunks.
    Returns a DataFrame with all 15-min data for the requested period.
    Args:
        start_datetime (str): Start datetime in ISO format (e.g., '2024-01-01T00:00:00').
        end_datetime (str): End datetime in ISO format (exclusive).
    Returns:
        pd.DataFrame: DataFrame with columns including 'datetime' and 'imbalanceprice'.
    """
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

def simulate_vectorized_setpoints_worker(args):
    """
    Simulates battery operation for a given set of setpoints (charge/discharge) and battery parameters.
    Used as a worker function for parallel execution.
    Args:
        args (tuple):
            imbalance_prices (np.ndarray): 1-min price array.
            battery_power_kw (float): Battery power in kW.
            battery_capacity_kwh (float): Battery capacity in kWh.
            soc_init (float): Initial state of charge (0-1).
            delay_min (int): Delay in minutes for price signal.
            charge_setpoint (float): Price below which to charge.
            discharge_setpoint (float): Price above which to discharge.
    Returns:
        tuple: (delivered_kwh, socs, actions, abs_energy_throughput)
            delivered_kwh (np.ndarray): Energy delivered per minute (kWh).
            socs (np.ndarray): State of charge per minute (kWh).
            actions (np.ndarray): Action taken per minute ('charge', 'discharge', 'idle').
            abs_energy_throughput (float): Total absolute energy throughput (kWh).
    """
    (imbalance_prices, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, charge_setpoint, discharge_setpoint) = args
    n_steps = len(imbalance_prices)
    socs = np.zeros(n_steps)
    delivered_kwh = np.zeros(n_steps)
    actions = np.empty(n_steps, dtype=object)
    soc = soc_init * battery_capacity_kwh
    delayed_indices = np.arange(n_steps) - delay_min
    delayed_indices[delayed_indices < 0] = 0
    delayed_prices = imbalance_prices[delayed_indices]
    duration = 1/60
    for idx in range(n_steps):
        price = delayed_prices[idx]
        if price < charge_setpoint:
            charge = min(battery_power_kw, battery_capacity_kwh - soc)
            actual = charge * duration
            soc = min(soc + actual, battery_capacity_kwh)
            actions[idx] = 'charge'
        elif price > discharge_setpoint:
            discharge = min(battery_power_kw, soc)
            actual = -discharge * duration
            soc = max(soc + actual, 0)
            actions[idx] = 'discharge'
        else:
            actual = 0
            actions[idx] = 'idle'
        delivered_kwh[idx] = actual
        socs[idx] = soc
    abs_energy_throughput = np.sum(np.abs(delivered_kwh))
    return delivered_kwh, socs, actions, abs_energy_throughput

def optimize_battery_with_delay_parallel(
    start, end, battery_power_kw=50, battery_capacity_kwh=100, soc_init=0.5, delay_min=3, max_cycles_per_day=1.4
):
    """
    Orchestrates the full optimization process:
    - Fetches 1-min and 15-min imbalance price data for the specified week.
    - Sets up a grid of charge/discharge setpoints.
    - Uses multiprocessing to parallelize the simulation of all setpoint pairs.
    - Applies a cycle constraint (max average cycles/day).
    - Selects the best setpoint pair based on total revenue.
    - Prints economic metrics and visualizes the results.
    Args:
        start (str): Start datetime in ISO format (e.g., '2024-01-01T00:00:00').
        end (str): End datetime in ISO format (exclusive).
        battery_power_kw (float): Battery power in kW.
        battery_capacity_kwh (float): Battery capacity in kWh.
        soc_init (float): Initial state of charge (0-1).
        delay_min (int): Delay in minutes for price signal.
        max_cycles_per_day (float): Maximum allowed average cycles per day.
    Returns:
        None. Prints results and shows plots.
    """
    df_1min = fetch_minutely_imbalance(start, end)
    df_15min = fetch_quarterhourly_imbalance(start, end)
    if df_1min.empty or df_15min.empty:
        print("No data fetched for the given period.")
        return
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
    df_15min['datetime'] = pd.to_datetime(df_15min['datetime'])
    df_1min['qh_start'] = df_1min['datetime'].dt.floor('15min')
    df_15min = df_15min.set_index('datetime')
    imbalance_prices = df_1min['imbalanceprice'].to_numpy()
    qh_starts = df_1min['qh_start'].to_numpy()
    datetimes = df_1min['datetime'].to_numpy()
    charge_price_range = range(int(df_15min['imbalanceprice'].min()), int(df_15min['imbalanceprice'].max()), 5)
    discharge_price_range = range(int(df_15min['imbalanceprice'].min()), int(df_15min['imbalanceprice'].max()), 5)
    max_total_cycles = max_cycles_per_day * 7
    best_revenue = float('-inf')
    best_result = None
    best_cycles_used = None
    tasks = []
    for charge_setpoint in charge_price_range:
        for discharge_setpoint in discharge_price_range:
            if discharge_setpoint <= charge_setpoint:
                continue
            tasks.append((imbalance_prices, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, charge_setpoint, discharge_setpoint))
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(simulate_vectorized_setpoints_worker, task): task for task in tasks}
        with tqdm(total=len(futures), desc="Grid search") as pbar:
            for future in as_completed(futures):
                charge_setpoint = futures[future][5]
                discharge_setpoint = futures[future][6]
                delivered_kwh, socs, actions, abs_energy_throughput = future.result()
                cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
                if cycles_used > max_total_cycles:
                    pbar.update(1)
                    continue
                df_1min_temp = pd.DataFrame({
                    'datetime': datetimes,
                    'qh_start': qh_starts,
                    'delivered_kwh': delivered_kwh,
                    'soc_kwh': socs,
                    'action': actions,
                    'imbalanceprice': imbalance_prices
                })
                qh_delivered = df_1min_temp.groupby('qh_start')['delivered_kwh'].sum().reset_index()
                qh_merged = pd.merge(qh_delivered, df_15min[['imbalanceprice']], left_on='qh_start', right_index=True, how='left')
                qh_merged['revenue'] = qh_merged['delivered_kwh'] / 1000 * qh_merged['imbalanceprice']
                total_revenue = qh_merged['revenue'].sum()
                if total_revenue > best_revenue:
                    best_revenue = total_revenue
                    best_result = (charge_setpoint, discharge_setpoint, df_1min_temp.copy(), qh_merged.copy())
                    best_cycles_used = cycles_used
                pbar.update(1)
    if best_result is not None:
        charge_setpoint, discharge_setpoint, df_1min, qh_merged = best_result
        print(f"Best charge setpoint: {charge_setpoint}, Best discharge setpoint: {discharge_setpoint}, Revenue: {best_revenue:.2f}")
        print(f"Average cycles per day: {round(best_cycles_used/7, 2)}")
        charge_prices = df_1min.loc[df_1min['action'] == 'charge', 'imbalanceprice']
        discharge_prices = df_1min.loc[df_1min['action'] == 'discharge', 'imbalanceprice']
        avg_charge_price = charge_prices.mean() if not charge_prices.empty else float('nan')
        avg_discharge_price = discharge_prices.mean() if not discharge_prices.empty else float('nan')
        avg_spread = avg_discharge_price - avg_charge_price
        total_injected_mwh = qh_merged.loc[qh_merged['delivered_kwh'] < 0, 'delivered_kwh'].abs().sum() / 1000
        revenue_per_mwh = best_revenue / total_injected_mwh if total_injected_mwh > 0 else float('nan')
        print(f"Average charge price: {avg_charge_price:.2f} €/MWh, Average discharge price: {avg_discharge_price:.2f} €/MWh, Average spread: {avg_spread:.2f} €/MWh")
        print(f"Revenue per MWh injected: {revenue_per_mwh:.2f} €/MWh")
        print(qh_merged.head(20))
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax1.plot(df_1min['datetime'], df_1min['soc_kwh'], '-', label='State of Charge (kWh)', color='tab:blue', linewidth=1)
        ax1.set_ylabel('State of Charge (kWh)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlabel('Datetime')
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d-%b'))
        ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))
        plt.setp(ax1.xaxis.get_minorticklabels(), rotation=0, fontsize=8, color='gray')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, fontsize=10)
        charge_mask = df_1min['action'] == 'charge'
        discharge_mask = df_1min['action'] == 'discharge'
        ax1.scatter(df_1min.loc[charge_mask, 'datetime'], df_1min.loc[charge_mask, 'soc_kwh'], color='green', label='Charge', alpha=0.5, marker='^', s=20, edgecolor='black', linewidth=0.3)
        ax1.scatter(df_1min.loc[discharge_mask, 'datetime'], df_1min.loc[discharge_mask, 'soc_kwh'], color='red', label='Discharge', alpha=0.5, marker='v', s=20, edgecolor='black', linewidth=0.3)
        ax2 = ax1.twinx()
        # Removed 1-min imbalance price plot
        ax2.step(qh_merged['qh_start'], qh_merged['imbalanceprice'], where='post', label='15-min Imbalance Price', color='tab:orange', alpha=0.9, linewidth=2)
        ax2.set_ylabel('Imbalance Price (EUR/MWh)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
        plt.title('Battery SoC, 15-min Imbalance Price, and Actions', fontsize=14)
        plt.tight_layout()
        plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.3)
        plt.show()
    else:
        print("No feasible solution found with the given constraints.")

if __name__ == "__main__":
    today = datetime.now()
    last_monday = today - timedelta(days=today.weekday() + 7)
    start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S")
    optimize_battery_with_delay_parallel(start_str, end_str)
