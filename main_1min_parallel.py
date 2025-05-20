import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIG ---
ODS133_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods133/records"
ODS134_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"

# --- DATA FETCHING ---
def fetch_minutely_imbalance(start_datetime, end_datetime):
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900
    while offset <= max_offset:
        url = (
            f"{ODS133_URL}?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
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

# --- PARALLELIZED OPTIMIZATION LOGIC ---
def simulate_setpoints(args):
    (charge_setpoint, discharge_setpoint, df_1min, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, max_total_cycles) = args
    soc = soc_init * battery_capacity_kwh
    delivered_kwh = []
    socs = []
    actions = []
    abs_energy_throughput = 0
    for idx, row in df_1min.iterrows():
        delayed_idx = idx - delay_min if idx - delay_min >= 0 else 0
        delayed_price = df_1min.iloc[delayed_idx]['imbalanceprice']
        duration = 1/60
        if delayed_price < charge_setpoint:
            charge = min(battery_power_kw, battery_capacity_kwh - soc)
            actual = charge * duration
            soc = min(soc + actual, battery_capacity_kwh)
            actions.append('charge')
        elif delayed_price > discharge_setpoint:
            discharge = min(battery_power_kw, soc)
            actual = -discharge * duration
            soc = max(soc + actual, 0)
            actions.append('discharge')
        else:
            actual = 0
            actions.append('idle')
        delivered_kwh.append(actual)
        socs.append(soc)
        abs_energy_throughput += abs(actual)
    cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
    if cycles_used > max_total_cycles:
        return None
    return (charge_setpoint, discharge_setpoint, delivered_kwh, socs, actions, cycles_used)

def optimize_battery_with_delay_parallel(start, end, battery_power_kw=50, battery_capacity_kwh=100, soc_init=0.5, delay_min=3, max_cycles_per_day=1.4, n_workers=4):
    df_1min = fetch_minutely_imbalance(start, end)
    df_15min = fetch_quarterhourly_imbalance(start, end)
    if df_1min.empty or df_15min.empty:
        print("No data fetched for the given period.")
        return
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
    df_15min['datetime'] = pd.to_datetime(df_15min['datetime'])
    df_1min['qh_start'] = df_1min['datetime'].dt.floor('15min')
    df_15min = df_15min.set_index('datetime')
    charge_price_range = range(int(df_15min['imbalanceprice'].min()), int(df_15min['imbalanceprice'].max()), 5)
    discharge_price_range = range(int(df_15min['imbalanceprice'].min()), int(df_15min['imbalanceprice'].max()), 5)
    max_total_cycles = max_cycles_per_day * 7
    best_revenue = float('-inf')
    best_result = None
    best_cycles_used = None

    # Prepare all setpoint pairs
    # Only pass the minimal data needed to each worker: pass the 1-min price as a numpy array, not the full DataFrame
    imbalance_prices = df_1min['imbalanceprice'].to_numpy()
    n_steps = len(imbalance_prices)
    qh_starts = df_1min['qh_start'].to_numpy()
    datetimes = df_1min['datetime'].to_numpy()
    setpoint_pairs = [(c, d, imbalance_prices, n_steps, qh_starts, datetimes, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, max_total_cycles)
                      for c in charge_price_range for d in discharge_price_range if d > c]

    def simulate_setpoints_minimal(args):
        (charge_setpoint, discharge_setpoint, imbalance_prices, n_steps, qh_starts, datetimes, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, max_total_cycles) = args
        soc = soc_init * battery_capacity_kwh
        delivered_kwh = np.zeros(n_steps)
        socs = np.zeros(n_steps)
        actions = np.empty(n_steps, dtype=object)
        abs_energy_throughput = 0
        for idx in range(n_steps):
            delayed_idx = idx - delay_min if idx - delay_min >= 0 else 0
            delayed_price = imbalance_prices[delayed_idx]
            duration = 1/60
            if delayed_price < charge_setpoint:
                charge = min(battery_power_kw, battery_capacity_kwh - soc)
                actual = charge * duration
                soc = min(soc + actual, battery_capacity_kwh)
                actions[idx] = 'charge'
            elif delayed_price > discharge_setpoint:
                discharge = min(battery_power_kw, soc)
                actual = -discharge * duration
                soc = max(soc + actual, 0)
                actions[idx] = 'discharge'
            else:
                actual = 0
                actions[idx] = 'idle'
            delivered_kwh[idx] = actual
            socs[idx] = soc
            abs_energy_throughput += abs(actual)
        cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
        if cycles_used > max_total_cycles:
            return None
        return (charge_setpoint, discharge_setpoint, delivered_kwh, socs, actions, cycles_used)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(simulate_setpoints_minimal, args) for args in setpoint_pairs]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            charge_setpoint, discharge_setpoint, delivered_kwh, socs, actions, cycles_used = res
            # Only reconstruct the DataFrame for the best result
            qh_starts_pd = pd.Series(qh_starts)
            datetimes_pd = pd.to_datetime(datetimes)
            df_1min_temp = pd.DataFrame({
                'datetime': datetimes_pd,
                'qh_start': qh_starts_pd,
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
    # Print results
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
        # Plot
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
        ax2.plot(df_1min['datetime'], df_1min['imbalanceprice'], '-', label='1-min Imbalance Price', color='tab:orange', alpha=0.3, linewidth=1)
        ax2.step(qh_merged['qh_start'], qh_merged['imbalanceprice'], where='post', label='15-min Imbalance Price', color='tab:orange', alpha=0.9, linewidth=2)
        ax2.set_ylabel('Imbalance Price (EUR/MWh)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
        plt.title('Battery SoC, 1-min/15-min Imbalance Price, and Actions', fontsize=14)
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
