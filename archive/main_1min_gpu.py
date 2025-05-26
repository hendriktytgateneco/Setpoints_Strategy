import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

ODS133_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods133/records"
ODS134_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"

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

def simulate_vectorized_setpoints_gpu(
    imbalance_prices, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, charge_setpoint, discharge_setpoint
):
    if not CUPY_AVAILABLE:
        raise ImportError("cupy is not installed. Please install cupy to use GPU acceleration.")
    n_steps = len(imbalance_prices)
    imbalance_prices = cp.asarray(imbalance_prices)
    socs = cp.zeros(n_steps)
    delivered_kwh = cp.zeros(n_steps)
    actions = cp.empty(n_steps, dtype=cp.int8)  # 0: idle, 1: charge, 2: discharge
    soc = soc_init * battery_capacity_kwh
    delayed_indices = cp.arange(n_steps) - delay_min
    delayed_indices = cp.where(delayed_indices < 0, 0, delayed_indices)
    delayed_prices = imbalance_prices[delayed_indices]
    duration = 1/60
    for idx in range(n_steps):
        price = delayed_prices[idx]
        if price < charge_setpoint:
            charge = min(battery_power_kw, battery_capacity_kwh - soc)
            actual = charge * duration
            soc = min(soc + actual, battery_capacity_kwh)
            actions[idx] = 1
        elif price > discharge_setpoint:
            discharge = min(battery_power_kw, soc)
            actual = -discharge * duration
            soc = max(soc + actual, 0)
            actions[idx] = 2
        else:
            actual = 0
            actions[idx] = 0
        delivered_kwh[idx] = actual
        socs[idx] = soc
    abs_energy_throughput = cp.sum(cp.abs(delivered_kwh))
    return cp.asnumpy(delivered_kwh), cp.asnumpy(socs), cp.asnumpy(actions), float(abs_energy_throughput)

def optimize_battery_with_delay_gpu(
    start, end, battery_power_kw=50, battery_capacity_kwh=100, soc_init=0.5, delay_min=3, max_cycles_per_day=1.4
):
    if not CUPY_AVAILABLE:
        print("cupy is not installed. Please install cupy to use GPU acceleration.")
        return
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
    for charge_setpoint in charge_price_range:
        for discharge_setpoint in discharge_price_range:
            if discharge_setpoint <= charge_setpoint:
                continue
            delivered_kwh, socs, actions, abs_energy_throughput = simulate_vectorized_setpoints_gpu(
                imbalance_prices, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, charge_setpoint, discharge_setpoint
            )
            cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
            if cycles_used > max_total_cycles:
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
    if best_result is not None:
        charge_setpoint, discharge_setpoint, df_1min, qh_merged = best_result
        print(f"Best charge setpoint: {charge_setpoint}, Best discharge setpoint: {discharge_setpoint}, Revenue: {best_revenue:.2f}")
        print(f"Average cycles per day: {round(best_cycles_used/7, 2)}")
        charge_prices = df_1min.loc[df_1min['action'] == 1, 'imbalanceprice']
        discharge_prices = df_1min.loc[df_1min['action'] == 2, 'imbalanceprice']
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
        charge_mask = df_1min['action'] == 1
        discharge_mask = df_1min['action'] == 2
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
    optimize_battery_with_delay_gpu(start_str, end_str)
