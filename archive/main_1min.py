import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# --- CONFIG ---
ODS133_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods133/records"
ODS134_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"

# --- DATA FETCHING ---
def fetch_minutely_imbalance(start_datetime, end_datetime):
    """
    Fetch 1-min imbalance price data from ods133 for a given period (with pagination).
    """
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900  # offset+limit < 10000
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
    """
    Fetch 15-min imbalance price data from ods134 for a given period (with pagination).
    """
    all_records = []
    offset = 0
    limit = 100
    max_offset = 9900  # offset+limit < 10000
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

# --- OPTIMIZATION LOGIC ---
def optimize_battery_with_delay(start, end, battery_power_kw=50, battery_capacity_kwh=100, soc_init=0.5, delay_min=3, max_cycles_per_day=1.4):
    """
    Optimize battery setpoints using 1-min imbalance price (with delay), but revenue is calculated on 15-min price.
    """
    # Fetch data
    df_1min = fetch_minutely_imbalance(start, end)
    df_15min = fetch_quarterhourly_imbalance(start, end)
    if df_1min.empty or df_15min.empty:
        print("No data fetched for the given period.")
        return
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
    df_15min['datetime'] = pd.to_datetime(df_15min['datetime'])
    # Merge 1-min and 15-min data for mapping
    df_1min['qh_start'] = df_1min['datetime'].dt.floor('15min')
    df_15min = df_15min.set_index('datetime')
    # Setpoint grid
    charge_price_range = range(int(df_1min['imbalanceprice'].min()), int(df_1min['imbalanceprice'].max()), 5)
    discharge_price_range = range(int(df_1min['imbalanceprice'].min()), int(df_1min['imbalanceprice'].max()), 5)
    best_revenue = float('-inf')
    best_charge_setpoint = None
    best_discharge_setpoint = None
    best_results = None
    best_cycles_used = None
    max_total_cycles = max_cycles_per_day * 7
    for charge_setpoint in charge_price_range:
        for discharge_setpoint in discharge_price_range:
            if discharge_setpoint <= charge_setpoint:
                continue
            soc = soc_init * battery_capacity_kwh
            delivered_kwh = []
            socs = []
            actions = []
            abs_energy_throughput = 0

            for idx, row in df_1min.iterrows():
                # Apply delay: only act on the price from (now - delay_min)
                delayed_idx = idx - delay_min if idx - delay_min >= 0 else 0
                delayed_price = df_1min.iloc[delayed_idx]['imbalanceprice']
                duration = 1/60  # 1 min in hours
                if delayed_price < charge_setpoint:
                    # Charge
                    charge = min(battery_power_kw, battery_capacity_kwh - soc)
                    actual = charge * duration
                    soc = min(soc + actual, battery_capacity_kwh)
                    actions.append('charge')
                elif delayed_price > discharge_setpoint:
                    # Discharge
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
            # Map 1-min actions to 15-min intervals for revenue calculation
            df_1min['delivered_kwh'] = delivered_kwh
            df_1min['soc_kwh'] = socs
            df_1min['action'] = actions
            # Aggregate delivered_kwh per 15-min interval
            qh_delivered = df_1min.groupby('qh_start')['delivered_kwh'].sum().reset_index()
            # Merge with 15-min price
            qh_merged = pd.merge(qh_delivered, df_15min[['imbalanceprice']], left_on='qh_start', right_index=True, how='left')
            qh_merged['revenue'] = qh_merged['delivered_kwh'] / 1000 * qh_merged['imbalanceprice']
            total_revenue = qh_merged['revenue'].sum()
            cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
            if cycles_used > max_total_cycles:
                continue
            if total_revenue > best_revenue:
                best_revenue = total_revenue
                best_charge_setpoint = charge_setpoint
                best_discharge_setpoint = discharge_setpoint
                best_cycles_used = cycles_used
                best_results = (df_1min.copy(), qh_merged.copy())
    # Print results
    if best_results is not None:
        df_1min, qh_merged = best_results
        print(f"Best charge setpoint: {best_charge_setpoint}, Best discharge setpoint: {best_discharge_setpoint}, Revenue: {best_revenue:.2f}")
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
        # Overlay 15-min price as step
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
    optimize_battery_with_delay(start_str, end_str)
