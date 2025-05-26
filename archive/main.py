import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def fetch_opendata_elia_with_query(start_datetime, end_datetime):
    url = (
        "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"
        f"?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
        "&order_by=datetime"
        "&limit=673"
        "&timezone=Europe/Brussels"
        "&use_labels=true"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    records = data.get("results", [])
    return pd.DataFrame(records)

def fetch_weekly_opendata_elia(start_datetime, end_datetime):
    """
    Fetches all quarter-hour data for a week by making multiple API calls (limit=100, offset increments).
    Returns a combined DataFrame for the week.
    """
    all_records = []
    current_start = pd.to_datetime(start_datetime)
    current_end = pd.to_datetime(end_datetime)
    # There are 672 quarter-hours in a week, so 7 calls with 100 limit (last one may be less)
    offset = 0
    limit = 100
    while True:
        url = (
            "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"
            f"?where=datetime >= \"{start_datetime}\" AND datetime < \"{end_datetime}\""
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
        # Order by datetime to ensure correct sequence
        all_records.sort(key=lambda x: x['datetime'])
    return pd.DataFrame(all_records)

if __name__ == "__main__":
    # Example usage
    today = datetime.now()
    # Find last Monday
    last_monday = today - timedelta(days=today.weekday() + 7)
    # Start is last Monday at 00:00:00
    start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    # End is next Sunday midnight (Monday 00:00:00 after last week)
    end = start + timedelta(days=7)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S")

    df = fetch_weekly_opendata_elia(start_str, end_str)

    # Example battery parameters
    battery_power_kw = 50      # Maximum charge/discharge power in kW
    battery_capacity_kwh = 100 # Total battery capacity in kWh
    soc = 0.5 * battery_capacity_kwh  # Start at 50% state of charge

    # Check if imbalanceprice column exists
    if 'imbalanceprice' not in df.columns:
        print("No 'imbalanceprice' column found in data.")
    else:
        # Setpoint optimization: grid search over charge/discharge price setpoints
        charge_price_range = range(int(df['imbalanceprice'].min()), int(df['imbalanceprice'].max()), 5)  # step 5 EUR/MWh
        discharge_price_range = range(int(df['imbalanceprice'].min()), int(df['imbalanceprice'].max()), 5)
        best_revenue = float('-inf')
        best_charge_setpoint = None
        best_discharge_setpoint = None
        best_results = None
        best_cycles_used = None

        max_cycles_per_day = 1.4  # Default constraint
        max_total_cycles = max_cycles_per_day * 7
        # Each cycle = 2 * battery_capacity_kwh (full charge + full discharge)
        # We'll count total absolute energy throughput and divide by (2 * capacity) to get cycles

        for charge_setpoint in charge_price_range:
            for discharge_setpoint in discharge_price_range:
                if discharge_setpoint <= charge_setpoint:
                    continue  # Only consider discharge > charge setpoint
                soc = 0.5 * battery_capacity_kwh
                delivered_kwh = []
                socs = []
                actions = []
                abs_energy_throughput = 0
                for idx, row in df.iterrows():
                    price = row['imbalanceprice']
                    duration = row.get('duration_hours', 0.25)  # 15 min = 0.25h
                    if price < charge_setpoint:
                        # Charge
                        charge = min(battery_power_kw, battery_capacity_kwh - soc)
                        actual = charge * duration
                        soc = min(soc + actual, battery_capacity_kwh)
                        actions.append('charge')
                    elif price > discharge_setpoint:
                        # Discharge
                        discharge = min(battery_power_kw, soc)
                        actual = -discharge * duration
                        soc = max(soc + actual, 0)
                        actions.append('discharge')
                        actual = -actual
                    else:
                        actual = 0
                        actions.append('idle')
                    delivered_kwh.append(actual)
                    socs.append(soc)
                    abs_energy_throughput += abs(actual)
                # Calculate cycles used
                cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
                if cycles_used > max_total_cycles:
                    continue  # Skip this setpoint pair, exceeds cycle constraint
                df['delivered_kwh'] = delivered_kwh
                df['soc_kwh'] = socs
                df['action'] = actions
                df['revenue'] = df['delivered_kwh'] / 1000 * df['imbalanceprice']
                total_revenue = df['revenue'].sum()
                if total_revenue > best_revenue:
                    best_revenue = total_revenue
                    best_charge_setpoint = charge_setpoint
                    best_discharge_setpoint = discharge_setpoint
                    best_cycles_used = cycles_used 
                    best_results = df.copy()

        print(f"Best charge setpoint: {best_charge_setpoint}, Best discharge setpoint: {best_discharge_setpoint}, Revenue: {best_revenue:.2f}")
        print(f"Average cycles per day: {round(best_cycles_used/7, 2)}")
        # Calculate average spread and revenue per MWh injected
        charge_prices = best_results.loc[best_results['action'] == 'charge', 'imbalanceprice']
        discharge_prices = best_results.loc[best_results['action'] == 'discharge', 'imbalanceprice']
        avg_charge_price = charge_prices.mean() if not charge_prices.empty else float('nan')
        avg_discharge_price = discharge_prices.mean() if not discharge_prices.empty else float('nan')
        avg_spread = avg_discharge_price - avg_charge_price

        print(f"Average charge price: {avg_charge_price:.2f} €/MWh, Average discharge price: {avg_discharge_price:.2f} €/MWh, Average spread: {avg_spread:.2f} €/MWh")

        print(best_results[['datetime', 'imbalanceprice', 'delivered_kwh', 'soc_kwh', 'action', 'revenue']].head(20))

        # Plotting results (cleaner and more dynamic)
        import matplotlib.dates as mdates
        if not pd.api.types.is_datetime64_any_dtype(best_results['datetime']):
            best_results['datetime'] = pd.to_datetime(best_results['datetime'])
        fig, ax1 = plt.subplots(figsize=(15, 6))
        # Plot SoC as a line
        ax1.plot(best_results['datetime'], best_results['soc_kwh'], '-', label='State of Charge (kWh)', color='tab:blue', linewidth=2)
        ax1.set_ylabel('State of Charge (kWh)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlabel('Datetime')
        # Format x-axis for better readability
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d-%b'))
        ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))
        plt.setp(ax1.xaxis.get_minorticklabels(), rotation=0, fontsize=8, color='gray')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, fontsize=10)
        # Plot charge/discharge events as scatter points
        charge_mask = best_results['action'] == 'charge'
        discharge_mask = best_results['action'] == 'discharge'
        ax1.scatter(best_results.loc[charge_mask, 'datetime'], best_results.loc[charge_mask, 'soc_kwh'], color='green', label='Charge', alpha=0.7, marker='^', s=50, edgecolor='black', linewidth=0.5)
        ax1.scatter(best_results.loc[discharge_mask, 'datetime'], best_results.loc[discharge_mask, 'soc_kwh'], color='red', label='Discharge', alpha=0.7, marker='v', s=50, edgecolor='black', linewidth=0.5)
        # Imbalance price on secondary axis as a line
        ax2 = ax1.twinx()
        ax2.plot(best_results['datetime'], best_results['imbalanceprice'], '-', label='Imbalance Price', color='tab:orange', alpha=0.6, linewidth=2)
        ax2.set_ylabel('Imbalance Price (EUR/MWh)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        # Combine legends from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
        plt.title('Battery State of Charge, Charge/Discharge Events, and Imbalance Price', fontsize=14)
        plt.tight_layout()
        plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.3)
        plt.show()