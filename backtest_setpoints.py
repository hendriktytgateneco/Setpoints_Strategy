"""
backtest_setpoints.py

Backtest battery operation for user-specified charge and discharge setpoints.
- Allows you to specify charge/discharge setpoints and see the resulting revenue, cycles, and other metrics for a given period.
- Optionally uses a local CSV cache for imbalance price data to avoid repeated API calls for the same period.
- Automatically removes outdated cache files (older than 30 days by default).

How to use:
- Run the script and enter your desired setpoints and date range when prompted, or set them in the code.
- The script will fetch (or load) the price data, simulate the battery, and print/save the results.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from import_data import fetch_minutely_imbalance, fetch_quarterhourly_imbalance, clean_cache

def simulate_battery_backtest(df_1min, charge_setpoint, discharge_setpoint, battery_power_kw=50, battery_capacity_kwh=100, soc_init=0.5, delay_min=3):
    n_steps = len(df_1min)
    imbalance_prices = df_1min['imbalanceprice'].to_numpy()
    socs = np.zeros(n_steps)
    delivered_kwh = np.zeros(n_steps)
    actions = np.empty(n_steps, dtype=object)
    soc = soc_init * battery_capacity_kwh
    duration = 1/60
    delayed_indices = np.arange(n_steps) - delay_min
    delayed_indices[delayed_indices < 0] = -1
    delayed_prices = np.full(n_steps, np.nan)
    valid_mask = delayed_indices >= 0
    delayed_prices[valid_mask] = imbalance_prices[delayed_indices[valid_mask]]
    nan_mask = np.isnan(delayed_prices)
    charge_mask = (delayed_prices < charge_setpoint) & (~nan_mask)
    discharge_mask = (delayed_prices > discharge_setpoint) & (~nan_mask)
    for idx in range(n_steps):
        if nan_mask[idx]:
            actual = 0
            actions[idx] = 'idle'
        elif charge_mask[idx]:
            charge = min(battery_power_kw, battery_capacity_kwh - soc)
            actual = charge * duration
            soc = min(soc + actual, battery_capacity_kwh)
            actions[idx] = 'charge'
            actual = -actual
        elif discharge_mask[idx]:
            discharge = min(battery_power_kw, soc)
            actual = -discharge * duration
            soc = max(soc + actual, 0)
            actions[idx] = 'discharge'
            actual = -actual
        else:
            actual = 0
            actions[idx] = 'idle'
        delivered_kwh[idx] = actual
        socs[idx] = soc
    abs_energy_throughput = np.sum(np.abs(delivered_kwh))
    df_1min['delivered_kwh'] = delivered_kwh
    df_1min['soc_kwh'] = socs
    df_1min['action'] = actions
    return df_1min, abs_energy_throughput

def main():
    clean_cache()

    # User input (or set here)
    today = datetime.now().date()
    last = today - timedelta(days=7)
    default_start = last.strftime("%Y-%m-%dT00:00:00")
    default_end = today.strftime("%Y-%m-%dT00:00:00")
    start = input(f"Enter start datetime (YYYY-MM-DDTHH:MM:SS) [{default_start}]: ") or default_start
    end = input(f"Enter end datetime (YYYY-MM-DDTHH:MM:SS) [{default_end}]: ") or default_end
    charge_setpoint = float(input("Enter charge setpoint (€/MWh): ") or 20)
    discharge_setpoint = float(input("Enter discharge setpoint (€/MWh): ") or 100)
    battery_power_kw = float(input("Enter battery power (kW): ") or 50)
    battery_capacity_kwh = float(input("Enter battery capacity (kWh): ") or 100)
    delay_min = int(input("Enter delay in minutes: ") or 3)
    soc_init = float(input("Enter initial SoC (0-1): ") or 0.5)

    # Fetch or load data
    df_1min = fetch_minutely_imbalance(start, end)
    df_15min = fetch_quarterhourly_imbalance(start, end)
    clean_cache()
    if df_1min.empty or df_15min.empty:
        print("No data fetched for the given period.")
        return
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
    df_1min['qh_start'] = df_1min['datetime'].dt.floor('15min')
    df_15min['datetime'] = pd.to_datetime(df_15min['datetime'])
    df_15min = df_15min.set_index('datetime')

    # Simulate
    df_1min, abs_energy_throughput = simulate_battery_backtest(
        df_1min, charge_setpoint, discharge_setpoint, battery_power_kw, battery_capacity_kwh, soc_init, delay_min
    )

    # Calculate results
    qh_delivered = df_1min.groupby('qh_start')['delivered_kwh'].sum().reset_index()
    qh_merged = pd.merge(qh_delivered, df_15min[['imbalanceprice']], left_on='qh_start', right_index=True, how='left')
    qh_merged['revenue'] = qh_merged['delivered_kwh'] / 1000 * qh_merged['imbalanceprice']
    total_revenue = qh_merged['revenue'].sum()
    cycles_used = abs_energy_throughput / (2 * battery_capacity_kwh)
    charge_prices = df_1min.loc[df_1min['action'] == 'charge', 'imbalanceprice']
    discharge_prices = df_1min.loc[df_1min['action'] == 'discharge', 'imbalanceprice']
    avg_charge_price = charge_prices.mean() if not charge_prices.empty else float('nan')
    avg_discharge_price = discharge_prices.mean() if not discharge_prices.empty else float('nan')
    avg_spread = avg_discharge_price - avg_charge_price
    total_injected_mwh = qh_merged.loc[qh_merged['delivered_kwh'] < 0, 'delivered_kwh'].abs().sum() / 1000
    revenue_per_mwh = total_revenue / total_injected_mwh if total_injected_mwh > 0 else float('nan')
    num_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
    avg_cycles_per_day = cycles_used / num_days if num_days > 0 else float('nan')
    print(f"Backtest results for {start} to {end}:")
    print(f"Charge setpoint: {charge_setpoint}, Discharge setpoint: {discharge_setpoint}")
    print(f"Revenue: {total_revenue:.2f} €")
    print(f"Average Cycles per day: {avg_cycles_per_day:.2f}")
    print(f"Average charge price: {avg_charge_price:.2f} €/MWh, Average discharge price: {avg_discharge_price:.2f} €/MWh, Spread: {avg_spread:.2f} €/MWh")
    print(f"Revenue per MWh injected: {revenue_per_mwh:.2f} €/MWh")

    # Plot and save to PDF
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
    pdf_filename = f"backtest_{start[:10]}_{end[:10]}_setpoints_{int(charge_setpoint)}_{int(discharge_setpoint)}.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Add text page with summary in requested format
        fig_text, ax_text = plt.subplots(figsize=(8.5, 4))
        ax_text.axis('off')
        summary = (
            f"Battery Optimization Results\n"
            f"Start: {start}\n"
            f"End: {end}\n"
            f"Battery Power & Energy: {int(battery_power_kw)}kW | {int(battery_capacity_kwh)}kWh\n"
            f"Best charge setpoint: {charge_setpoint:g}\n"
            f"Best discharge setpoint: {discharge_setpoint:g}\n"
            f"Revenue: {total_revenue:.2f}\n"
            f"Average cycles per day: {avg_cycles_per_day:.2f}\n"
            f"Average charge price: {avg_charge_price:.2f} /MWh\n"
            f"Average discharge price: {avg_discharge_price:.2f} /MWh\n"
            f"Average spread: {avg_spread:.2f} /MWh\n"
            f"Revenue per MWh injected: {revenue_per_mwh:.2f} /MWh\n"
        )
        ax_text.text(0.01, 0.99, summary, va='top', ha='left', fontsize=12, family='monospace')
        pdf.savefig(fig_text)
        plt.close(fig_text)
        # Add the main plot
        pdf.savefig(fig)

    # Plot and save to PDF
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
    pdf_filename = f"backtest_{start[:10]}_{end[:10]}_setpoints_{int(charge_setpoint)}_{int(discharge_setpoint)}.pdf"

    with PdfPages(pdf_filename) as pdf:
        # Add text page with summary
        fig_text, ax_text = plt.subplots(figsize=(8.5, 4))
        ax_text.axis('off')
        summary = (
            f"Backtest Results\n\n"
            f"Start: {start}\nEnd: {end}\n\n"
            f"Charge setpoint: {charge_setpoint}\n"
            f"Discharge setpoint: {discharge_setpoint}\n"
            f"Revenue: {total_revenue:.2f} €\n"
            f"Average Daily Cycles used: {avg_cycles_per_day:.2f}\n"
            f"Average charge price: {avg_charge_price:.2f} €/MWh\n"
            f"Average discharge price: {avg_discharge_price:.2f} €/MWh\n"
            f"Spread: {avg_spread:.2f} €/MWh\n"
            f"Revenue per MWh injected: {revenue_per_mwh:.2f} €/MWh\n"
        )
        ax_text.text(0.01, 0.99, summary, va='top', ha='left', fontsize=12, family='monospace')
        pdf.savefig(fig_text)
        plt.close(fig_text)

        # Add the main plot
        pdf.savefig(fig)
    print(f"Backtest results and plot saved to {pdf_filename}")

if __name__ == "__main__":
    main()
