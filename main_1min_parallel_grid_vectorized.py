"""
main_1min_parallel_grid_vectorized.py

Battery Optimization with Parallel and Vectorized Simulation
----------------------------------------------------------
This script optimizes the operation of a battery using Elia's 1-minute and 15-minute imbalance price data for a given week.
It finds the best charge and discharge price setpoints to maximize revenue, while respecting battery cycling constraints.

Key Features:
- Fetches 1-min and 15-min price data for a week, handling API row limits.
- Simulates battery operation for many setpoint combinations using parallel processing (all CPU cores).
- Uses a vectorized approach for efficient simulation.
- Automatically focuses the search on realistic price ranges (ignores outliers).
- Saves a summary and graph of the best result to a PDF file.
- Easy to run: just execute the script, no technical background needed.

How It Works (Simple Explanation):
1. The script downloads price data for the last full week.
2. It tries many combinations of charge and discharge price thresholds (setpoints).
3. For each combination, it simulates how the battery would operate and calculates the revenue.
4. It keeps track of the best result (highest revenue, within cycling limits).
5. It saves a PDF with a summary and a graph of the best strategy.

How to Control the Number of Iterations:
- The number of setpoint combinations (iterations) depends on the price range and the step size (see 'step' variable).
- To increase iterations (finer search): decrease 'step' (e.g., step = 1).
- To decrease iterations (coarser search): increase 'step' (e.g., step = 5).
- The script automatically uses all available CPU cores for speed.

How to Use All CPU Cores:
- The script uses Python's ProcessPoolExecutor, which by default uses as many processes as there are CPU cores.
- You can control the number of workers by changing 'ProcessPoolExecutor(max_workers=...)' if you want to limit or increase it (not usually needed).

"""
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from import_data import fetch_minutely_imbalance, fetch_quarterhourly_imbalance, clean_cache

def simulate_vectorized_setpoints_worker(args):
    """
    Fully vectorized simulation of battery operation for a given set of setpoints (charge/discharge) and battery parameters.
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
    """
    (
        imbalance_prices, battery_power_kw, battery_capacity_kwh, soc_init, delay_min, charge_setpoint, discharge_setpoint
    ) = args
    n_steps = len(imbalance_prices)
    socs = np.zeros(n_steps)
    delivered_kwh = np.zeros(n_steps)
    actions = np.empty(n_steps, dtype=object)
    soc = soc_init * battery_capacity_kwh
    duration = 1/60
    # Compute delayed prices with NaN for invalid indices
    delayed_indices = np.arange(n_steps) - delay_min
    delayed_indices[delayed_indices < 0] = -1  # mark invalid indices
    delayed_prices = np.full(n_steps, np.nan)
    valid_mask = delayed_indices >= 0
    delayed_prices[valid_mask] = imbalance_prices[delayed_indices[valid_mask]]
    # Vectorized masks
    nan_mask = np.isnan(delayed_prices)
    charge_mask = (delayed_prices < charge_setpoint) & (~nan_mask)
    discharge_mask = (delayed_prices > discharge_setpoint) & (~nan_mask)
    idle_mask = ~(charge_mask | discharge_mask) | nan_mask
    # Step through time, updating SoC sequentially (SoC is path-dependent)
    for idx in range(n_steps):
        if nan_mask[idx]:
            actual = 0
            actions[idx] = 'idle'
        elif charge_mask[idx]:
            charge = min(battery_power_kw, battery_capacity_kwh - soc)
            actual = charge * duration
            soc = min(soc + actual, battery_capacity_kwh)
            actions[idx] = 'charge'
            actual  = -actual #invert sign for delivered kWh so revenue is positive
        elif discharge_mask[idx]:
            discharge = min(battery_power_kw, soc)
            actual = -discharge * duration
            soc = max(soc + actual, 0)
            actions[idx] = 'discharge'
            actual = -actual #invert sign for delivered kwh so revenue is positive
        else:
            actual = 0
            actions[idx] = 'idle'
        delivered_kwh[idx] = actual
        socs[idx] = soc
    abs_energy_throughput = np.sum(np.abs(delivered_kwh))
    return delivered_kwh, socs, actions, abs_energy_throughput

def optimize_battery_with_delay_parallel(
    start, end, step = 5 ,battery_power_kw=50, battery_capacity_kwh=100, soc_init=0.5, delay_min=3, max_cycles_per_day=1.4
):
    df_1min = fetch_minutely_imbalance(start, end)
    df_15min = fetch_quarterhourly_imbalance(start, end)
    clean_cache()
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
    # Use percentiles for realistic setpoint bounds
    low, high = np.percentile(df_15min['imbalanceprice'], [5, 95])
    # You can adjust the step size for grid density
    charge_price_range = np.arange(int(low), int(high), step)
    discharge_price_range = np.arange(int(low), int(high), step)
    num_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
    max_total_cycles = max_cycles_per_day * num_days
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
        with tqdm(total=len(futures), desc="Grid search (vectorized)") as pbar:
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
        ax2.step(qh_merged['qh_start'], qh_merged['imbalanceprice'], where='post', label='15-min Imbalance Price', color='tab:orange', alpha=0.9, linewidth=2)
        ax2.set_ylabel('Imbalance Price (EUR/MWh)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
        plt.title('Battery SoC, 15-min Imbalance Price, and Actions', fontsize=14)
        plt.tight_layout()
        plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.3)
        # Save results and plot to PDF
        pdf_filename = f"battery_optimization_{start[:10]}_{end[:10]}.pdf"
        with PdfPages(pdf_filename) as pdf:
            # Add text page with summary
            fig_text, ax_text = plt.subplots(figsize=(8.5, 4))
            ax_text.axis('off')
            summary = (
                f"Battery Optimization Results\n\n"
                f"Start: {start}\nEnd: {end}\n\n"
                f"Battery Power & Energy: {battery_power_kw}kW | {battery_capacity_kwh}kWh"
                f"Best charge setpoint: {charge_setpoint}\n"
                f"Best discharge setpoint: {discharge_setpoint}\n"
                f"Revenue: {best_revenue:.2f} €\n"
                f"Average cycles per day: {round(best_cycles_used/7, 2)}\n"
                f"Average charge price: {avg_charge_price:.2f} €/MWh\n"
                f"Average discharge price: {avg_discharge_price:.2f} €/MWh\n"
                f"Average spread: {avg_spread:.2f} €/MWh\n"
                f"Revenue per MWh injected: {revenue_per_mwh:.2f} €/MWh\n"
            )
            ax_text.text(0.01, 0.99, summary, va='top', ha='left', fontsize=12, family='monospace')
            pdf.savefig(fig_text)
            plt.close(fig_text)
            # Add the main plot
            pdf.savefig(fig)
        print(f"Results and plot saved to {pdf_filename}")
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

