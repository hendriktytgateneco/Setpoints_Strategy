# Battery Setpoint Optimization & Backtesting

## Executive Summary
This project automates the optimization and backtesting of battery operation strategies using Belgian grid imbalance price data. It enables data-driven decision-making for maximizing battery revenue, with robust reporting and extensible code.

## Key Features
- **Automated Data Fetching:** Downloads and caches 1-min and 15-min imbalance price data from Elia's ODS API.
- **Battery Optimization:** Finds optimal charge/discharge setpoints to maximize revenue, subject to cycle constraints.
- **Parallel & Vectorized Simulation:** Fast grid search using all CPU cores or GPU (if available).
- **Backtesting:** Simulate battery operation for user-specified setpoints and periods.
- **PDF Reporting:** Generates clear, shareable PDF reports for each run.
- **Extensible & Modular:** Easy to adapt for new strategies or data sources.

## Main Scripts
- `main_1min_parallel_grid_vectorized.py`: Parallel, vectorized weekly optimization. Produces a PDF summary and plot.
- `backtest_setpoints.py`: Backtest for user-specified setpoints and period. Interactive CLI.
- `import_data.py`: Data fetching and caching utilities.
- `archive/`: Older or alternative implementations (serial, parallel, GPU, etc.).

## Usage

### 1. Weekly Optimization
Run the main optimizer for the last full week:
```sh
python main_1min_parallel_grid_vectorized.py
```
- Results and plots are saved as `battery_optimization_<start>_<end>.pdf`.

### 2. Backtest Custom Setpoints
Run the backtest script and follow the prompts:
```sh
python backtest_setpoints.py
```
- Enter the desired date range and setpoints.
- Results and plots are saved as `backtest_<start>_<end>_setpoints_<charge>_<discharge>.pdf`.

### 3. Data Caching
- Data is cached in the `data_cache/` directory.
- Old cache files (older than 30 days) are automatically cleaned.

## Requirements
- Python 3.8 or higher
- Required packages: `pandas`, `numpy`, `matplotlib`, `requests`, `tqdm`
- Optional: `cupy` (for GPU acceleration in some archive scripts)
- For Outlook email automation: `pywin32` (Windows only)

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Project Structure
- `main_1min_parallel_grid_vectorized.py` — Main optimizer (parallel, vectorized)
- `backtest_setpoints.py` — User-driven backtesting
- `import_data.py` — Data fetching and caching
- `archive/` — Alternative/older scripts
- `data_cache/` — Cached data files
- `price_cache/` — (Unused/legacy)
- `.gitignore` — Ignores cache, output, and temp files

## Notes
- All scripts fetch data directly from Elia's open data API.
- Revenue is calculated using 15-min prices, but battery operation is simulated at 1-min resolution.
- Setpoint optimization is subject to a cycle constraint (default: 1.4 cycles/day).
- For troubleshooting, ensure all dependencies are installed and that your Python version is supported.

## License
This project is for research and educational purposes. Please check Elia's data usage terms for API access.

---
For questions or contributions, please open an issue or pull request.