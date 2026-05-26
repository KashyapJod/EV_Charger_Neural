# EV Charger Placement Optimizer

This repository contains an EV charger placement optimization tool built with a neural network-based optimizer, GUI interface, and headless execution mode. It supports:

- interactive GUI optimization and visualization
- headless optimization runs with saved JSON results
- publication-ready figure generation
- baseline method comparison (grid vs random vs optimized placement)
- saved result preview and map export

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the GUI

Start the GUI application with:

```bash
python main.py
```

### GUI menu access

The application opens a window with three tabs:

- **Optimization**
  - `Number of new chargers`: choose how many new stations to place.
  - `Coverage radius (km)`: set the service radius for each charger.
  - `Existing chargers to include`: select which stored existing chargers should be considered in the optimization.
  - `Show Advanced Options`: expand this section to adjust feature weights for the optimization model.
  - `Calculate Optimal Locations`: run the optimizer and generate results.
  - `Stop Calculation`: stop the current background optimization worker.

- **Results**
  - `Show Map`: render the current result on an interactive HTML map.
  - `Save Results`: persist the current optimization outcome to `results/`.
  - `Load Results`: load a previously saved result file.
  - `Calculate Efficiency`: compute model efficiency metrics.
  - `Generate Report`: create a text-style report for the current result.
  - `Generate Figures`: export publication-ready plot images for the current result.
  - `Report Type`: choose between `Summary`, `Coverage Analysis`, `Distance Matrix`, `Weights Overview`, `Efficiency Metrics`, or `Full Report`.

- **Settings**
  - `UI Theme`: switch between White, Dark, and System appearance.
  - `UI Scaling`: adjust interface scale.
  - `Clear Cache`: reset optimizer caches.

## Running the headless optimizer

Use the headless runner to compute results without the GUI:

```bash
python run_headless.py
```

This performs the optimization, saves a JSON result to `results/`, and automatically generates publication-ready figure images under `results/figures/run_<timestamp>/`.

## Previewing saved results

To load and preview the latest saved result in the browser:

```bash
python preview_locations.py
```

This script loads the newest `optimization_result_*.json` file, regenerates the coverage stats, and opens `gui/charger_map.html`.

## Output files

- `results/optimization_result_<timestamp>.json` — saved optimization data and parameters.
- `results/figures/run_<timestamp>/*.png` — generated publication figures:
  - `charger_layout.png`
  - `coverage_summary.png`
  - `weight_distribution.png`
  - `distance_distribution.png`
  - `efficiency_overview.png`
  - `method_comparison.png`

## What the generated graphs show

- **Charger layout**: existing and new charger positions with approximate coverage circles.
- **Coverage summary**: coverage, overlap, and efficiency percentages.
- **Weight distribution**: relative importance of the optimization features.
- **Distance distribution**: pairwise separation between new charger locations.
- **Efficiency overview**: overall performance metrics for the current placement.
- **Method comparison**: direct comparison of optimized placement versus grid and random baseline placements.

## Notes for research and publication

- Use the generated figures from `results/figures/` in your paper.
- The method comparison chart is especially useful to demonstrate improvement over common baseline placement strategies.
- The generated report and JSON output can be referenced for experimental setup, parameters, and numeric results.

## Requirements

- Python 3.9+
- Optional: GPU support for TensorFlow if running advanced models, but not required for current execution.
