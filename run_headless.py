"""
Headless runner for the EV Charger Placement Optimizer.
Runs the neural network model without the GUI and prints results to console.
"""
import os
# Set environment variables before any other imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['KMP_WARNINGS'] = '0'
os.environ['DEFAULT_COVERAGE_RADIUS'] = '4.0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import time
import json
import numpy as np

# Now import project modules
from models.neural_optimizer import ChargerOptimizer
from models.location import Location, haversine_distance
from data.results_handler import ResultsHandler

def print_separator(title=""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def generate_grid_baseline(optimizer: ChargerOptimizer, num_new_chargers: int):
    candidates = optimizer._generate_candidates(num_new_chargers * 6)
    selected = []
    for candidate in candidates:
        if any(haversine_distance(candidate, existing) < optimizer.min_distance_between_chargers
               for existing in optimizer.existing_chargers + selected):
            continue
        selected.append(candidate)
        if len(selected) >= num_new_chargers:
            break
    return selected


def generate_random_baseline(optimizer: ChargerOptimizer, num_new_chargers: int, max_attempts: int = 3000):
    min_lat, max_lat, min_lon, max_lon = optimizer.area_bounds
    selected = []
    attempts = 0
    while len(selected) < num_new_chargers and attempts < max_attempts:
        attempts += 1
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)
        if not optimizer._is_point_in_goa(lat, lon, optimizer.coastline):
            continue
        candidate = Location(lat, lon)
        if any(haversine_distance(candidate, existing) < optimizer.min_distance_between_chargers
               for existing in optimizer.existing_chargers + selected):
            continue
        selected.append(candidate)
    return selected


def run_optimization():
    # ── Configuration ──
    goa_bounds = (15.14, 15.72, 73.71, 74.15)
    num_chargers = 10  # Reasonable number for quick results
    coverage_radius = 5.0

    existing_chargers = [
        Location(15.62, 73.81, True),   # Pernem
        Location(15.18, 73.95, True),   # Canacona
        Location(15.38, 74.10, True),   # Dharbandora
        Location(15.49, 73.79, True),   # Bardez
        Location(15.40, 73.89, True),   # Ponda
    ]

    weights = {
        'population_density': 0.20,
        'traffic_flow': 0.25,
        'points_of_interest': 0.15,
        'power_availability': 0.15,
        'charging_demand': 0.10,
        'accessibility': 0.10,
        'road_quality': 0.03,
        'revenue_potential': 0.02,
    }

    # ── Initialize Optimizer ──
    print_separator("EV CHARGER PLACEMENT OPTIMIZER — GOA, INDIA")
    print(f"\nConfiguration:")
    print(f"  Area bounds (lat): {goa_bounds[0]:.2f}° – {goa_bounds[1]:.2f}°")
    print(f"  Area bounds (lon): {goa_bounds[2]:.2f}° – {goa_bounds[3]:.2f}°")
    print(f"  Coverage radius:   {coverage_radius} km")
    print(f"  Existing chargers: {len(existing_chargers)}")
    print(f"  New chargers:      {num_chargers}")
    print(f"\nOptimization weights:")
    for k, v in weights.items():
        print(f"    {k:25s} {v:.2f}")

    print_separator("INITIALIZING NEURAL NETWORK MODEL")
    start_init = time.time()
    optimizer = ChargerOptimizer(
        area_bounds=goa_bounds,
        existing_chargers=existing_chargers,
        coverage_radius=coverage_radius,
        weights=weights,
    )
    init_time = time.time() - start_init
    print(f"  Model initialized in {init_time:.2f}s")

    # Print model summary
    print(f"\n  Neural Network Architecture:")
    optimizer.model.summary(print_fn=lambda x: print(f"    {x}"))

    # ── Run Optimization ──
    print_separator("RUNNING OPTIMIZATION")
    start_opt = time.time()
    new_locations = optimizer.generate_optimal_locations(num_chargers)
    opt_time = time.time() - start_opt
    print(f"\n  Optimization completed in {opt_time:.2f}s")
    print(f"  Found {len(new_locations)} optimal locations")

    # ── Display New Charger Locations ──
    print_separator("OPTIMAL NEW CHARGER LOCATIONS")
    print(f"  {'#':<4} {'Latitude':>10} {'Longitude':>11} {'Nearest Existing (km)':>22}")
    print(f"  {'—'*4} {'—'*10} {'—'*11} {'—'*22}")

    for i, loc in enumerate(new_locations, 1):
        nearest_dist = min(haversine_distance(loc, ec) for ec in existing_chargers)
        print(f"  {i:<4} {loc.latitude:>10.4f} {loc.longitude:>11.4f} {nearest_dist:>22.2f}")

    # ── Coverage Statistics ──
    print_separator("COVERAGE STATISTICS")
    all_chargers = existing_chargers + new_locations
    stats = optimizer._calculate_coverage_efficiency(all_chargers)

    print(f"  Total chargers (existing + new): {len(all_chargers)}")
    print(f"  Total land area:    {stats['total_area']:.2f} km²")
    print(f"  Covered area:       {stats['covered_area']:.2f} km²")
    print(f"  Coverage:           {stats['coverage']:.2f}%")
    print(f"  Overlap area:       {stats['overlap_area']:.2f} km²")
    print(f"  Overlap:            {stats['overlap']:.2f}%")
    print(f"  Efficiency:         {stats['efficiency']:.2f}%")

    # ── Before vs After comparison ──
    print_separator("BEFORE vs AFTER COMPARISON")
    stats_before = optimizer._calculate_coverage_efficiency(existing_chargers)
    print(f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
    print(f"  {'—'*25} {'—'*12} {'—'*12} {'—'*12}")
    print(f"  {'Coverage %':<25} {stats_before['coverage']:>11.2f}% {stats['coverage']:>11.2f}% {stats['coverage']-stats_before['coverage']:>+11.2f}%")
    print(f"  {'Overlap %':<25} {stats_before['overlap']:>11.2f}% {stats['overlap']:>11.2f}% {stats['overlap']-stats_before['overlap']:>+11.2f}%")
    print(f"  {'Efficiency %':<25} {stats_before['efficiency']:>11.2f}% {stats['efficiency']:>11.2f}% {stats['efficiency']-stats_before['efficiency']:>+11.2f}%")
    print(f"  {'Covered area (km²)':<25} {stats_before['covered_area']:>12.2f} {stats['covered_area']:>12.2f} {stats['covered_area']-stats_before['covered_area']:>+12.2f}")

    # ── Baseline Comparison ──
    print_separator('BASELINE METHOD COMPARISON')
    grid_locations = generate_grid_baseline(optimizer, num_chargers)
    random_locations = generate_random_baseline(optimizer, num_chargers)

    baseline_stats = {
        'Existing Only': stats_before,
        'Grid Placement': optimizer._calculate_coverage_efficiency(existing_chargers + grid_locations),
        'Random Placement': optimizer._calculate_coverage_efficiency(existing_chargers + random_locations),
        'Optimized': stats
    }

    for name, baseline in baseline_stats.items():
        print(f"  {name:15s}: Coverage={baseline['coverage']:.2f}%, Overlap={baseline['overlap']:.2f}%, Efficiency={baseline['efficiency']:.2f}%")

    # ── Model Efficiency Metrics ──
    print_separator("NEURAL NETWORK MODEL EFFICIENCY")
    eff = optimizer.evaluate_model_efficiency(num_chargers=len(new_locations))
    print(f"  Composite Efficiency:      {eff.get('composite_efficiency', 0):.4f}")
    print(f"  Coverage Efficiency:       {eff.get('coverage_efficiency', 0):.2f}%")
    print(f"  Coverage Percentage:       {eff.get('coverage_percentage', 0):.2f}%")
    print(f"  Overlap Percentage:        {eff.get('overlap_percentage', 0):.2f}%")
    print(f"  Computational Efficiency:  {eff.get('computational_efficiency', 0):.4f}")
    print(f"  Optimization Time:         {eff.get('optimization_time', 0):.2f}s")
    print(f"  Prediction Consistency:    {eff.get('prediction_consistency', 0):.4f}")

    # ── Inter-charger distance matrix for new locations ──
    print_separator("INTER-CHARGER DISTANCES (NEW LOCATIONS, km)")
    n = len(new_locations)
    if n <= 15:
        header = f"  {'':>4}" + "".join(f" {j+1:>6}" for j in range(n))
        print(header)
        for i in range(n):
            row = f"  {i+1:>4}"
            for j in range(n):
                if i == j:
                    row += f" {'—':>6}"
                else:
                    d = haversine_distance(new_locations[i], new_locations[j])
                    row += f" {d:>6.1f}"
            print(row)
    else:
        print("  (Skipped — too many locations to display matrix)")

    # ── Save results ──
    print_separator("SAVING RESULTS")
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_data = {
        "timestamp": timestamp,
        "parameters": {
            "num_chargers": num_chargers,
            "coverage_radius": coverage_radius,
            "bounds": list(goa_bounds),
            "weights": weights,
        },
        "existing_chargers": [
            {"lat": loc.latitude, "lon": loc.longitude} for loc in existing_chargers
        ],
        "new_chargers": [
            {"lat": loc.latitude, "lon": loc.longitude} for loc in new_locations
        ],
        "coverage_stats": {
            "coverage_pct": stats['coverage'],
            "overlap_pct": stats['overlap'],
            "efficiency_pct": stats['efficiency'],
            "total_area_km2": stats['total_area'],
            "covered_area_km2": stats['covered_area'],
        },
        "model_efficiency": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in eff.items()},
        "timings": {
            "init_seconds": init_time,
            "optimization_seconds": opt_time,
        },
    }
    
    filepath = os.path.join(results_dir, f"optimization_result_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"  Results saved to: {filepath}")

    try:
        results_handler = ResultsHandler()
        figure_paths = results_handler.generate_publication_figures_from_data(
            existing_chargers,
            new_locations,
            result_data['parameters'],
            stats,
            weights,
            baseline_stats=baseline_stats,
            result_name=f"run_{timestamp}"
        )
        print("  Generated figures:")
        for path in figure_paths:
            print(f"    {path}")
    except Exception as e:
        print(f"  Warning: could not generate figures: {e}")

    print_separator("DONE")
    print(f"  Total runtime: {init_time + opt_time:.2f}s\n")

    return new_locations, stats, eff


if __name__ == "__main__":
    run_optimization()
