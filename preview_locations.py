import json
import os
import webbrowser
from gui.map_view import MapVisualizer
from models.location import Location
from models.neural_optimizer import ChargerOptimizer


def load_latest_result(results_dir="results"):
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, results_dir)

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    result_files = [f for f in os.listdir(results_dir)
                    if f.startswith("optimization_result_") and f.endswith(".json")]
    if not result_files:
        raise FileNotFoundError("No optimization result JSON files found in the results directory.")

    result_files.sort(reverse=True)
    latest_file = result_files[0]
    with open(os.path.join(results_dir, latest_file), "r") as f:
        data = json.load(f)

    existing_chargers = [Location(item["lat"], item["lon"], True)
                         for item in data.get("existing_chargers", [])]
    new_chargers = [Location(item["lat"], item["lon"], False)
                   for item in data.get("new_chargers", [])]

    return existing_chargers, new_chargers, latest_file


def main():
    existing_chargers, new_chargers, filename = load_latest_result()

    print(f"Loaded latest result: {filename}")
    print(f"Existing chargers: {len(existing_chargers)}")
    print(f"New chargers: {len(new_chargers)}")

    optimizer = ChargerOptimizer(
        area_bounds=(15.14, 15.72, 73.71, 74.15),
        existing_chargers=existing_chargers,
        coverage_radius=5.0
    )

    invalid_locations = [loc for loc in existing_chargers + new_chargers
                         if not optimizer._is_point_in_goa(loc.latitude, loc.longitude, optimizer.coastline)]
    if invalid_locations:
        print("Warning: some saved charger locations fall outside land boundaries and will be omitted from the preview:")
        for loc in invalid_locations:
            print(f"  - {loc.latitude:.4f}, {loc.longitude:.4f}")
        existing_chargers = [loc for loc in existing_chargers
                             if optimizer._is_point_in_goa(loc.latitude, loc.longitude, optimizer.coastline)]
        new_chargers = [loc for loc in new_chargers
                        if optimizer._is_point_in_goa(loc.latitude, loc.longitude, optimizer.coastline)]

    stats = optimizer._calculate_coverage_efficiency(existing_chargers + new_chargers)
    visualizer = MapVisualizer()
    visualizer.create_map(
        existing_chargers,
        new_chargers,
        area_bounds=(15.14, 15.72, 73.71, 74.15),
        coverage_radius=5.0,
        stats=stats
    )

    base_dir = os.path.dirname(__file__)
    html_path = os.path.join(base_dir, "gui", "charger_map.html")
    if os.path.exists(html_path):
        print(f"HTML preview generated: {html_path}")
        webbrowser.open("file://" + os.path.abspath(html_path))
    else:
        print("Failed to generate HTML preview.")


if __name__ == "__main__":
    main()
