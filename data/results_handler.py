import json
from datetime import datetime
import os
from typing import List, Dict
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Circle
import numpy as np
from models.location import Location, haversine_distance

class ResultsHandler:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def save_optimization_result(self, 
                               new_locations: List[Location],
                               existing_locations: List[Location],
                               parameters: Dict) -> str:
        """Save optimization results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_result_{timestamp}.json"
        
        result_data = {
            "timestamp": timestamp,
            "parameters": parameters,
            "existing_chargers": [
                {"lat": loc.latitude, "lon": loc.longitude}
                for loc in existing_locations
            ],
            "new_chargers": [
                {"lat": loc.latitude, "lon": loc.longitude}
                for loc in new_locations
            ]
        }
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=4)
            
        return filepath

    def generate_publication_figures_from_data(self,
                                               existing_locations: List[Location],
                                               new_locations: List[Location],
                                               parameters: Dict,
                                               stats: Dict,
                                               weights: Dict,
                                               baseline_stats: Dict[str, Dict] = None,
                                               result_name: str = None) -> List[str]:
        """Generate publication-ready graphs from optimization result data."""
        if result_name is None:
            result_name = datetime.now().strftime("figures_%Y%m%d_%H%M%S")

        figure_dir = os.path.join(self.results_dir, 'figures', result_name)
        os.makedirs(figure_dir, exist_ok=True)

        figures = []
        bounds = parameters.get('bounds', None)
        coverage_radius = parameters.get('coverage_radius', 5.0)

        figures.append(self._plot_charger_layout(
            existing_locations,
            new_locations,
            bounds,
            coverage_radius,
            os.path.join(figure_dir, 'charger_layout.png')
        ))

        figures.append(self._plot_coverage_summary(
            stats,
            os.path.join(figure_dir, 'coverage_summary.png')
        ))

        if weights:
            figures.append(self._plot_weight_pie(
                weights,
                os.path.join(figure_dir, 'weight_distribution.png')
            ))

        if len(new_locations) > 1:
            figures.append(self._plot_distance_distribution(
                new_locations,
                os.path.join(figure_dir, 'distance_distribution.png')
            ))

        figures.append(self._plot_efficiency_overview(
            stats,
            os.path.join(figure_dir, 'efficiency_overview.png')
        ))

        if baseline_stats:
            figures.append(self._plot_method_comparison(
                baseline_stats,
                os.path.join(figure_dir, 'method_comparison.png')
            ))

        return figures

    def _plot_charger_layout(self,
                              existing_locations: List[Location],
                              new_locations: List[Location],
                              bounds: List[float],
                              coverage_radius: float,
                              output_path: str) -> str:
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()

        if bounds and len(bounds) == 4:
            ax.set_xlim(bounds[2], bounds[3])
            ax.set_ylim(bounds[0], bounds[1])

        ax.scatter(
            [loc.longitude for loc in existing_locations],
            [loc.latitude for loc in existing_locations],
            c='red', label='Existing chargers', s=60, edgecolors='black'
        )
        ax.scatter(
            [loc.longitude for loc in new_locations],
            [loc.latitude for loc in new_locations],
            c='green', label='New chargers', s=60, edgecolors='black'
        )

        # Draw approximate coverage radius in degrees
        radius_deg = coverage_radius / 111.0
        for loc in existing_locations + new_locations:
            patch = Circle((loc.longitude, loc.latitude), radius_deg,
                           edgecolor='blue', facecolor='none', linewidth=0.8, alpha=0.4)
            ax.add_patch(patch)

        ax.set_title('Charger Placement and Approximate Coverage')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)
        canvas = FigureCanvas(fig)
        canvas.print_figure(output_path, dpi=200, bbox_inches='tight')
        return output_path

    def _plot_coverage_summary(self, stats: Dict, output_path: str) -> str:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()

        categories = ['Coverage', 'Overlap', 'Efficiency']
        values = [stats.get('coverage', 0.0), stats.get('overlap', 0.0), stats.get('efficiency', 0.0)]
        bar_colors = ['#2c7fb8', '#f03b20', '#7fc97f']

        ax.bar(categories, values, color=bar_colors)
        ax.set_ylabel('Percentage')
        ax.set_title('Coverage, Overlap, and Efficiency')
        ax.set_ylim(0, max(100, max(values) * 1.1))
        for i, v in enumerate(values):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')

        canvas = FigureCanvas(fig)
        canvas.print_figure(output_path, dpi=200, bbox_inches='tight')
        return output_path

    def _plot_distance_distribution(self,
                                    new_locations: List[Location],
                                    output_path: str) -> str:
        distances = []
        for i, loc in enumerate(new_locations):
            for j in range(i + 1, len(new_locations)):
                distances.append(haversine_distance(loc, new_locations[j]))

        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        ax.hist(distances, bins=min(20, len(distances)), color='#66c2a5', edgecolor='black')
        ax.set_xlabel('Distance between new chargers (km)')
        ax.set_ylabel('Count')
        ax.set_title('New Charger Distance Distribution')
        canvas = FigureCanvas(fig)
        canvas.print_figure(output_path, dpi=200, bbox_inches='tight')
        return output_path

    def _plot_weight_pie(self, weights: Dict, output_path: str) -> str:
        labels = list(weights.keys())
        values = [max(0.0, float(weights.get(k, 0.0))) for k in labels]
        if sum(values) <= 0:
            values = [1.0 for _ in values]

        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
        ax.set_title('Feature Weight Distribution')
        canvas = FigureCanvas(fig)
        canvas.print_figure(output_path, dpi=200, bbox_inches='tight')
        return output_path

    def _plot_efficiency_overview(self, stats: Dict, output_path: str) -> str:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()

        categories = ['Coverage', 'Overlap', 'Efficiency']
        values = [stats.get('coverage', 0.0), stats.get('overlap', 0.0), stats.get('efficiency', 0.0)]
        ax.plot(categories, values, marker='o', linestyle='-', color='#1f78b4')
        ax.set_ylabel('Percentage')
        ax.set_title('Efficiency Overview')
        ax.grid(True, linestyle='--', alpha=0.4)
        canvas = FigureCanvas(fig)
        canvas.print_figure(output_path, dpi=200, bbox_inches='tight')
        return output_path

    def _plot_method_comparison(self, baseline_stats: Dict[str, Dict], output_path: str) -> str:
        methods = list(baseline_stats.keys())
        coverage = [baseline_stats[m].get('coverage', 0.0) for m in methods]
        overlap = [baseline_stats[m].get('overlap', 0.0) for m in methods]
        efficiency = [baseline_stats[m].get('efficiency', 0.0) for m in methods]

        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        x = np.arange(len(methods))
        width = 0.25

        ax.bar(x - width, coverage, width, label='Coverage %', color='#377eb8')
        ax.bar(x, overlap, width, label='Overlap %', color='#e41a1c')
        ax.bar(x + width, efficiency, width, label='Efficiency %', color='#4daf4a')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylabel('Percentage')
        ax.set_title('Method Comparison: Coverage, Overlap, and Efficiency')
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.95))
        ax.grid(True, linestyle='--', alpha=0.3)

        for i, v in enumerate(coverage):
            ax.text(i - width, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(overlap):
            ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(efficiency):
            ax.text(i + width, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

        canvas = FigureCanvas(fig)
        canvas.print_figure(output_path, dpi=200, bbox_inches='tight')
        return output_path

    def load_optimization_result(self, filename: str) -> Dict:
        """Load optimization results from a JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert data back to Location objects
        data['existing_chargers'] = [
            Location(c['lat'], c['lon'], True)
            for c in data['existing_chargers']
        ]
        data['new_chargers'] = [
            Location(c['lat'], c['lon'], False)
            for c in data['new_chargers']
        ]
        
        return data

    def list_results(self) -> List[str]:
        """List all available optimization results"""
        return [f for f in os.listdir(self.results_dir) 
                if f.startswith("optimization_result_")]

    def save_efficiency_metrics(self, filepath: str, efficiency_metrics: Dict):
        """Save efficiency metrics to the same JSON file as the optimization results"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data['efficiency_metrics'] = efficiency_metrics
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
