import folium
from folium import plugins
import numpy as np
from branca.colormap import LinearColormap
import webbrowser
import os
from models.location import Location, haversine_distance
from models.neural_optimizer import ChargerOptimizer
from typing import List, Tuple, Dict, Optional

class MapVisualizer:
    def __init__(self):
        self.map = None
        self.coverage_radius = 5.0  # 5km coverage radius

    def create_map(self, existing_chargers: List[Location], 
                   new_chargers: List[Location], 
                   area_bounds: Optional[Tuple[float, float, float, float]] = None,
                   coverage_radius: float = 5.0,
                   stats: Optional[Dict[str, float]] = None):
        """Create and save an interactive map visualization"""
        try:
            if not isinstance(coverage_radius, (int, float)):
                raise ValueError("Coverage radius must be numeric")
            self.coverage_radius = float(coverage_radius)

            # Choose a sensible default center from Goa
            center_lat = 15.4 if area_bounds is None else (area_bounds[0] + area_bounds[1]) / 2
            center_lon = 73.9 if area_bounds is None else (area_bounds[2] + area_bounds[3]) / 2

            self.map = folium.Map(location=[float(center_lat), float(center_lon)], 
                                  zoom_start=11,
                                  tiles='OpenStreetMap')

            if stats is None and area_bounds is not None:
                optimizer = ChargerOptimizer(
                    area_bounds=area_bounds,
                    existing_chargers=existing_chargers,
                    coverage_radius=self.coverage_radius
                )
                stats = optimizer._calculate_coverage_efficiency(existing_chargers + new_chargers)

            self._add_water_overlays()
            self._add_coverage_circles(existing_chargers, new_chargers)
            self._add_heatmap_layer(existing_chargers + new_chargers)
            self._add_charger_markers(existing_chargers, new_chargers)
            self._add_coastline_overlay()
            self._add_coverage_stats(stats)

            # Save and display the map
            map_path = os.path.join(os.path.dirname(__file__), 'charger_map.html')
            self.map.save(map_path)
            webbrowser.open('file://' + os.path.abspath(map_path))

        except Exception as e:
            print(f"Error creating or saving map: {str(e)}")
            raise

    def _add_coverage_circles(self, existing_chargers: List[Location], new_chargers: List[Location]):
        """Add coverage circles for chargers"""
        for charger in existing_chargers:
            folium.Circle(
                location=[float(charger.latitude), float(charger.longitude)],
                radius=int(self.coverage_radius * 1000),  # Convert to meters and ensure int
                color='red',
                fill=True,
                fill_opacity=0.1,
                opacity=0.5,
                popup='Existing Charger<br>Coverage: {}km'.format(self.coverage_radius)
            ).add_to(self.map)

        for i, charger in enumerate(new_chargers):
            overlaps = sum(1 for c in existing_chargers + new_chargers 
                          if c != charger and 
                          haversine_distance(c, charger) < self.coverage_radius * 2)
            
            folium.Circle(
                location=[float(charger.latitude), float(charger.longitude)],
                radius=int(self.coverage_radius * 1000),  # Convert to meters and ensure int
                color='green' if overlaps == 0 else 'orange',
                fill=True,
                fill_opacity=0.15,
                opacity=0.5,
                popup='New Charger #{}<br>Coverage: {}km'.format(i+1, self.coverage_radius)
            ).add_to(self.map)

    def _add_heatmap_layer(self, chargers: List[Location]):
        """Add heatmap visualization"""
        heat_data = []
        for charger in chargers:
            heat_data.append([
                float(charger.latitude), 
                float(charger.longitude), 
                1.0  # Fixed intensity for chargers
            ])
        
        if heat_data:
            plugins.HeatMap(
                data=heat_data,
                radius=20,
                blur=15,
                min_opacity=0.3,
                gradient={
                    "0.4": 'blue',
                    "0.65": 'lime',
                    "1.0": 'red'
                }
            ).add_to(self.map)

    def _add_coverage_stats(self, stats: Optional[Dict[str, float]] = None):
        """Add coverage statistics panel"""
        stats = stats or {}

        stats_html = """
        <div style="position: absolute; bottom: 50px; left: 10px; z-index: 1000; max-width: 300px;">
            <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid black; box-shadow: 2px 2px 8px rgba(0,0,0,0.2);">
                <h4 style="margin:0 0 5px 0">Coverage Statistics</h4>
                <div>Coverage: {:.1f}%</div>
                <div>Overlap: {:.1f}%</div>
                <div>Efficiency: {:.1f}%</div>
                <hr style="margin:6px 0;">
                <div>Total Area: {:.2f} km²</div>
                <div>Covered Area: {:.2f} km²</div>
                <div>Overlap Area: {:.2f} km²</div>
            </div>
        </div>
        """.format(
            stats.get('coverage', 0.0),
            stats.get('overlap', 0.0),
            stats.get('efficiency', 0.0),
            stats.get('total_area', 0.0),
            stats.get('covered_area', 0.0),
            stats.get('overlap_area', 0.0)
        )

        self.map.get_root().html.add_child(folium.Element(stats_html))

    def _add_water_overlays(self):
        """Overlay water bodies on the map for visual context"""
        water_polygons = [
            [(15.38, 73.85), (15.40, 73.87), (15.42, 73.89), (15.38, 73.91)],
            [(15.50, 73.80), (15.52, 73.82), (15.51, 73.85), (15.49, 73.83)],
            [(15.58, 73.75), (15.59, 73.77), (15.57, 73.79), (15.56, 73.77)],
        ]
        for polygon in water_polygons:
            folium.Polygon(
                locations=polygon,
                color='blue',
                fill=True,
                fill_opacity=0.15,
                weight=1,
                popup='Water body / river'
            ).add_to(self.map)

    def _add_coastline_overlay(self):
        """Overlay the Goa coastline on the map"""
        coastline = [
            (15.65, 73.75), (15.60, 73.71), (15.58, 73.73), (15.55, 73.75),
            (15.51, 73.77), (15.50, 73.77), (15.48, 73.80), (15.49, 73.82),
            (15.47, 73.84), (15.45, 73.85), (15.38, 73.88), (15.34, 73.90),
            (15.31, 73.91), (15.27, 73.92), (15.22, 73.94), (15.16, 73.96),
            (15.14, 73.96), (15.16, 73.98), (15.20, 74.05), (15.30, 74.12),
            (15.45, 74.15), (15.55, 74.12), (15.65, 74.05), (15.72, 73.95),
            (15.70, 73.85), (15.68, 73.78), (15.65, 73.75)
        ]
        folium.PolyLine(
            locations=coastline,
            color='navy',
            weight=2,
            opacity=0.7,
            popup='Goa coastline'
        ).add_to(self.map)

    def _add_charger_markers(self, existing_chargers: List[Location], 
                            new_chargers: List[Location]):
        """Add markers for chargers"""
        for charger in existing_chargers:
            folium.Marker(
                [float(charger.latitude), float(charger.longitude)],
                popup='Existing Charger',
                icon=folium.Icon(color='red', icon='plug', prefix='fa')
            ).add_to(self.map)

        for i, charger in enumerate(new_chargers, 1):
            folium.Marker(
                [float(charger.latitude), float(charger.longitude)],
                popup=f'New Charger {i}',
                icon=folium.Icon(color='green', icon='plug', prefix='fa')
            ).add_to(self.map)

    def _calculate_coverage_efficiency(self, chargers: List[Location]) -> dict:
        """Calculate coverage efficiency metrics"""
        # Placeholder implementation for coverage efficiency calculation
        coverage = 100.0  # Assume full coverage for simplicity
        overlap = 0.0  # Assume no overlap for simplicity
        # Updated efficiency calculation to penalize overlap more strongly
        efficiency = coverage / (1 + overlap / 100)
        
        return {
            'coverage': coverage,
            'overlap': overlap,
            'efficiency': efficiency,
            'total_area': 500.0,  # Placeholder value for total area
            'covered_area': 450.0,  # Placeholder value for covered area
            'overlap_area': 50.0  # Placeholder value for overlap area
        }
