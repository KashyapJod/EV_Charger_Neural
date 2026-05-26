import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np
import json
import os
import sys
import time  # Add this import for timing
from typing import List, Tuple, Dict
from .location import Location, haversine_distance

class ChargerOptimizer:
    def __init__(self, area_bounds: Tuple[float, float, float, float], 
                 existing_chargers: List[Location],
                 coverage_radius: float = 5.0,
                 weights: Dict[str, float] = None):
        self.area_bounds = area_bounds
        self.existing_chargers = existing_chargers
        self.coverage_radius = coverage_radius
        
        # Initialize weights
        self.weights = weights or {
            'population_density': 0.15,
            'traffic_flow': 0.20,
            'points_of_interest': 0.10,
            'power_availability': 0.15,
            'charging_demand': 0.15,
            'accessibility': 0.10,
            'road_quality': 0.05,
            'revenue_potential': 0.10
        }
        
        # Define Goa coastline with more detailed points
        self.coastline = [
            (15.65, 73.75),  # Tiracol
            (15.60, 73.71),  # Arambol
            (15.58, 73.73),  # Mandrem
            (15.55, 73.75),  # Morjim
            (15.51, 73.77),  # Vagator
            (15.50, 73.77),  # Anjuna
            (15.48, 73.80),  # Panaji North
            (15.49, 73.82),  # Panaji
            (15.47, 73.84),  # Dona Paula
            (15.45, 73.85),  # Bambolim
            (15.38, 73.88),  # Zuari River
            (15.34, 73.90),  # Velsao
            (15.31, 73.91),  # Colva
            (15.27, 73.92),  # Benaulim
            (15.22, 73.94),  # Mobor
            (15.16, 73.96),  # Canacona
            (15.14, 73.96),  # Palolem
            (15.16, 73.98),  # Galgibaga
            (15.20, 74.05),  # Eastern Border
            (15.30, 74.12),  # Bhagwan Mahavir WLS
            (15.45, 74.15),  # Bondla
            (15.55, 74.12),  # Chorla
            (15.65, 74.05),  # Surla
            (15.72, 73.95),  # Northeast
            (15.70, 73.85),  # Pernem
            (15.68, 73.78),  # Northwest
            (15.65, 73.75),  # Back to Tiracol
        ]
        
        # Define water bodies to avoid
        self.water_bodies = [
            # Zuari River
            [(15.38, 73.85), (15.40, 73.87), (15.42, 73.89), (15.38, 73.91)],
            # Mandovi River
            [(15.50, 73.80), (15.52, 73.82), (15.51, 73.85), (15.49, 73.83)],
            # Chapora River
            [(15.58, 73.75), (15.59, 73.77), (15.57, 73.79), (15.56, 73.77)],
        ]
        
        # Load hotspots before using them
        try:
            self.hotspots = self._load_hotspots_data()
        except Exception as e:
            print(f"Error loading hotspots: {str(e)}")
            self.hotspots = self._generate_dummy_hotspots()
        
        self.min_distance_between_chargers = coverage_radius * 1.8  # Increased from 1.5 to reduce overlap
        self.candidate_cache = {}  # Add cache for candidate evaluations
        self.max_iterations = 100  # Add maximum iterations limit
        self.min_score_threshold = 0.1  # Add minimum score threshold
        
        # Initialize neural model
        try:
            self.model = self._build_model()
            print("Neural network model built successfully")
        except Exception as e:
            print(f"Error building neural network model: {str(e)}")
            # Fallback to a simple model if advanced one fails
            self.model = self._build_simple_model()
    
    def _load_hotspots_data(self) -> Dict:
        """Load hotspot data from JSON file"""
        try:
            filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'goa_data.json')
            if os.path.exists(filepath):
                print(f"Loading hotspots from {filepath}")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    return data.get('hotspots', {})
            else:
                print(f"Warning: Hotspot data file not found at {filepath}")
                return self._generate_dummy_hotspots()
        except Exception as e:
            print(f"Warning: Could not load hotspots data: {e}")
            return self._generate_dummy_hotspots()

    def _generate_dummy_hotspots(self) -> Dict:
        """Generate dummy hotspot data if real data isn't available"""
        print("Generating dummy hotspots")
        min_lat, max_lat, min_lon, max_lon = self.area_bounds
        
        # Create synthetic hotspots for testing
        tourist_spots = []
        for _ in range(5):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            tourist_spots.append({
                'lat': lat, 
                'lon': lon, 
                'weight': np.random.uniform(0.5, 1.0)
            })
            
        commercial_areas = []
        for _ in range(3):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            commercial_areas.append({
                'lat': lat, 
                'lon': lon, 
                'weight': np.random.uniform(0.7, 1.0)
            })
            
        return {
            'tourist_spots': tourist_spots,
            'commercial_areas': commercial_areas,
            'transport_hubs': [],
            'industrial_zones': []
        }

    def _build_simple_model(self):
        """Build a simple neural network model as fallback"""
        inputs = tf.keras.Input(shape=(6,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def _build_model(self):
        # Enhanced neural network architecture for hotspot analysis
        inputs = tf.keras.Input(shape=(6,))  # Updated for 6 features
        
        # Wider network for more feature processing
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01))(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def _evaluate_batch(self, batch: List[Location], current_optimal: List[Location]) -> np.ndarray:
        """Evaluate a batch of candidates efficiently"""
        if not batch:
            return np.array([])
            
        scores = np.zeros(len(batch))
        
        # Quick distance check first
        for i, candidate in enumerate(batch):
            if any(not self._quick_distance_check(candidate, existing) 
                  for existing in self.existing_chargers + current_optimal):
                continue
                
            # Calculate score only for valid candidates
            cache_key = f"{candidate.latitude:.6f},{candidate.longitude:.6f}"
            if cache_key in self.candidate_cache:
                scores[i] = self.candidate_cache[cache_key]
            else:
                score = self._evaluate_location(candidate, current_optimal)
                self.candidate_cache[cache_key] = score
                scores[i] = score
                
        return scores

    def _quick_distance_check(self, loc1: Location, loc2: Location) -> bool:
        """Quick approximate distance check before using haversine"""
        lat_diff = abs(loc1.latitude - loc2.latitude) * 111  # km per degree
        lon_diff = abs(loc1.longitude - loc2.longitude) * 111 * np.cos(np.radians(loc1.latitude))
        quick_dist = np.sqrt(lat_diff**2 + lon_diff**2)
        return quick_dist >= self.min_distance_between_chargers

    def _calculate_distance_score(self, location: Location, existing_locations: List[Location]) -> float:
        """Calculate score based on distance distribution to other chargers"""
        if not existing_locations:
            return 1.0

        distances = [haversine_distance(location, existing) for existing in existing_locations]
        if not distances:
            return 1.0
            
        min_dist = min(distances)
        
        # Prefer locations that maintain good spacing
        if min_dist < self.min_distance_between_chargers:
            return 0.0
        elif min_dist > 2 * self.coverage_radius:
            return 1.0
        else:
            # Linear score between minimum and optimal distance
            return (min_dist - self.min_distance_between_chargers) / (self.coverage_radius * 0.5)

    def _calculate_hotspot_score(self, location: Location) -> float:
        """Calculate location score based on proximity to hotspots"""
        if not self.hotspots:
            return 0.5  # Default score if no hotspot data

        total_score = 0
        total_weight = 0

        for category, spots in self.hotspots.items():
            category_score = 0
            category_weight = {
                'tourist_spots': 0.3,
                'commercial_areas': 0.3,
                'transport_hubs': 0.25,
                'industrial_zones': 0.15
            }.get(category, 0.25)

            for spot in spots:
                spot_loc = Location(spot['lat'], spot['lon'])
                distance = haversine_distance(location, spot_loc)
                # Convert distance to score (closer is better)
                distance_score = 1.0 / (1.0 + distance)
                weight = spot.get('weight', 1.0)
                category_score += distance_score * weight

            total_score += category_score * category_weight
            total_weight += category_weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _calculate_demand_score(self, location: Location) -> float:
        """Calculate potential demand based on surrounding hotspots"""
        total_demand = 0
        
        # Consider all types of hotspots for demand calculation
        for category, spots in self.hotspots.items():
            for spot in spots:
                spot_loc = Location(spot['lat'], spot['lon'])
                distance = haversine_distance(location, spot_loc)
                
                # Demand decreases with distance but is weighted by spot importance
                weight = spot.get('weight', 1.0)
                demand = weight * np.exp(-distance / 5.0)  # 5km decay factor
                total_demand += demand
        
        # Normalize demand score
        return min(1.0, total_demand / 10.0)  # Cap at 1.0

    def _prepare_input_data(self, candidates: List[Location]) -> np.ndarray:
        """Prepare input data for neural network"""
        if not candidates:
            return np.array([])
            
        input_data = []
        min_lat, max_lat, min_lon, max_lon = self.area_bounds
        
        for location in candidates:
            # Basic features
            norm_lat = (location.latitude - min_lat) / (max_lat - min_lat)
            norm_lon = (location.longitude - min_lon) / (max_lon - min_lon)
            
            if self.existing_chargers:
                min_dist = min(haversine_distance(location, ec) for ec in self.existing_chargers)
                norm_dist = min_dist / 50.0
            else:
                norm_dist = 1.0
            
            hotspot_score = self._calculate_hotspot_score(location)
            demand_score = self._calculate_demand_score(location)
            
            # Ensure the feature vector has exactly 6 features
            features = [
                norm_lat, norm_lon,
                norm_dist,
                hotspot_score,
                demand_score,
                0.0  # Placeholder for the 6th feature if missing
            ]
            input_data.append(features)
        
        return np.array(input_data)

    def _score_candidate(self, location: Location, current_optimal: List[Location]) -> float:
        """Score a candidate location using a lightweight proxy function."""
        if not self._is_point_in_goa(location.latitude, location.longitude, self.coastline):
            return 0.0

        combined_locations = self.existing_chargers + list(current_optimal)
        distance_score = self._calculate_distance_score(location, combined_locations)
        hotspot_score = self._calculate_hotspot_score(location)
        demand_score = self._calculate_demand_score(location)

        return (
            0.35 * distance_score +
            0.35 * hotspot_score +
            0.30 * demand_score
        )

    def generate_optimal_locations(self, num_new_chargers: int) -> List[Location]:
        """Generate optimal locations for new chargers with minimum overlap and maximum coverage."""
        print(f"Generating candidate locations for {num_new_chargers} chargers...")
        candidates = self._generate_candidates(300)
        candidates = [c for c in candidates if self._is_point_in_goa(c.latitude, c.longitude, self.coastline)]

        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, [])
            if score > self.min_score_threshold:
                scored_candidates.append((candidate, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[: max(num_new_chargers * 5, 50)]

        selected_candidates = []
        for candidate, score in top_candidates:
            if any(haversine_distance(candidate, existing) < self.min_distance_between_chargers
                   for existing in self.existing_chargers + selected_candidates):
                continue
            selected_candidates.append(candidate)
            if len(selected_candidates) >= num_new_chargers:
                break

        if len(selected_candidates) < num_new_chargers:
            for candidate, score in scored_candidates:
                if candidate in selected_candidates:
                    continue
                if any(haversine_distance(candidate, existing) < self.min_distance_between_chargers
                       for existing in self.existing_chargers + selected_candidates):
                    continue
                selected_candidates.append(candidate)
                if len(selected_candidates) >= num_new_chargers:
                    break

        print(f"Selected {len(selected_candidates)} candidate locations for optimization.")
        return selected_candidates

    def _calculate_coverage_efficiency(self, chargers: List[Location]) -> dict:
        """Calculate coverage efficiency using area-based metrics"""
        total_area = self._calculate_land_area()
        chargers = [c for c in chargers if self._is_point_in_goa(c.latitude, c.longitude, self.coastline)]

        covered_area, overlap_area = self._estimate_union_coverage(chargers)
        coverage_percentage = min(100, (covered_area / total_area) * 100) if total_area > 0 else 0
        overlap_percentage = (overlap_area / total_area) * 100 if total_area > 0 else 0
        efficiency = coverage_percentage * (1 - min(overlap_percentage / 100, 1))

        return {
            'coverage': coverage_percentage,
            'overlap': overlap_percentage,
            'efficiency': efficiency,
            'total_area': total_area,
            'covered_area': covered_area,
            'overlap_area': overlap_area
        }

    def _calculate_land_area(self) -> float:
        """Calculate total land area excluding water bodies"""
        # Calculate area of Goa polygon
        total_area = self._calculate_polygon_area(self.coastline)
        
        # Subtract water body areas
        for water_body in self.water_bodies:
            water_area = self._calculate_polygon_area(water_body)
            total_area -= water_area
        
        return total_area

    def _estimate_union_coverage(self, chargers: List[Location], samples: int = 12000) -> tuple[float, float]:
        """Estimate union coverage and overlap area using sampling over Goa land."""
        total_area = self._calculate_land_area()
        if total_area <= 0 or not chargers:
            return 0.0, 0.0

        min_lat, max_lat, min_lon, max_lon = self.area_bounds
        land_points = 0
        covered_points = 0
        overlap_points = 0

        for _ in range(samples):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            if not self._is_point_in_goa(lat, lon, self.coastline):
                continue

            land_points += 1
            count = sum(1 for charger in chargers if haversine_distance(Location(lat, lon), charger) <= self.coverage_radius)
            if count >= 1:
                covered_points += 1
            if count >= 2:
                overlap_points += 1

        if land_points == 0:
            return 0.0, 0.0

        covered_area = (covered_points / land_points) * total_area
        overlap_area = (overlap_points / land_points) * total_area
        return covered_area, overlap_area

    def _calculate_polygon_area(self, points: List[tuple]) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(points) < 3:
            return 0
            
        # Convert lat/lon to approximate kilometers (rough approximation)
        km_points = []
        for lat, lon in points:
            y = lat * 111  # 1 degree latitude ≈ 111 km
            x = lon * 111 * np.cos(np.radians(lat))  # Adjust longitude for latitude
            km_points.append((x, y))
        
        area = 0
        j = len(km_points) - 1
        
        for i in range(len(km_points)):
            area += (km_points[j][0] + km_points[i][0]) * (km_points[j][1] - km_points[i][1])
            j = i
            
        return abs(area) / 2

    def _calculate_effective_coverage_area(self, charger: Location) -> float:
        """Calculate effective coverage area considering geographic constraints"""
        # Base circle area
        base_area = np.pi * (self.coverage_radius ** 2)
        
        # Calculate intersection with water bodies and boundaries
        effective_area = base_area
        
        # Reduce area if circle intersects with water bodies
        for water_body in self.water_bodies:
            intersection = self._calculate_circle_polygon_intersection(
                charger, self.coverage_radius, water_body)
            effective_area -= intersection
        
        # Ensure area doesn't exceed boundaries
        boundary_intersection = self._calculate_circle_boundary_intersection(
            charger, self.coverage_radius)
        effective_area *= boundary_intersection
        
        return effective_area

    def _calculate_circle_overlap(self, center1: Location, center2: Location, radius: float) -> float:
        """Calculate overlap area between two circles"""
        d = haversine_distance(center1, center2)
        
        # No overlap
        if d >= 2 * radius:
            return 0
            
        # Complete overlap
        if d <= 0:
            return np.pi * (radius ** 2)
            
        # Partial overlap
        r = radius
        area = 2 * (r**2 * np.arccos(d/(2*r)) - (d/2) * np.sqrt(r**2 - (d/2)**2))
        return area

    def _calculate_circle_polygon_intersection(self, center: Location, radius: float, 
                                            polygon: List[tuple]) -> float:
        """Approximate intersection area between circle and polygon"""
        # Simplified approximation using grid sampling
        points = 100
        count = 0
        total = 0
        
        min_lat = min(p[0] for p in polygon) - radius/111
        max_lat = max(p[0] for p in polygon) + radius/111
        min_lon = min(p[1] for p in polygon) - radius/(111 * np.cos(np.radians(center.latitude)))
        max_lon = max(p[1] for p in polygon) + radius/(111 * np.cos(np.radians(center.latitude)))
        
        for _ in range(points):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            point = Location(lat, lon)
            
            if (haversine_distance(center, point) <= radius and 
                self._is_point_in_polygon(lat, lon, polygon)):
                count += 1
            total += 1
        
        return (count / total) * np.pi * (radius ** 2) if total > 0 else 0

    def _calculate_circle_boundary_intersection(self, center: Location, radius: float) -> float:
        """Calculate what fraction of circle lies within boundaries"""
        points = 100
        count = 0
        
        for _ in range(points):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            
            # Convert polar to lat/lon
            lat = center.latitude + (r/111) * np.cos(angle)
            lon = center.longitude + (r/(111 * np.cos(np.radians(center.latitude)))) * np.sin(angle)
            
            if self._is_point_in_goa(lat, lon, self.coastline):
                count += 1
                
        return count / points if points > 0 else 0

    def _generate_candidates(self, num_candidates: int) -> List[Location]:
        """Generate candidate locations using a grid-based approach for better coverage"""
        min_lat, max_lat, min_lon, max_lon = self.area_bounds
        candidates = []
        
        # Create a more strategic grid of candidates instead of random points
        # This ensures better distribution across the entire region
        grid_size = int(np.sqrt(num_candidates * 3))  # Create more candidates than needed for better selection
        
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        # Generate candidates in a grid pattern for better coverage
        for i in range(grid_size):
            for j in range(grid_size):
                lat = min_lat + lat_step * (i + 0.5)  # Center within each grid cell
                lon = min_lon + lon_step * (j + 0.5)
                
                # Add small random noise to avoid perfectly aligned grid
                lat += np.random.uniform(-0.3, 0.3) * lat_step
                lon += np.random.uniform(-0.3, 0.3) * lon_step
                
                # Check if point is within Goa's boundaries
                if self._is_point_in_goa(lat, lon, self.coastline):
                    candidates.append(Location(latitude=lat, longitude=lon))
        
        # If we don't have enough candidates, add some random points
        max_attempts = 100
        attempts = 0
        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            if self._is_point_in_goa(lat, lon, self.coastline):
                candidates.append(Location(latitude=lat, longitude=lon))
        
        print(f"Generated {len(candidates)} candidate locations")
        return candidates

    def _is_point_in_goa(self, lat: float, lon: float, coastline: List[tuple]) -> bool:
        """Check if point is within Goa and not in water bodies"""
        if not self._is_point_in_polygon(lat, lon, coastline):
            return False
            
        # Check if point is in any water body
        for water_body in self.water_bodies:
            if self._is_point_in_polygon(lat, lon, water_body):
                return False
                
        return True

    def _is_point_in_polygon(self, lat: float, lon: float, polygon: List[tuple]) -> bool:
        """Ray casting algorithm for point in polygon test"""
        inside = False
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            if ((polygon[i][1] > lon) != (polygon[j][1] > lon) and
                lat < (polygon[j][0] - polygon[i][0]) * (lon - polygon[i][1]) /
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
            
        return inside

    def _evaluate_location(self, location: Location, current_optimal: List[Location]) -> float:
        """Evaluate candidate location based on overall network coverage efficiency.
        This method simulates the addition of the candidate to the existing and already selected chargers,
        then returns the computed coverage efficiency (maximized for best overall coverage).
        """
        if not self._is_point_in_goa(location.latitude, location.longitude, self.coastline):
            return 0.0

        if current_optimal and isinstance(current_optimal[0], tuple):
            optimal_locations = [loc for loc, _ in current_optimal]
        else:
            optimal_locations = list(current_optimal)

        combined_chargers = self.existing_chargers + optimal_locations + [location]
        stats = self._calculate_coverage_efficiency(combined_chargers)
        return stats['efficiency']

    def evaluate_model_efficiency(self, num_chargers: int = 5) -> Dict:
        """Calculate various efficiency metrics for the model and optimization process"""
        start_time = time.time()
        
        try:
            # Generate test candidates
            candidates = self._generate_candidates(100)
            
            # Calculate coverage metrics
            all_chargers = self.existing_chargers + candidates[:num_chargers]
            coverage_stats = self._calculate_coverage_efficiency(all_chargers)
            
            # Calculate neural network prediction consistency
            if len(candidates) > 0:
                input_data = self._prepare_input_data(candidates[:10])  # Test with first 10 candidates
                if input_data.size > 0:
                    predictions = self.model.predict(input_data, verbose=0)
                    if predictions.shape[0] > 1:
                        corr = np.corrcoef(predictions[:, 0], predictions[:, 1])[0, 1]
                        prediction_consistency = float(np.nan_to_num(corr, nan=0.0))
                    else:
                        prediction_consistency = 0.0
                else:
                    prediction_consistency = 0.0
            else:
                prediction_consistency = 0.0
            
            # Calculate processing time
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Calculate computational efficiency (normalized score)
            comp_efficiency = 1.0 / (1.0 + (optimization_time / num_chargers))
            
            # Calculate composite efficiency score with weighted components
            composite_efficiency = (
                0.5 * (coverage_stats['efficiency'] / 100) +         # Coverage efficiency (50%)
                0.3 * comp_efficiency +                              # Computational efficiency (30%)
                0.2 * (1.0 - abs(prediction_consistency))           # Neural network diversity (20%)
            )
            
            return {
                'composite_efficiency': composite_efficiency,
                'coverage_efficiency': coverage_stats['efficiency'],
                'coverage_percentage': coverage_stats['coverage'],
                'overlap_percentage': coverage_stats['overlap'],
                'computational_efficiency': comp_efficiency,
                'optimization_time': optimization_time,
                'prediction_consistency': prediction_consistency
            }
            
        except Exception as e:
            print(f"Error in evaluate_model_efficiency: {str(e)}")
            # Return a basic error result
            return {
                'composite_efficiency': 0.0,
                'error': str(e),
                'coverage_efficiency': 0.0,
                'coverage_percentage': 0.0,
                'overlap_percentage': 0.0,
                'computational_efficiency': 0.0,
                'optimization_time': 0.0,
                'prediction_consistency': 0.0
            }

