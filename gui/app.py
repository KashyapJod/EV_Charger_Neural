import customtkinter as ctk
ctk.set_appearance_mode("Light")
import threading
import queue
import time
import numpy as np  # Add NumPy import for np.argmax
import os  # Add os import for environment variable
from typing import List, Callable
from models.neural_optimizer import ChargerOptimizer
from models.location import Location, haversine_distance
from .map_view import MapVisualizer
from data.results_handler import ResultsHandler

class EVChargerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("EV Charger Placement Optimizer")
        self.geometry("1000x800")
        self.minsize(800, 600)
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Setup threading components
        self.queue = queue.Queue()
        self.is_processing = False

        # More precise Goa bounds
        self.goa_bounds = (15.14, 15.72, 73.71, 74.15)

        # Updated strategic locations avoiding water
        self.existing_chargers = [
            Location(15.62, 73.81, True),  # Pernem (adjusted)
            Location(15.18, 73.95, True),  # Canacona (adjusted)
            Location(15.38, 74.10, True),  # Dharbandora (adjusted)
            Location(15.49, 73.79, True),  # Bardez (adjusted)
            Location(15.40, 73.89, True),  # Ponda (adjusted)
        ]
        self.selected_existing_chargers = self.existing_chargers.copy()

        self.new_locations = []
        self.optimizer = None
        
        # Setup UI components
        self.setup_ui()
        
        # Initialize resources
        self.map_visualizer = MapVisualizer()
        self.results_handler = ResultsHandler()
        
        # Start the queue processing loop
        self.after(100, self.process_queue)

    def setup_ui(self):
        # Create tabview for organization
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create tabs
        self.tabview.add("Optimization")
        self.tabview.add("Results")
        self.tabview.add("Settings")
        
        # Configure tab grid layout
        for tab in ["Optimization", "Results", "Settings"]:
            self.tabview.tab(tab).grid_columnconfigure(0, weight=1)
            self.tabview.tab(tab).grid_rowconfigure(1, weight=1)
        
        # === Optimization Tab ===
        self.setup_optimization_tab()
        
        # === Results Tab ===
        self.setup_results_tab()
        
        # === Settings Tab ===
        self.setup_settings_tab()
        
        # Status bar
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready")
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        self.progress_bar.set(0)
        
    def setup_optimization_tab(self):
        opt_tab = self.tabview.tab("Optimization")
        
        # Input frame
        input_frame = ctk.CTkFrame(opt_tab)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Number of chargers
        ctk.CTkLabel(input_frame, text="Number of new chargers:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.num_chargers = ctk.CTkEntry(input_frame, width=100)
        self.num_chargers.insert(0, "5")
        self.num_chargers.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Coverage radius
        ctk.CTkLabel(input_frame, text="Coverage radius (km):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.coverage_radius = ctk.CTkEntry(input_frame, width=100)
        self.coverage_radius.insert(0, "5.0")
        self.coverage_radius.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Existing charger selection
        ctk.CTkLabel(input_frame, text="Existing chargers to include:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.existing_frame = ctk.CTkFrame(input_frame)
        self.existing_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.existing_frame.grid_columnconfigure(0, weight=1)
        self.existing_charger_vars = []

        for i, charger in enumerate(self.existing_chargers):
            var = ctk.IntVar(value=1)
            checkbox = ctk.CTkCheckBox(
                self.existing_frame,
                text=f"{i+1}. {charger.latitude:.4f}, {charger.longitude:.4f}",
                variable=var,
                command=self.update_existing_selection_count
            )
            checkbox.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="w")
            self.existing_charger_vars.append((var, charger))

        self.selected_existing_label = ctk.CTkLabel(input_frame, text=f"Selected existing chargers: {len(self.existing_chargers)}")
        self.selected_existing_label.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="w")

        # Advanced options (expand on click)
        self.advanced_frame = ctk.CTkFrame(opt_tab)
        self.advanced_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.advanced_frame.grid_columnconfigure(0, weight=1)
        
        self.advanced_checkbox = ctk.CTkCheckBox(input_frame, text="Show Advanced Options", 
                                                command=self.toggle_advanced_options)
        self.advanced_checkbox.grid(row=4, column=1, padx=10, pady=(0, 10), sticky="w")
        self.advanced_frame.grid_remove()  # Hidden by default
        
        # Advanced options content
        self.weights = {}
        weights_list = [
            ('population_density', 'Population Density:', 0.20),
            ('traffic_flow', 'Traffic Flow:', 0.25),
            ('points_of_interest', 'Points of Interest:', 0.15),
            ('power_availability', 'Power Availability:', 0.15),
            ('charging_demand', 'Charging Demand:', 0.10),
            ('accessibility', 'Accessibility:', 0.10),
            ('road_quality', 'Road Quality:', 0.03),
            ('revenue_potential', 'Revenue Potential:', 0.02)
        ]
        
        for i, (key, label, default) in enumerate(weights_list):
            row = i % 4
            col = (i // 4) * 2
            
            ctk.CTkLabel(self.advanced_frame, text=label).grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            weight_entry = ctk.CTkEntry(self.advanced_frame, width=70)
            weight_entry.insert(0, str(default))
            weight_entry.grid(row=row, column=col+1, padx=10, pady=5, sticky="w")
            self.weights[key] = weight_entry
        
        # Button frame
        button_frame = ctk.CTkFrame(opt_tab)
        button_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        self.calculate_btn = ctk.CTkButton(button_frame, text="Calculate Optimal Locations", 
                                         command=self.calculate_locations)
        self.calculate_btn.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        
        self.stop_btn = ctk.CTkButton(button_frame, text="Stop Calculation", fg_color="darkred", 
                                     command=self.stop_calculation, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Log frame
        log_frame = ctk.CTkFrame(opt_tab)
        log_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(log_frame, text="Optimization Log:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.log_text = ctk.CTkTextbox(log_frame)
        self.log_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
    def setup_results_tab(self):
        result_tab = self.tabview.tab("Results")
        
        # Results frame
        self.result_frame = ctk.CTkFrame(result_tab)
        self.result_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(self.result_frame, text="Optimization Results:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.result_text = ctk.CTkTextbox(self.result_frame)
        self.result_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Action buttons frame
        action_frame = ctk.CTkFrame(result_tab)
        action_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.show_map_btn = ctk.CTkButton(action_frame, text="Show Map", 
                                        command=self.show_map, state="disabled")
        self.show_map_btn.pack(side="left", padx=10, pady=10)
        
        self.save_btn = ctk.CTkButton(action_frame, text="Save Results", 
                                     command=self.save_results, state="disabled")
        self.save_btn.pack(side="left", padx=10, pady=10)
        
        self.load_btn = ctk.CTkButton(action_frame, text="Load Results", 
                                     command=self.load_results)
        self.load_btn.pack(side="left", padx=10, pady=10)
        
        self.efficiency_btn = ctk.CTkButton(action_frame, text="Calculate Efficiency", 
                                          command=self.calculate_efficiency, state="disabled")
        self.efficiency_btn.pack(side="left", padx=10, pady=10)
        
        self.report_type = ctk.StringVar(value="Full Report")
        report_options = ["Summary", "Coverage Analysis", "Distance Matrix", "Weights Overview", "Efficiency Metrics", "Full Report"]
        self.report_menu = ctk.CTkOptionMenu(action_frame, values=report_options, variable=self.report_type)
        self.report_menu.pack(side="left", padx=10, pady=10)
        self.report_menu.set("Full Report")

        self.generate_report_btn = ctk.CTkButton(action_frame, text="Generate Report", 
                                                 command=self.generate_report, state="disabled")
        self.generate_report_btn.pack(side="left", padx=10, pady=10)

        self.generate_figures_btn = ctk.CTkButton(action_frame, text="Generate Figures", 
                                                 command=self.generate_figures, state="disabled")
        self.generate_figures_btn.pack(side="left", padx=10, pady=10)
        
    def setup_settings_tab(self):
        settings_tab = self.tabview.tab("Settings")
        
        # Settings frame
        settings_frame = ctk.CTkFrame(settings_tab)
        settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Theme setting
        ctk.CTkLabel(settings_frame, text="UI Theme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        theme_options = ctk.CTkOptionMenu(settings_frame, values=["White", "Dark", "System"], 
                                         command=self.change_theme)
        theme_options.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # UI scaling
        ctk.CTkLabel(settings_frame, text="UI Scaling:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        scaling_options = ctk.CTkOptionMenu(settings_frame, values=["80%", "90%", "100%", "110%", "120%"], 
                                           command=self.change_scaling)
        scaling_options.set("100%")
        scaling_options.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Cache settings
        ctk.CTkLabel(settings_frame, text="Clear Calculation Cache:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        clear_cache_btn = ctk.CTkButton(settings_frame, text="Clear Cache", 
                                       command=self.clear_cache)
        clear_cache_btn.grid(row=2, column=1, padx=10, pady=10, sticky="w")

    def toggle_advanced_options(self):
        if self.advanced_checkbox.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
            
    def get_selected_existing_chargers(self):
        """Return the list of existing chargers selected by the user."""
        return [charger for var, charger in self.existing_charger_vars if var.get() == 1]

    def update_existing_selection_count(self):
        selected_count = len(self.get_selected_existing_chargers())
        self.selected_existing_label.configure(text=f"Selected existing chargers: {selected_count}")

    def change_theme(self, theme):
        if theme == "White":
            theme = "Light"
        ctk.set_appearance_mode(theme)
        
    def change_scaling(self, scaling):
        new_scaling = int(scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling)
        
    def clear_cache(self):
        if self.optimizer:
            self.optimizer.grid_cache = {}
            self.optimizer.candidate_cache = {}
            self.log_message("Cache cleared successfully.")
            
    def process_queue(self):
        """Process messages from the queue (from background threads)"""
        try:
            while True:
                message = self.queue.get_nowait()
                message_type = message.get("type", "log")
                
                if message_type == "log":
                    self.log_message(message.get("text", ""))
                elif message_type == "progress":
                    self.update_progress(message.get("value", 0))
                elif message_type == "status":
                    self.status_label.configure(text=message.get("text", ""))
                elif message_type == "complete":
                    self.is_processing = False
                    self.new_locations = message.get("locations", [])
                    self.display_results()
                    self.calculate_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.show_map_btn.configure(state="normal")
                    self.save_btn.configure(state="normal")
                    self.efficiency_btn.configure(state="normal")
                    if hasattr(self, 'generate_report_btn'):
                        self.generate_report_btn.configure(state="normal")
                    if hasattr(self, 'generate_figures_btn'):
                        self.generate_figures_btn.configure(state="normal")
                    self.update_progress(1.0)  # 100% complete
                    self.status_label.configure(text="Optimization complete")
                elif message_type == "error":
                    self.is_processing = False
                    self.log_message(f"Error: {message.get('text', 'Unknown error')}")
                    self.status_label.configure(text="Error occurred")
                    self.calculate_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    
                self.queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next queue check
        self.after(100, self.process_queue)
        
    def update_progress(self, value):
        """Update progress bar with value between 0.0 and 1.0"""
        self.progress_bar.set(value)
        
    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        
    def calculate_locations(self):
        try:
            if self.is_processing:
                return
                
            self.is_processing = True
            self.calculate_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.status_label.configure(text="Initializing optimization...")
            self.update_progress(0.0)

            # Get input values
            try:
                num_chargers = int(self.num_chargers.get())
                if num_chargers < 1:
                    raise ValueError("Number of new chargers must be at least 1")
                
                coverage_radius = float(self.coverage_radius.get())
                
                # Get weights from advanced options
                weights = {}
                if self.advanced_checkbox.get():
                    for key, entry in self.weights.items():
                        try:
                            weights[key] = float(entry.get())
                        except ValueError:
                            self.queue.put({"type": "error", "text": f"Invalid weight value for {key}"})
                            return
                else:
                    weights = {
                        'population_density': 0.20,
                        'traffic_flow': 0.25,
                        'points_of_interest': 0.15,
                        'power_availability': 0.15,
                        'charging_demand': 0.10,
                        'accessibility': 0.10,
                        'road_quality': 0.03,
                        'revenue_potential': 0.02
                    }
                    
            except ValueError:
                self.queue.put({"type": "error", "text": "Invalid input: Please enter valid numbers"})
                return

            selected_existing = self.get_selected_existing_chargers()
            self.selected_existing_chargers = selected_existing

            # Start optimization in background thread
            threading.Thread(
                target=self.run_optimization_thread,
                args=(num_chargers, coverage_radius, weights, selected_existing),
                daemon=True
            ).start()
            
        except Exception as e:
            self.queue.put({"type": "error", "text": str(e)})
            
    def run_optimization_thread(self, num_chargers, coverage_radius, weights, existing_chargers):
        """Run the optimization process in a background thread"""
        try:
            self.queue.put({"type": "log", "text": f"Starting optimization for {num_chargers} chargers with {coverage_radius}km coverage radius"})
            self.queue.put({"type": "status", "text": "Creating optimizer..."})
            
            # Initialize optimizer with proper error handling
            try:
                self.optimizer = ChargerOptimizer(
                    area_bounds=self.goa_bounds,
                    existing_chargers=existing_chargers,
                    coverage_radius=coverage_radius,
                    weights=weights
                )
            except Exception as e:
                self.queue.put({"type": "error", "text": f"Error creating optimizer: {str(e)}"})
                return
                
            # Verify that generate_optimal_locations method exists
            if not hasattr(self.optimizer, 'generate_optimal_locations'):
                # If missing, add a simple version
                self.queue.put({"type": "log", "text": "Adding generate_optimal_locations method to optimizer"})
                
                def generate_optimal_locations(optimizer, num_new_chargers: int) -> List[Location]:
                    """Fallback optimization method"""
                    print("Using fallback generate_optimal_locations method")
                    candidates = optimizer._generate_candidates(300)
                    optimal_locations = []
                    
                    # Generate grid points
                    grid_points = optimizer._generate_grid_points(100)
                    
                    # Track covered points
                    covered_grid_points = set()
                    for existing in optimizer.existing_chargers:
                        for i, point in enumerate(grid_points):
                            if haversine_distance(point, existing) <= optimizer.coverage_radius:
                                covered_grid_points.add(i)
                    
                    # Find locations iteratively
                    for _ in range(min(num_new_chargers, len(candidates))):
                        best_score = -1
                        best_candidate = None
                        
                        for candidate in candidates:
                            # Skip if too close to existing chargers or already selected
                            if any(haversine_distance(candidate, existing) < optimizer.min_distance_between_chargers 
                                  for existing in optimizer.existing_chargers + optimal_locations):
                                continue
                            
                            # Count newly covered points
                            newly_covered = sum(1 for i, point in enumerate(grid_points) 
                                              if i not in covered_grid_points 
                                              and haversine_distance(point, candidate) <= optimizer.coverage_radius)
                            
                            # Simple score based on coverage
                            score = newly_covered / len(grid_points)
                            
                            if score > best_score:
                                best_score = score
                                best_candidate = candidate
                        
                        # If found a good candidate, add it
                        if best_candidate:
                            optimal_locations.append(best_candidate)
                            
                            # Update covered points
                            for i, point in enumerate(grid_points):
                                if haversine_distance(point, best_candidate) <= optimizer.coverage_radius:
                                    covered_grid_points.add(i)
                            
                            # Filter remaining candidates
                            candidates = [c for c in candidates 
                                         if haversine_distance(c, best_candidate) >= optimizer.min_distance_between_chargers]
                    
                    return optimal_locations
                    
                # Bind the method to the optimizer instance
                import types
                self.optimizer.generate_optimal_locations = types.MethodType(
                    generate_optimal_locations, self.optimizer)
                    
            # Run the optimization
            self.queue.put({"type": "status", "text": "Generating candidates..."})
            new_locations = self.optimizer.generate_optimal_locations(num_chargers)
            
            # Complete the process
            self.queue.put({
                "type": "complete", 
                "locations": new_locations
            })
            
        except Exception as e:
            import traceback
            self.queue.put({
                "type": "error", 
                "text": f"{str(e)}\n{traceback.format_exc()}"
            })
    
    def stop_calculation(self):
        """Stop the current optimization process"""
        if self.is_processing:
            self.is_processing = False
            self.status_label.configure(text="Stopping...")
            self.log_message("Stopping optimization... (Please wait)")
    
    def display_results(self):
        """Display optimization results in the results tab"""
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "Optimal locations for new EV chargers:\n\n")

        existing_chargers = getattr(self, 'selected_existing_chargers', self.existing_chargers)
        self.result_text.insert("end", f"Existing chargers included: {len(existing_chargers)}\n")
        self.result_text.insert("end", f"New chargers placed: {len(self.new_locations)}\n\n")
        
        # Display charger locations
        for i, location in enumerate(self.new_locations, 1):
            self.result_text.insert("end", 
                f"Charger {i}: Latitude: {location.latitude:.4f}, "
                f"Longitude: {location.longitude:.4f}\n")
        
        # Calculate and display coverage statistics
        if self.optimizer:
            stats = self.optimizer._calculate_coverage_efficiency(
                existing_chargers + self.new_locations
            )
            
            self.result_text.insert("end", f"\nCoverage Statistics:\n")
            self.result_text.insert("end", f"Coverage Percentage: {stats['coverage']:.2f}%\n")
            self.result_text.insert("end", f"Overlap Percentage: {stats['overlap']:.2f}%\n")
            self.result_text.insert("end", f"Efficiency: {stats['efficiency']:.2f}%\n")
        
        # Switch to results tab
        self.tabview.set("Results")

    def _get_current_weights(self):
        weights = {}
        if self.advanced_checkbox.get():
            for key, entry in self.weights.items():
                try:
                    weights[key] = float(entry.get())
                except ValueError:
                    weights[key] = 0.0
        else:
            weights = {
                'population_density': 0.20,
                'traffic_flow': 0.25,
                'points_of_interest': 0.15,
                'power_availability': 0.15,
                'charging_demand': 0.10,
                'accessibility': 0.10,
                'road_quality': 0.03,
                'revenue_potential': 0.02
            }
        return weights

    def generate_report(self):
        if not self.new_locations:
            self.status_label.configure(text="No results to report")
            return

        existing_chargers = getattr(self, 'selected_existing_chargers', self.existing_chargers)
        report_type = self.report_type.get() if hasattr(self, 'report_type') else "Full Report"
        weights = self._get_current_weights()
        report_text = self.build_report_text(report_type, existing_chargers, self.new_locations, weights)

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", report_text)
        self.status_label.configure(text=f"{report_type} generated")
        self.tabview.set("Results")

    def generate_figures(self):
        if not self.new_locations:
            self.status_label.configure(text="No results to generate figures")
            return

        existing_chargers = getattr(self, 'selected_existing_chargers', self.existing_chargers)
        weights = self._get_current_weights()
        stats = self.optimizer._calculate_coverage_efficiency(existing_chargers + self.new_locations)
        parameters = {
            "bounds": self.goa_bounds,
            "coverage_radius": float(self.coverage_radius.get()),
            "weights": weights,
            "num_chargers": len(self.new_locations)
        }
        result_name = time.strftime("figures_%Y%m%d_%H%M%S")
        figure_paths = self.results_handler.generate_publication_figures_from_data(
            existing_chargers,
            self.new_locations,
            parameters,
            stats,
            weights,
            result_name=result_name
        )

        self.result_text.insert("end", "\nGenerated publication figures:\n")
        for path in figure_paths:
            self.result_text.insert("end", f"  {path}\n")

        self.status_label.configure(text="Publication figures generated")
        self.tabview.set("Results")

    def build_report_text(self, report_type, existing_chargers, new_locations, weights):
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        title = f"EV Charger Optimization Report - {report_type}\nGenerated: {now}\n"
        divider = "\n" + "="*72 + "\n"
        lines = [title, divider]

        params = {
            'Area bounds': f"{self.goa_bounds}",
            'Coverage radius': f"{float(self.coverage_radius.get()):.2f} km",
            'New chargers requested': len(new_locations),
            'Existing chargers included': len(existing_chargers)
        }

        lines.append("Configuration:\n")
        for key, value in params.items():
            lines.append(f"  {key}: {value}\n")

        if report_type in ["Summary", "Full Report"]:
            lines.append("\nSelected Existing Chargers:\n")
            for i, charger in enumerate(existing_chargers, 1):
                lines.append(f"  {i}. {charger.latitude:.4f}, {charger.longitude:.4f}\n")

            lines.append("\nNew Charger Locations:\n")
            for i, charger in enumerate(new_locations, 1):
                lines.append(f"  {i}. {charger.latitude:.4f}, {charger.longitude:.4f}\n")

        if report_type in ["Coverage Analysis", "Full Report", "Summary"]:
            optimizer = self.optimizer or ChargerOptimizer(
                area_bounds=self.goa_bounds,
                existing_chargers=existing_chargers,
                coverage_radius=float(self.coverage_radius.get()),
                weights=weights
            )
            stats = optimizer._calculate_coverage_efficiency(existing_chargers + new_locations)
            lines.append("\nCoverage Analysis:\n")
            lines.append(f"  Total land area: {stats['total_area']:.2f} km²\n")
            lines.append(f"  Covered area: {stats['covered_area']:.2f} km²\n")
            lines.append(f"  Coverage: {stats['coverage']:.2f}%\n")
            lines.append(f"  Overlap area: {stats['overlap_area']:.2f} km²\n")
            lines.append(f"  Overlap: {stats['overlap']:.2f}%\n")
            lines.append(f"  Efficiency: {stats['efficiency']:.2f}%\n")

        if report_type in ["Distance Matrix", "Full Report"]:
            lines.append("\nInter-Charger Distance Summary:\n")
            distances = []
            for i, charger in enumerate(new_locations):
                for j, other in enumerate(new_locations):
                    if i < j:
                        distances.append(haversine_distance(charger, other))
            if distances:
                lines.append(f"  Minimum separation: {min(distances):.2f} km\n")
                lines.append(f"  Maximum separation: {max(distances):.2f} km\n")
                lines.append(f"  Average separation: {sum(distances)/len(distances):.2f} km\n")
            else:
                lines.append("  Not enough new locations to calculate pairwise distances.\n")

        if report_type in ["Weights Overview", "Full Report"]:
            lines.append("\nWeight Configuration:\n")
            total_weight = sum(weights.values()) if weights else 0.0
            for key, value in weights.items():
                pct = (value / total_weight * 100) if total_weight > 0 else 0.0
                lines.append(f"  {key}: {value:.3f} ({pct:.1f}% of total)\n")

        if report_type in ["Efficiency Metrics", "Full Report"]:
            efficiency_results = optimizer.evaluate_model_efficiency(num_chargers=len(new_locations))
            lines.append("\nEfficiency Metrics:\n")
            lines.append(f"  Composite efficiency: {efficiency_results.get('composite_efficiency', 0):.4f}\n")
            lines.append(f"  Coverage efficiency: {efficiency_results.get('coverage_efficiency', 0):.2f}%\n")
            lines.append(f"  Coverage percentage: {efficiency_results.get('coverage_percentage', 0):.2f}%\n")
            lines.append(f"  Overlap percentage: {efficiency_results.get('overlap_percentage', 0):.2f}%\n")
            lines.append(f"  Computational efficiency: {efficiency_results.get('computational_efficiency', 0):.4f}\n")
            lines.append(f"  Optimization time: {efficiency_results.get('optimization_time', 0):.2f} seconds\n")
            lines.append(f"  Prediction consistency: {efficiency_results.get('prediction_consistency', 0):.4f}\n")

        if report_type == "Full Report":
            lines.append("\nNote: This full report contains configuration, charger placement, coverage analysis, distance statistics, weight breakdown, and model efficiency metrics for use in IEEE paper preparation.\n")

        return "".join(lines)

    def show_map(self):
        """Display the map visualization"""
        existing_chargers = getattr(self, 'selected_existing_chargers', self.existing_chargers)
        stats = None
        if self.optimizer and self.new_locations:
            stats = self.optimizer._calculate_coverage_efficiency(
                existing_chargers + self.new_locations
            )
        self.map_visualizer.create_map(
            existing_chargers,
            self.new_locations,
            area_bounds=self.goa_bounds,
            coverage_radius=float(self.coverage_radius.get()) if self.coverage_radius.get() else 5.0,
            stats=stats
        )
        
    def save_results(self):
        """Save current results to a file"""
        if not self.new_locations:
            self.status_label.configure(text="No results to save")
            return
            
        weights = self._get_current_weights()
        parameters = {
            "num_chargers": len(self.new_locations),
            "bounds": self.goa_bounds,
            "coverage_radius": float(self.coverage_radius.get()),
            "weights": weights
        }
        
        existing_chargers = getattr(self, 'selected_existing_chargers', self.existing_chargers)
        filepath = self.results_handler.save_optimization_result(
            self.new_locations,
            existing_chargers,
            parameters
        )
        
        self.result_text.insert("end", f"\nResults saved to: {filepath}\n")
        self.status_label.configure(text=f"Results saved")
        
    def load_results(self):
        """Load results from a file"""
        results = self.results_handler.list_results()
        if not results:
            self.status_label.configure(text="No saved results found")
            self.result_text.insert("end", "\nNo saved results found.\n")
            return
            
        # Create a dialog window for selecting results
        dialog = ctk.CTkToplevel(self)
        dialog.title("Load Results")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()
        
        # Make the dialog modal
        dialog.focus_set()
        
        # Create a list of results
        ctk.CTkLabel(dialog, text="Select a result file to load:").pack(pady=10)
        
        results_frame = ctk.CTkScrollableFrame(dialog)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Dictionary to hold radiobutton variables
        selected_result = ctk.StringVar()
        selected_result.set("")  # Default empty
        
        for result in results:
            rb = ctk.CTkRadioButton(results_frame, text=result, variable=selected_result, value=result)
            rb.pack(anchor="w", padx=10, pady=5)
            
        # Buttons
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", 
                                  command=dialog.destroy)
        cancel_btn.pack(side="left", padx=10, pady=10)
        
        def load_selected():
            result_file = selected_result.get()
            if result_file:
                try:
                    data = self.results_handler.load_optimization_result(result_file)
                    self.new_locations = data['new_chargers']
                    self.display_results()
                    self.show_map_btn.configure(state="normal")
                    self.save_btn.configure(state="normal")
                    self.efficiency_btn.configure(state="normal")
                    if hasattr(self, 'generate_report_btn'):
                        self.generate_report_btn.configure(state="normal")
                    if hasattr(self, 'generate_figures_btn'):
                        self.generate_figures_btn.configure(state="normal")
                    self.status_label.configure(text=f"Loaded: {result_file}")
                    dialog.destroy()
                except Exception as e:
                    self.log_message(f"Error loading results: {str(e)}")
            
        load_btn = ctk.CTkButton(button_frame, text="Load", 
                               command=load_selected)
        load_btn.pack(side="right", padx=10, pady=10)
        
    def calculate_efficiency(self):
        """Calculate and display model efficiency metrics"""
        if not self.optimizer or not self.new_locations:
            self.status_label.configure(text="No optimization results")
            return
            
        # Start in background thread to avoid freezing UI
        threading.Thread(
            target=self.run_efficiency_calculation,
            daemon=True
        ).start()
        
    def run_efficiency_calculation(self):
        """Run efficiency calculation in background thread"""
        try:
            self.queue.put({"type": "status", "text": "Calculating efficiency..."})
            self.queue.put({"type": "log", "text": "Starting efficiency calculation..."})
            
            # Calculate efficiency
            try:
                efficiency_results = self.optimizer.evaluate_model_efficiency(
                    num_chargers=len(self.new_locations)
                )
                
                # Format results with safeguards against missing keys
                result_text = "\n=== Efficiency Metrics ===\n\n"
                
                # Access values with safeguards
                comp_eff = efficiency_results.get('composite_efficiency', 0)
                result_text += f"Composite Efficiency Score: {comp_eff:.4f}\n"
                
                cov_eff = efficiency_results.get('coverage_efficiency', 0)
                result_text += f"Coverage Efficiency: {cov_eff:.2f}%\n"
                
                comp_time = efficiency_results.get('computational_efficiency', 0)
                result_text += f"Computational Efficiency: {comp_time:.4f}\n"
                
                opt_time = efficiency_results.get('optimization_time', 0)
                result_text += f"Optimization Time: {opt_time:.2f} seconds\n"
                
                iterations = efficiency_results.get('iterations_required', 0)
                result_text += f"Iterations Required: {iterations}\n"
                
                dist_score = efficiency_results.get('spatial_distribution', 0)
                result_text += f"Spatial Distribution: {dist_score:.4f}\n"
                
                # Memory usage with safeguards
                memory = efficiency_results.get('memory_usage', {})
                result_text += f"\nMemory Usage:\n"
                result_text += f"  Grid Cache: {memory.get('grid_cache_kb', 0):.2f} KB\n"
                result_text += f"  Candidate Cache: {memory.get('candidate_cache_kb', 0):.2f} KB\n"
                result_text += f"  Model Size: {memory.get('model_size_kb', 0):.2f} KB\n"
                result_text += f"  Total: {memory.get('total_kb', 0):.2f} KB\n"
                
                # Update UI
                self.result_text.insert("end", result_text)
                self.queue.put({"type": "status", "text": "Efficiency calculation complete"})
                self.queue.put({"type": "log", "text": "Efficiency calculation complete"})
                
            except Exception as e:
                import traceback
                error_msg = f"Error during efficiency calculation: {str(e)}\n{traceback.format_exc()}"
                self.queue.put({"type": "log", "text": error_msg})
                self.queue.put({"type": "status", "text": "Efficiency calculation failed"})
                
                # Still show some partial results if possible
                result_text = "\n=== Efficiency Calculation Error ===\n\n"
                result_text += f"Error: {str(e)}\n\n"
                result_text += "Showing partial results if available:\n"
                
                # Try to extract any available data
                if hasattr(self.optimizer, 'grid_cache'):
                    result_text += f"Grid Cache Size: {len(self.optimizer.grid_cache)} entries\n"
                if hasattr(self.optimizer, 'candidate_cache'):
                    result_text += f"Candidate Cache Size: {len(self.optimizer.candidate_cache)} entries\n"
                    
                self.result_text.insert("end", result_text)
                
        except Exception as e:
            import traceback
            self.queue.put({
                "type": "error", 
                "text": f"Efficiency calculation error: {str(e)}\n{traceback.format_exc()}"
            })
