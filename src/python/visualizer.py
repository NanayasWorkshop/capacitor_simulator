"""
Capacitor Visualizer - Open3D dynamic 3D visualization
Shows sensor movement with heat color arrows based on ray distance
"""

import numpy as np
import open3d as o3d
import time
import threading
import logging
import warnings
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import os
import math

# Suppress Open3D and GLFW warnings
warnings.filterwarnings("ignore", message=".*GLFW.*")
warnings.filterwarnings("ignore", message=".*WGL.*")

logger = logging.getLogger(__name__)

class CapacitorVisualizer:
    """Dynamic 3D visualization of capacitor sensors using Open3D"""
    
    def __init__(self, results=None, fps: int = 30, export_png: bool = False, export_video: bool = False):
        """Initialize the visualizer"""
        self.results = results
        self.fps = fps
        self.export_png = export_png
        self.export_video = export_video
        self.movement_scale = 10.0  # 10 MILLION x scaling
        
        # Animation control
        self.current_step = 0
        self.max_steps = 0
        self.playing = False
        self.animation_thread = None
        
        # Open3D components
        self.vis = None
        self.geometries = {}
        self.sensor_positions = self._calculate_sensor_positions()
        
        # Colors for different sensors/models
        self.colors = {
            'A1': [1.0, 0.0, 1.0],    # Magenta
            'A2': [0.0, 1.0, 1.0],    # Cyan
            'B1': [1.0, 1.0, 0.0],    # Yellow
            'B2': [0.0, 1.0, 0.0],    # Green
            'C1': [0.0, 0.0, 1.0],    # Blue
            'C2': [1.0, 0.0, 0.0],    # Red
            'negative': [0.5, 0.5, 0.5]  # Gray
        }
        
        # C++ engine for ray calculations
        self.cpp_engine = None
        
        # Arrow length tracking for heatmap - GLOBAL min/max across ALL data
        self.global_min_distance = float('inf')
        self.global_max_distance = 0.0
        
        logger.info(f"CapacitorVisualizer initialized with {self.movement_scale}x movement scaling")
    
    def _calculate_sensor_positions(self):
        """Calculate triangular positions for sensors A, B, C"""
        positions = {}
        radius = 26.45  # mm
        
        # Triangular formation with 120 degree spacing
        for i, sensor in enumerate(['A', 'C', 'B']):
            angle = i * (2 * math.pi / 3) + (math.pi / 2)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.0
            positions[sensor] = np.array([x, y, z])
        
        logger.info(f"Sensor positions (radius {radius} mm):")
        for sensor in ['A', 'B', 'C']:
            if sensor in positions:
                pos = positions[sensor]
                logger.info(f"  Sensor {sensor}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) mm")
        
        return positions
    
    def _initialize_cpp_engine(self):
        """Initialize C++ engine for ray calculations"""
        try:
            if os.path.exists("C:\\embree\\bin"):
                os.add_dll_directory("C:\\embree\\bin")
            
            import capacitor_cpp
            
            config = capacitor_cpp.CapacitorConfig()
            config.verbose_logging = False
            config.max_ray_distance = 5.0
            config.bidirectional_rays = True
            config.ray_density = 10
            config.collect_ray_data = True
            
            self.cpp_engine = capacitor_cpp.CapacitorEngine(config)
            logger.info("C++ ray tracing engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize C++ engine: {e}")
            raise
    
    def _load_models_into_engine(self):
        """Load all 3D models into the C++ engine"""
        logger.info("Loading models into C++ engine...")
        
        models_loaded = 0
        for model_name, model_data in self.data['model_data'].items():
            vertices = model_data['vertices'].flatten().astype(float).tolist()
            faces = model_data['faces'].flatten().astype(int).tolist()
            
            success = self.cpp_engine.load_mesh(model_name, vertices, faces)
            if success:
                models_loaded += 1
                
        logger.info(f"Loaded {models_loaded} models into ray tracing engine")
    
    def _load_simulation_data(self):
        """Load data needed for visualization from simulation results"""
        if not self.results:
            raise ValueError("No simulation results provided")
        
        if os.path.exists("C:\\embree\\bin"):
            os.add_dll_directory("C:\\embree\\bin")
        
        import sys
        sys.path.insert(0, 'src/python')
        from data_loader import DataLoader
        
        loader = DataLoader()
        self.data = loader.load_all_data(self.results.data_info['data_path'])
        
        self._initialize_cpp_engine()
        self._load_models_into_engine()
        
        if self.data['transformations']:
            self.max_steps = len(list(self.data['transformations'].values())[0])
        else:
            self.max_steps = 1
        
        # Calculate GLOBAL distance range across ALL time steps for proper heat mapping
        self._calculate_global_distance_range()
        
        logger.info(f"Loaded visualization data: {self.max_steps} time steps")
    
    def _calculate_global_distance_range(self):
        """Calculate global min/max distances across ALL time steps and sensors for heat mapping"""
        logger.info("Calculating global distance range for heat mapping...")
        
        all_distances = []
        
        # Go through ALL time steps (or sample if too many)
        max_sample_steps = 20
        step_indices = np.linspace(0, self.max_steps - 1, min(max_sample_steps, self.max_steps), dtype=int)
        
        for step_idx in step_indices:
            for sensor in ['A', 'B', 'C']:
                if sensor not in self.data['transformations']:
                    continue
                
                transformation = self.data['transformations'][sensor][step_idx]
                
                for model_num in ['1', '2']:
                    model_name = f"{sensor}{model_num}"
                    distances = self._get_ray_distances_for_step(model_name, transformation)
                    all_distances.extend(distances)
        
        if all_distances:
            self.global_min_distance = min(all_distances)
            self.global_max_distance = max(all_distances)
            logger.info(f"Global distance range: {self.global_min_distance:.4f} - {self.global_max_distance:.4f} mm")
        else:
            self.global_min_distance = 0.0
            self.global_max_distance = 5.0
            logger.warning("No distances found, using default range")
    
    def _get_ray_distances_for_step(self, model_name: str, transformation: np.ndarray) -> List[float]:
        """Get ray distances for a specific model and transformation"""
        if self.cpp_engine is None:
            return []
        
        try:
            import capacitor_cpp
            
            cpp_matrix = capacitor_cpp.Matrix4x4()
            cpp_matrix.set_from_numpy(transformation.astype(float))
            
            ray_data = self.cpp_engine.get_ray_data(model_name, 'stationary_negative', cpp_matrix)
            
            distances = []
            for ray in ray_data:
                if ray.hit:
                    distances.append(ray.distance)
            
            return distances
            
        except Exception as e:
            logger.debug(f"Failed to get distances for {model_name}: {e}")
            return []
    
    def _hex_to_rgb(self, hex_color: str) -> List[float]:
        """Convert hex color to RGB float values (0-1)"""
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    
    def _distance_to_heat_color(self, distance: float) -> List[float]:
        """Convert distance to heat color using the specified palette"""
        # Handle edge case
        if self.global_max_distance <= self.global_min_distance:
            return self._hex_to_rgb("#000046")  # Default to darkest
        
        # Normalize distance to 0-1 range using GLOBAL min/max
        normalized = (distance - self.global_min_distance) / (self.global_max_distance - self.global_min_distance)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to 0-1
        
        # Heat palette: shortest to longest
        heat_colors = [
            "#000046",  # Dark blue (SHORTEST distances)
            "#2E008C",  # Purple blue
            "#B70395",  # Magenta
            "#EE5E05",  # Orange
            "#FEC201",  # Yellow
            "#FFF5A0"   # Light yellow (LONGEST distances)
        ]
        
        # Convert to RGB
        rgb_colors = [self._hex_to_rgb(color) for color in heat_colors]
        
        # Find segment and interpolate
        num_segments = len(rgb_colors) - 1
        segment_size = 1.0 / num_segments
        segment_index = min(int(normalized / segment_size), num_segments - 1)
        
        segment_start = segment_index * segment_size
        local_t = (normalized - segment_start) / segment_size if segment_size > 0 else 0.0
        
        color1 = rgb_colors[segment_index]
        color2 = rgb_colors[min(segment_index + 1, len(rgb_colors) - 1)]
        
        # Linear interpolation
        result = [
            color1[0] + (color2[0] - color1[0]) * local_t,
            color1[1] + (color2[1] - color1[1]) * local_t,
            color1[2] + (color2[2] - color1[2]) * local_t
        ]
        
        return result
    
    def _create_mesh_from_data(self, model_name: str, color: List[float]):
        """Create Open3D mesh from model data"""
        if model_name not in self.data['model_data']:
            logger.warning(f"Model {model_name} not found")
            return None
        
        model_data = self.data['model_data'][model_name]
        vertices = model_data['vertices']
        faces = model_data['faces']
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        
        return mesh
    
    def _apply_transformation(self, mesh: o3d.geometry.TriangleMesh, transformation: np.ndarray):
        """Apply transformation with extreme scaling"""
        if mesh is None:
            return None
        
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.vertices = mesh.vertices
        transformed_mesh.triangles = mesh.triangles
        transformed_mesh.vertex_normals = mesh.vertex_normals
        transformed_mesh.vertex_colors = mesh.vertex_colors
        
        # Apply extreme scaling to see movement
        scaled_transformation = transformation.copy()
        scaled_transformation[0:3, 3] *= self.movement_scale
        
        transformed_mesh.transform(scaled_transformation)
        return transformed_mesh
    
    def _apply_sensor_offset(self, mesh: o3d.geometry.TriangleMesh, sensor_group: str):
        """Apply sensor positioning offset"""
        if mesh is None or sensor_group not in self.sensor_positions:
            return mesh
        
        offset = self.sensor_positions[sensor_group]
        translation_matrix = np.eye(4)
        translation_matrix[0:3, 3] = offset
        
        mesh.transform(translation_matrix)
        return mesh
    
    def _create_coordinate_axes(self):
        """Create coordinate axes for reference"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    
    def _create_step_counter(self):
        """Create step counter display"""
        try:
            text_position = [0, 0, 40]
            text_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            text_frame.translate(text_position)
            
            progress = self.current_step / max(1, self.max_steps - 1)
            color = [progress, 1.0 - progress, 0.5]
            text_frame.paint_uniform_color(color)
            
            return text_frame
        except:
            return None
    
    def _update_step_counter(self):
        """Update step counter display"""
        if 'step_counter' in self.geometries:
            self.vis.remove_geometry(self.geometries['step_counter'], reset_bounding_box=False)
            del self.geometries['step_counter']
        
        step_frame = self._create_step_counter()
        if step_frame:
            self.vis.add_geometry(step_frame, reset_bounding_box=False)
            self.geometries['step_counter'] = step_frame
        
        logger.info(f"Step: {self.current_step + 1}/{self.max_steps}")
    
    def _create_heat_arrows(self, model_name: str, sensor_group: str, transformation: np.ndarray):
        """Create arrows with heat colors based on distance"""
        if self.cpp_engine is None:
            return []
        
        try:
            import capacitor_cpp
            
            cpp_matrix = capacitor_cpp.Matrix4x4()
            cpp_matrix.set_from_numpy(transformation.astype(float))
            
            ray_data = self.cpp_engine.get_ray_data(model_name, 'stationary_negative', cpp_matrix)
            
            arrows = []
            
            for ray in ray_data:
                if ray.hit:  # Only arrows that actually hit
                    origin = np.array(ray.origin)
                    hit_point = np.array(ray.hit_point)
                    
                    # Apply sensor offset
                    offset = self.sensor_positions[sensor_group]
                    origin += offset
                    hit_point += offset
                    
                    # Get heat color based on distance using GLOBAL range
                    heat_color = self._distance_to_heat_color(ray.distance)
                    
                    # Create thick arrow line
                    line_points = [origin, hit_point]
                    lines = [[0, 1]]
                    
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([heat_color])
                    
                    arrows.append(line_set)
            
            return arrows
            
        except Exception as e:
            logger.error(f"Failed to create heat arrows for {model_name}: {e}")
            return []
    
    def _setup_visualization(self):
        """Initialize visualization"""
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Capacitor Heat Map Visualization", width=1200, height=800)
        
        self._load_simulation_data()
        
        logger.info("Creating geometries...")
        
        # Add coordinate axes
        axes = self._create_coordinate_axes()
        self.vis.add_geometry(axes)
        self.geometries['axes'] = axes
        
        # Add negative plates
        for sensor in ['A', 'B', 'C']:
            negative_mesh = self._create_mesh_from_data('stationary_negative', self.colors['negative'])
            if negative_mesh:
                negative_mesh = self._apply_sensor_offset(negative_mesh, sensor)
                self.vis.add_geometry(negative_mesh)
                self.geometries[f'negative_{sensor}'] = negative_mesh
                logger.info(f"Added negative plate at sensor {sensor}")
        
        # Add sensor meshes
        for sensor in ['A', 'B', 'C']:
            for model_num in ['1', '2']:
                model_name = f"{sensor}{model_num}"
                color = self.colors.get(model_name, [0.5, 0.5, 0.5])
                
                mesh = self._create_mesh_from_data(model_name, color)
                if mesh:
                    mesh = self._apply_sensor_offset(mesh, sensor)
                    self.vis.add_geometry(mesh)
                    self.geometries[model_name] = mesh
        
        # Setup camera and rendering
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.3)
        view_control.set_front([0.5, 0.5, 0.8])
        view_control.set_up([0, 0, 1])
        
        render_option = self.vis.get_render_option()
        render_option.show_coordinate_frame = True
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
        render_option.line_width = 6.0  # THICK arrows
        render_option.mesh_show_back_face = True  # Both sides
        
        logger.info(f"Setup complete. {self.max_steps} time steps with {self.movement_scale}x scaling")
        logger.info(f"Heat range: {self.global_min_distance:.4f} - {self.global_max_distance:.4f} mm")
        logger.info("Heat colors: Dark Blue (short) → Purple → Magenta → Orange → Yellow → Light Yellow (long)")
    
    def _update_step(self, step: int):
        """Update visualization for specific time step"""
        if step < 0 or step >= self.max_steps:
            return
        
        self.current_step = step
        self._update_step_counter()
        
        # Update each sensor
        for sensor in ['A', 'B', 'C']:
            if sensor not in self.data['transformations']:
                continue
                
            transformation = self.data['transformations'][sensor][step]
            
            for model_num in ['1', '2']:
                model_name = f"{sensor}{model_num}"
                
                if model_name not in self.geometries:
                    continue
                
                # Update mesh
                color = self.colors.get(model_name, [0.5, 0.5, 0.5])
                base_mesh = self._create_mesh_from_data(model_name, color)
                
                if base_mesh is None:
                    continue
                
                transformed_mesh = self._apply_transformation(base_mesh, transformation)
                transformed_mesh = self._apply_sensor_offset(transformed_mesh, sensor)
                
                old_mesh = self.geometries[model_name]
                old_mesh.vertices = transformed_mesh.vertices
                old_mesh.triangles = transformed_mesh.triangles
                old_mesh.vertex_normals = transformed_mesh.vertex_normals
                old_mesh.vertex_colors = transformed_mesh.vertex_colors
                
                self.vis.update_geometry(old_mesh)
                
                # Remove old arrows
                arrow_keys = [key for key in self.geometries.keys() if key.startswith(f"{model_name}_arrow_")]
                for arrow_key in arrow_keys:
                    self.vis.remove_geometry(self.geometries[arrow_key], reset_bounding_box=False)
                    del self.geometries[arrow_key]
                
                # Create new heat arrows
                heat_arrows = self._create_heat_arrows(model_name, sensor, transformation)
                for j, arrow in enumerate(heat_arrows):
                    arrow_name = f"{model_name}_arrow_{j}"
                    self.vis.add_geometry(arrow, reset_bounding_box=False)
                    self.geometries[arrow_name] = arrow
    
    def _animation_loop(self):
        """Background animation loop"""
        frame_time = 1.0 / self.fps
        
        while self.playing:
            start_time = time.time()
            
            next_step = (self.current_step + 1) % self.max_steps
            self._update_step(next_step)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
    
    def toggle_play_pause(self):
        """Toggle animation"""
        self.playing = not self.playing
        
        if self.playing:
            logger.info("Animation started")
            self.animation_thread = threading.Thread(target=self._animation_loop)
            self.animation_thread.daemon = True
            self.animation_thread.start()
        else:
            logger.info("Animation paused")
    
    def next_step(self):
        """Move to next step"""
        next_step = min(self.current_step + 1, self.max_steps - 1)
        self._update_step(next_step)
    
    def previous_step(self):
        """Move to previous step"""
        prev_step = max(self.current_step - 1, 0)
        self._update_step(prev_step)
    
    def restart_animation(self):
        """Restart animation"""
        self.playing = False
        time.sleep(0.1)
        self._update_step(0)
        logger.info("Animation restarted")
    
    def show_interactive(self):
        """Show interactive visualization"""
        logger.info("Starting heat map visualization...")
        logger.info("Heat Colors: Dark Blue (shortest) → Purple → Magenta → Orange → Yellow → Light Yellow (longest)")
        logger.info("Controls:")
        logger.info("  Spacebar: Play/Pause")
        logger.info("  Arrow keys: Step forward/backward")
        logger.info("  R: Restart")
        logger.info("  Mouse: Rotate/pan/zoom")
        
        self._setup_visualization()
        self._update_step(0)
        self.toggle_play_pause()
        
        try:
            while True:
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Visualization interrupted")
        finally:
            self.playing = False
            if self.animation_thread and self.animation_thread.is_alive():
                self.animation_thread.join(timeout=1.0)
            
            self.vis.destroy_window()
            logger.info("Visualization closed")