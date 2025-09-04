"""
Capacitor Visualizer - Open3D dynamic 3D visualization
Shows sensor movement and capacitance changes in real-time
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
        
        logger.info("CapacitorVisualizer initialized")
    
    def _calculate_sensor_positions(self):
        """Calculate triangular positions for sensors A, B, C - EXACT COPY from original working code"""
        positions = {}
        radius = 26.45  # mm
        
        # Triangular formation with 120 degree spacing around Y axis (ZX plane) - FROM ORIGINAL
        for i, sensor in enumerate(['A', 'C', 'B']):  # ORIGINAL ORDER: A, C, B
            angle = i * (2 * math.pi / 3) + (math.pi / 2)  # Add 90° rotation
            x = radius * math.cos(angle)  # X coordinate
            y = radius * math.sin(angle)  # Y coordinate
            z = 0.0                       # Z is constant (horizontal plane)
            positions[sensor] = np.array([x, y, z])
        
        logger.info(f"Sensor positions (radius {radius} mm, triangular formation):")
        for sensor in ['A', 'B', 'C']:  # Print in ABC order
            if sensor in positions:
                pos = positions[sensor]
                logger.info(f"  Sensor {sensor}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) mm")
        
        return positions
    
    def _load_simulation_data(self):
        """Load data needed for visualization from simulation results"""
        if not self.results:
            raise ValueError("No simulation results provided")
        
        # Add DLL directory for C++ extension
        if os.path.exists("C:\\embree\\bin"):
            os.add_dll_directory("C:\\embree\\bin")
        
        # Import required modules
        import sys
        sys.path.insert(0, 'src/python')
        from data_loader import DataLoader
        
        # Load the original data to get meshes and transformations
        loader = DataLoader()
        self.data = loader.load_all_data(self.results.data_info['data_path'])
        
        # Determine max steps from transformations
        if self.data['transformations']:
            self.max_steps = len(list(self.data['transformations'].values())[0])
        else:
            self.max_steps = 1
        
        logger.info(f"Loaded visualization data: {self.max_steps} time steps")
    
    def _create_mesh_from_data(self, model_name: str, color: List[float]):
        """Create Open3D mesh from model data"""
        if model_name not in self.data['model_data']:
            logger.warning(f"Model {model_name} not found in data")
            return None
        
        model_data = self.data['model_data'][model_name]
        vertices = model_data['vertices']
        faces = model_data['faces']
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Set color
        mesh.paint_uniform_color(color)
        
        # Calculate normals for better rendering
        mesh.compute_vertex_normals()
        
        return mesh
    
    def _apply_transformation(self, mesh: o3d.geometry.TriangleMesh, transformation: np.ndarray):
        """Apply transformation matrix to mesh - FIXED for Open3D compatibility"""
        if mesh is None:
            return None
        
        # Create a new mesh with same data (Open3D version compatibility fix)
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.vertices = mesh.vertices
        transformed_mesh.triangles = mesh.triangles
        transformed_mesh.vertex_normals = mesh.vertex_normals
        transformed_mesh.vertex_colors = mesh.vertex_colors
        
        # Apply transformation
        transformed_mesh.transform(transformation)
        
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
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        return axes
    
    def _create_triangle_normals_arrows(self, mesh: o3d.geometry.TriangleMesh, arrow_scale: float = 2.0, every_nth: int = 10):
        """Create arrow geometries showing triangle normals (every 10th triangle)"""
        if mesh is None:
            return []
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        arrows = []
        
        # Process every nth triangle
        for i in range(0, len(triangles), every_nth):
            tri = triangles[i]
            
            # Get triangle vertices
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            
            # Calculate triangle center
            center = (v0 + v1 + v2) / 3.0
            
            # Calculate triangle normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_length = np.linalg.norm(normal)
            
            if normal_length > 0:
                normal = normal / normal_length * arrow_scale
                
                # Create arrow from center in normal direction
                arrow_end = center + normal
                
                # Create a simple line set for the arrow
                line_points = [center, arrow_end]
                lines = [[0, 1]]
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([[0.0, 0.8, 0.0]])  # Green arrows
                
                arrows.append(line_set)
        
        return arrows
    
    def _setup_visualization(self):
        """Initialize Open3D visualizer with warning suppression"""
        # Suppress Open3D warnings completely
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Capacitor Sensor Visualization", width=1200, height=800)
        
        # Load simulation data
        self._load_simulation_data()
        
        # Create static geometries - 3 negative plates, one at each sensor position
        logger.info("Creating static geometries...")
        
        # Add coordinate axes
        axes = self._create_coordinate_axes()
        self.vis.add_geometry(axes)
        self.geometries['axes'] = axes
        
        # Add stationary negative plates at each sensor position (A, B, C)
        for sensor in ['A', 'B', 'C']:
            negative_mesh = self._create_mesh_from_data('stationary_negative', self.colors['negative'])
            if negative_mesh:
                # Apply sensor positioning offset to negative plate
                negative_mesh = self._apply_sensor_offset(negative_mesh, sensor)
                
                self.vis.add_geometry(negative_mesh)
                self.geometries[f'negative_{sensor}'] = negative_mesh
                logger.info(f"Added negative plate at sensor {sensor} position")
        
        # Create initial sensor meshes with arrows
        logger.info("Creating sensor meshes...")
        for sensor in ['A', 'B', 'C']:
            for model_num in ['1', '2']:
                model_name = f"{sensor}{model_num}"
                color = self.colors.get(model_name, [0.5, 0.5, 0.5])
                
                mesh = self._create_mesh_from_data(model_name, color)
                if mesh:
                    # Apply sensor positioning offset
                    mesh = self._apply_sensor_offset(mesh, sensor)
                    
                    self.vis.add_geometry(mesh)
                    self.geometries[model_name] = mesh
                    
                    # Create and add normal arrows for this mesh (every 10th triangle)
                    arrows = self._create_triangle_normals_arrows(mesh, arrow_scale=1.0, every_nth=10)
                    for j, arrow in enumerate(arrows):
                        arrow_name = f"{model_name}_arrow_{j}"
                        self.vis.add_geometry(arrow)
                        self.geometries[arrow_name] = arrow
        
        # Setup camera
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.3)
        view_control.set_front([0.5, 0.5, 0.8])
        view_control.set_up([0, 0, 1])
        
        # Setup render options - WHITE BACKGROUND
        render_option = self.vis.get_render_option()
        render_option.show_coordinate_frame = True
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # WHITE BACKGROUND
        render_option.point_size = 2.0
        render_option.line_width = 1.0
        
        logger.info(f"Visualization setup complete. Ready for {self.max_steps} time steps.")
    
    def _update_step(self, step: int):
        """Update visualization for a specific time step"""
        if step < 0 or step >= self.max_steps:
            return
        
        self.current_step = step
        
        # Update each sensor mesh with transformation for this step
        for sensor in ['A', 'B', 'C']:
            if sensor not in self.data['transformations']:
                continue
                
            transformation = self.data['transformations'][sensor][step]
            
            for model_num in ['1', '2']:
                model_name = f"{sensor}{model_num}"
                
                if model_name not in self.geometries:
                    continue
                
                # Get base mesh
                color = self.colors.get(model_name, [0.5, 0.5, 0.5])
                base_mesh = self._create_mesh_from_data(model_name, color)
                
                if base_mesh is None:
                    continue
                
                # Apply transformation
                transformed_mesh = self._apply_transformation(base_mesh, transformation)
                
                # Apply sensor positioning offset
                transformed_mesh = self._apply_sensor_offset(transformed_mesh, sensor)
                
                # Update geometry in visualizer
                old_mesh = self.geometries[model_name]
                old_mesh.vertices = transformed_mesh.vertices
                old_mesh.triangles = transformed_mesh.triangles
                old_mesh.vertex_normals = transformed_mesh.vertex_normals
                old_mesh.vertex_colors = transformed_mesh.vertex_colors
                
                self.vis.update_geometry(old_mesh)
                
                # Update arrows for this mesh
                updated_arrows = self._create_triangle_normals_arrows(transformed_mesh, arrow_scale=1.0, every_nth=10)
                
                # Remove old arrows and add new ones
                arrow_keys = [key for key in self.geometries.keys() if key.startswith(f"{model_name}_arrow_")]
                for arrow_key in arrow_keys:
                    self.vis.remove_geometry(self.geometries[arrow_key], reset_bounding_box=False)
                    del self.geometries[arrow_key]
                
                # Add updated arrows
                for j, arrow in enumerate(updated_arrows):
                    arrow_name = f"{model_name}_arrow_{j}"
                    self.vis.add_geometry(arrow, reset_bounding_box=False)
                    self.geometries[arrow_name] = arrow
        
        logger.debug(f"Updated visualization to step {step}")
    
    def _animation_loop(self):
        """Background animation loop"""
        frame_time = 1.0 / self.fps
        
        while self.playing:
            start_time = time.time()
            
            # Update to next step
            next_step = (self.current_step + 1) % self.max_steps
            self._update_step(next_step)
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
    
    def toggle_play_pause(self):
        """Toggle animation play/pause"""
        self.playing = not self.playing
        
        if self.playing:
            logger.info("Animation started")
            self.animation_thread = threading.Thread(target=self._animation_loop)
            self.animation_thread.daemon = True
            self.animation_thread.start()
        else:
            logger.info("Animation paused")
    
    def next_step(self):
        """Move to next time step"""
        next_step = min(self.current_step + 1, self.max_steps - 1)
        self._update_step(next_step)
        logger.info(f"Step: {self.current_step}")
    
    def previous_step(self):
        """Move to previous time step"""
        prev_step = max(self.current_step - 1, 0)
        self._update_step(prev_step)
        logger.info(f"Step: {self.current_step}")
    
    def restart_animation(self):
        """Restart animation from beginning"""
        self.playing = False
        time.sleep(0.1)
        self._update_step(0)
        logger.info("Animation restarted")
    
    def show_interactive(self):
        """Show interactive visualization"""
        logger.info("Starting interactive visualization...")
        logger.info("Controls:")
        logger.info("  Spacebar: Play/Pause animation")
        logger.info("  Arrow keys: Step forward/backward")  
        logger.info("  R: Restart animation")
        logger.info("  S: Save screenshot")
        logger.info("  Mouse: Rotate/pan/zoom view")
        
        # Setup visualization
        self._setup_visualization()
        
        # Start with first step
        self._update_step(0)
        
        # Start animation automatically
        self.toggle_play_pause()
        
        # Run visualization loop
        try:
            while True:
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Visualization interrupted by user")
        finally:
            # Cleanup
            self.playing = False
            if self.animation_thread and self.animation_thread.is_alive():
                self.animation_thread.join(timeout=1.0)
            
            self.vis.destroy_window()
            logger.info("Visualization closed")