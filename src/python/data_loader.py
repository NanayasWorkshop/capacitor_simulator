"""
Data Loader for Capacitor Simulator - Modified Version
A2, B2, C2 are now the stationary reference frames
A1, B1, C1 are the moving sensors that get transformations applied
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import math

logger = logging.getLogger(__name__)

class TransformationCalculator:
    """Calculate transformation matrices from displacement data"""
    
    def __init__(self, radius: float = 1.0):
        """
        Initialize transformation calculator
        
        Args:
            radius: Radius of the reference triangle (default: 1.0)
        """
        self.radius = radius
        self.reference_triangle = self._create_reference_triangle()
        
    def _create_reference_triangle(self) -> np.ndarray:
        """Create a perfect equilateral triangle with given radius in XY plane"""
        angle_offset = 2 * math.pi / 3  # 120 degrees
        points = []
        
        for i in range(3):
            angle = i * angle_offset
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            z = 0.0
            points.append([x, y, z])
            
        return np.array(points)
    
    def _calculate_center_and_normal(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate center and normal vector from 3 points"""
        points = np.array(points)
        
        # Center is average of points
        center = np.mean(points, axis=0)
        
        # Calculate normal using cross product
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        
        # Normalize
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length
        else:
            normal = np.array([0, 0, 1])  # Default to Z-up
            
        return center, normal
    
    def _create_transformation_matrix(self, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Create 4x4 transformation matrix from center and normal"""
        # Create rotation matrix to align Z-axis with normal
        z_axis = np.array([0, 0, 1])
        
        if np.allclose(normal, z_axis):
            # Already aligned
            rotation = np.eye(3)
        elif np.allclose(normal, -z_axis):
            # Opposite direction, rotate 180 degrees around X
            rotation = np.array([[ 1,  0,  0],
                               [ 0, -1,  0],
                               [ 0,  0, -1]])
        else:
            # General case: rotate z_axis to normal
            axis = np.cross(z_axis, normal)
            axis = axis / np.linalg.norm(axis)
            
            cos_angle = np.dot(z_axis, normal)
            sin_angle = np.linalg.norm(np.cross(z_axis, normal))
            
            # Rodrigues' rotation formula
            K = np.array([[     0, -axis[2],  axis[1]],
                         [ axis[2],      0, -axis[0]],
                         [-axis[1],  axis[0],     0]])
            
            rotation = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3] = center
        
        return transform
    
    def _transform_points(self, transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Transform a list of 3D points using a 4x4 matrix"""
        points = np.array(points)
        
        # Convert to homogeneous coordinates
        if points.shape[1] == 3:
            ones = np.ones((points.shape[0], 1))
            homogeneous_points = np.hstack([points, ones])
        else:
            homogeneous_points = points
        
        # Apply transformation
        transformed_homogeneous = np.dot(transformation_matrix, homogeneous_points.T).T
        
        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]
        
        return transformed_points
    
    def calculate_relative_transformation(self, points_reference: np.ndarray, points_moving: np.ndarray) -> np.ndarray:
        """
        Calculate transformation from reference frame to moving sensor's position
        
        FLIPPED APPROACH:
        - points_reference: The stationary sensor (A2, B2, or C2)
        - points_moving: The moving sensor (A1, B1, or C1)
        
        1. Make the reference sensor (A2/B2/C2) the stationary frame
        2. Calculate where the moving sensor (A1/B1/C1) is positioned relative to reference
        3. Return transformation matrix to apply to the moving sensor's mesh
        
        Args:
            points_reference: 3x3 array of reference sensor's triangle vertices (A2/B2/C2)
            points_moving: 3x3 array of moving sensor's triangle vertices (A1/B1/C1)
            
        Returns:
            4x4 transformation matrix to apply to moving sensor mesh
        """
        
        # Step 1: Calculate reference sensor's center and normal (this becomes our stationary frame)
        center_ref, normal_ref = self._calculate_center_and_normal(points_reference)
        
        # Step 2: Create transformation matrix: Reference → World
        # This aligns reference center with origin and reference normal with Z-axis
        T_ref_to_world = self._create_transformation_matrix(center_ref, normal_ref)
        
        # Step 3: Invert to get World → Reference frame
        # This is the "look through reference sensor's eyes" transformation
        T_world_to_ref = np.linalg.inv(T_ref_to_world)
        
        # Step 4: Transform moving sensor's points into reference frame
        # Now moving sensor is expressed relative to reference sensor's coordinate system
        moving_in_ref_frame = self._transform_points(T_world_to_ref, points_moving)
        
        # Step 5: Calculate moving sensor's center and normal in reference frame
        # This tells us where moving sensor is positioned relative to reference (which is now at origin)
        center_moving_in_ref, normal_moving_in_ref = self._calculate_center_and_normal(moving_in_ref_frame)
        
        # Step 6: Create transformation from reference origin to moving sensor's position
        # This represents: "Start at reference (origin), move to where moving sensor is"
        T_ref_to_moving_position = self._create_transformation_matrix(center_moving_in_ref, normal_moving_in_ref)
        
        return T_ref_to_moving_position

class CSVDataLoader:
    """Load and process CSV displacement data from Ansys"""
    
    def __init__(self):
        self.data = {}
        
    def load_csv_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load a single CSV file with robust error handling
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        logger.info(f"Loading CSV file: {filepath}")
        
        try:
            # Try different encodings and separators
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                for sep in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Valid if multiple columns
                            logger.debug(f"Successfully loaded with encoding={encoding}, sep='{sep}'")
                            break
                    except:
                        continue
                else:
                    continue
                break
            else:
                raise ValueError(f"Could not parse CSV file: {filepath}")
            
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def get_column_names(self, df: pd.DataFrame) -> List[str]:
        """Extract the 3 node column names from DataFrame"""
        columns = df.columns.tolist()
        
        # Group columns by node name
        nodes = {}
        for col in columns:
            if col.endswith('_X') or col.endswith('_Y') or col.endswith('_Z'):
                node_name = col[:-2]  # Remove _X, _Y, or _Z
                if node_name not in nodes:
                    nodes[node_name] = []
                nodes[node_name].append(col)
        
        # Should have exactly 3 nodes
        if len(nodes) != 3:
            raise ValueError(f"Expected 3 nodes, found {len(nodes)}: {list(nodes.keys())}")
        
        # Sort to ensure consistent order
        node_names = sorted(nodes.keys())
        
        # Verify each node has X, Y, Z components
        for node in node_names:
            expected_cols = [f"{node}_X", f"{node}_Y", f"{node}_Z"]
            if not all(col in columns for col in expected_cols):
                raise ValueError(f"Node {node} missing X/Y/Z columns")
        
        return node_names
    
    def extract_displacements(self, df: pd.DataFrame, row_index: int, node_names: List[str]) -> np.ndarray:
        """Extract XYZ displacements for given row and nodes"""
        displacements = []
        for node in node_names:
            x = df.iloc[row_index][f'{node}_X'] * 1000.0  # Convert meters to mm
            y = df.iloc[row_index][f'{node}_Y'] * 1000.0  # Convert meters to mm  
            z = df.iloc[row_index][f'{node}_Z'] * 1000.0  # Convert meters to mm
            displacements.append([x, y, z])
        return np.array(displacements)
    
    def load_sensor_data(self, data_folder: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Load CSV data for all sensors
        
        Args:
            data_folder: Path to folder containing CSV files
            
        Returns:
            Dictionary mapping sensor names to DataFrames
        """
        data_folder = Path(data_folder)
        fem_folder = data_folder / "fem"
        
        if not fem_folder.exists():
            raise FileNotFoundError(f"FEM data folder not found: {fem_folder}")
        
        sensors = {}
        sensor_files = ['A1Sample.csv', 'A2Sample.csv', 'B1Sample.csv', 
                       'B2Sample.csv', 'C1Sample.csv', 'C2Sample.csv']
        
        for sensor_file in sensor_files:
            sensor_path = fem_folder / sensor_file
            sensor_name = sensor_file.replace('Sample.csv', '')  # e.g., 'A1'
            
            try:
                df = self.load_csv_file(sensor_path)
                sensors[sensor_name] = df
                logger.info(f"Loaded sensor {sensor_name}: {len(df)} time steps")
            except Exception as e:
                logger.error(f"Failed to load sensor {sensor_name}: {e}")
                raise
        
        return sensors

class ModelLoader:
    """Load and process OBJ model files"""
    
    def __init__(self):
        self.models = {}
    
    def load_obj_file(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load OBJ file and return vertices and faces
        
        Args:
            filepath: Path to OBJ file
            
        Returns:
            Dictionary with 'vertices' and 'faces' arrays
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"OBJ file not found: {filepath}")
        
        logger.info(f"Loading OBJ file: {filepath}")
        
        vertices = []
        faces = []
        
        try:
            with open(filepath, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':
                        # Vertex: v x y z
                        if len(parts) >= 4:
                            try:
                                # Convert from meters to millimeters (assuming KeyShot format is already in mm)
                                KEYSHOT_UNITS_MM = True
                                if KEYSHOT_UNITS_MM:
                                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                else:
                                    x, y, z = float(parts[1]) * 1000, float(parts[2]) * 1000, float(parts[3]) * 1000
                                vertices.append([x, y, z])
                            except ValueError:
                                logger.warning(f"Invalid vertex on line {line_num}: {line}")
                    
                    elif parts[0] == 'f':
                        # Face: f v1 v2 v3 or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                        try:
                            # Extract vertex indices (handle different face formats)
                            vertex_indices = []
                            for part in parts[1:]:
                                # Handle formats: "1", "1/2", "1/2/3", "1//3"
                                vertex_idx = part.split('/')[0]
                                idx = int(vertex_idx)
                                # Convert to 0-based indexing
                                vertex_indices.append(idx - 1 if idx > 0 else len(vertices) + idx)
                            
                            # Only process triangular faces for now
                            if len(vertex_indices) >= 3:
                                faces.append(vertex_indices[:3])
                                
                                # Handle quads by creating second triangle
                                if len(vertex_indices) == 4:
                                    faces.append([vertex_indices[0], vertex_indices[2], vertex_indices[3]])
                                    
                        except (ValueError, IndexError):
                            logger.warning(f"Invalid face on line {line_num}: {line}")
            
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
            
            logger.info(f"Loaded {len(vertices)} vertices, {len(faces)} faces from {filepath}")
            
            return {
                'vertices': vertices,
                'faces': faces
            }
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def load_all_models(self, data_folder: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load all OBJ model files
        
        Args:
            data_folder: Path to data directory
            
        Returns:
            Dictionary mapping model names to model data
        """
        data_folder = Path(data_folder)
        models_folder = data_folder / "models"
        
        if not models_folder.exists():
            raise FileNotFoundError(f"Models folder not found: {models_folder}")
        
        models = {}
        model_files = ['A1_model.obj', 'A2_model.obj', 'B1_model.obj',
                      'B2_model.obj', 'C1_model.obj', 'C2_model.obj',
                      'stationary_negative.obj']
        
        for model_file in model_files:
            model_path = models_folder / model_file
            model_name = model_file.replace('_model.obj', '').replace('.obj', '')
            
            try:
                model_data = self.load_obj_file(model_path)
                models[model_name] = model_data
                logger.info(f"Loaded model {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return models

class DataLoader:
    """Main data loader class combining CSV and model loading"""
    
    def __init__(self, use_cache: bool = True, cache_dir: Union[str, Path] = "output/cache"):
        """
        Initialize data loader
        
        Args:
            use_cache: Whether to use caching for processed data
            cache_dir: Directory for cache files
        """
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_loader = CSVDataLoader()
        self.model_loader = ModelLoader()
        self.transform_calc = TransformationCalculator(radius=1.0)
        
    def load_all_data(self, data_folder: Union[str, Path]) -> Dict[str, any]:
        """
        Load all data (CSV and models) and compute transformations
        
        Args:
            data_folder: Path to data directory
            
        Returns:
            Dictionary containing all loaded and processed data
        """
        data_folder = Path(data_folder)
        logger.info(f"Loading all data from: {data_folder}")
        
        # Load CSV data
        logger.info("Loading CSV displacement data...")
        csv_data = self.csv_loader.load_sensor_data(data_folder)
        
        # Load model data
        logger.info("Loading OBJ model data...")
        model_data = self.model_loader.load_all_models(data_folder)
        
        # Calculate transformations for each sensor pair
        logger.info("Calculating transformation matrices...")
        transformations = {}
        
        for sensor_group in ['A', 'B', 'C']:
            reference_sensor = f"{sensor_group}2"  # A2, B2, C2 are now reference (stationary)
            moving_sensor = f"{sensor_group}1"    # A1, B1, C1 are now moving
            
            if reference_sensor in csv_data and moving_sensor in csv_data:
                logger.info(f"Processing sensor pair {sensor_group} (reference: {reference_sensor}, moving: {moving_sensor})")
                transforms = self._calculate_sensor_transformations(
                    csv_data[reference_sensor],  # Reference sensor (stationary)
                    csv_data[moving_sensor],     # Moving sensor 
                    sensor_group
                )
                transformations[sensor_group] = transforms
            else:
                logger.warning(f"Missing data for sensor pair {sensor_group}")
        
        result = {
            'csv_data': csv_data,
            'model_data': model_data,
            'transformations': transformations,
            'data_folder': data_folder
        }
        
        logger.info("Data loading completed successfully")
        return result
    
    def _calculate_sensor_transformations(self, df_reference: pd.DataFrame, df_moving: pd.DataFrame, sensor_group: str) -> np.ndarray:
        """
        Calculate transformation matrices for a sensor pair
        
        Args:
            df_reference: DataFrame for reference sensor (A2, B2, or C2)
            df_moving: DataFrame for moving sensor (A1, B1, or C1)
            sensor_group: Sensor group identifier ('A', 'B', or 'C')
        """
        # Get node names
        nodes_reference = self.csv_loader.get_column_names(df_reference)
        nodes_moving = self.csv_loader.get_column_names(df_moving)
        
        # Ensure same number of rows
        min_rows = min(len(df_reference), len(df_moving))
        
        transformations = []
        
        for i in range(min_rows):
            # Extract displacements
            displacements_reference = self.csv_loader.extract_displacements(df_reference, i, nodes_reference)
            displacements_moving = self.csv_loader.extract_displacements(df_moving, i, nodes_moving)
            
            # Calculate actual positions
            actual_reference = self.transform_calc.reference_triangle + displacements_reference
            actual_moving = self.transform_calc.reference_triangle + displacements_moving
            
            # Calculate relative transformation: reference is stationary, moving gets the transformation
            rel_transform = self.transform_calc.calculate_relative_transformation(actual_reference, actual_moving)
            transformations.append(rel_transform)
        
        logger.info(f"Generated {len(transformations)} transformation matrices for sensor {sensor_group} (reference: {sensor_group}2, moving: {sensor_group}1)")
        
        # Cache transformations if enabled
        if self.use_cache:
            cache_file = self.cache_dir / "transformations" / f"transformations_{sensor_group}.npy"
            cache_file.parent.mkdir(exist_ok=True)
            np.save(cache_file, np.array(transformations))
            logger.debug(f"Cached transformations to: {cache_file}")
        
        return np.array(transformations)