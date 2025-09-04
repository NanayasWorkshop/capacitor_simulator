"""
Capacitor Simulator - Main simulation controller
Orchestrates data loading, computation, and result export
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from data_loader import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class SimulationResults:
    """Container for all simulation results"""
    sensor_results: Dict[str, Dict[str, List[Dict[str, Any]]]]  # sensor -> model -> results
    computation_time: float
    total_steps: int
    total_calculations: int
    data_info: Dict[str, Any]
    
    def get_sensor_capacitances(self, sensor_name: str, model_name: str) -> List[float]:
        """Get capacitance values for a specific sensor model"""
        if sensor_name in self.sensor_results and model_name in self.sensor_results[sensor_name]:
            return [r['capacitance_pF'] for r in self.sensor_results[sensor_name][model_name]]
        return []
    
    def get_all_results_flat(self) -> List[Dict[str, Any]]:
        """Get all results as a flat list with sensor/model identifiers"""
        flat_results = []
        for sensor, models in self.sensor_results.items():
            for model, results in models.items():
                for result in results:
                    flat_result = result.copy()
                    flat_result['sensor'] = sensor
                    flat_result['model'] = model
                    flat_results.append(flat_result)
        return flat_results

class CapacitorSimulator:
    """Main simulation controller"""
    
    def __init__(self, data_path: Union[str, Path], output_path: Union[str, Path], 
                 sensors: List[str] = None, use_cache: bool = True, num_threads: int = None):
        """
        Initialize the capacitor simulator
        
        Args:
            data_path: Path to data directory
            output_path: Path to output directory 
            sensors: List of sensor names to process (default: ['A', 'B', 'C'])
            use_cache: Whether to use cached data
            num_threads: Number of threads for parallel processing
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.sensors = sensors if sensors else ['A', 'B', 'C']
        self.use_cache = use_cache
        self.num_threads = num_threads
        
        self.data_loader = None
        self.loaded_data = None
        self.cpp_engine = None
        
        # Ensure output directories exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "results").mkdir(exist_ok=True)
        
        logger.info(f"Simulator initialized:")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Output path: {self.output_path}")
        logger.info(f"  Sensors: {self.sensors}")
        logger.info(f"  Use cache: {self.use_cache}")
    
    def _initialize_cpp_engine(self):
        """Initialize the C++ ray tracing engine"""
        try:
            # Add DLL directory for Embree
            if os.path.exists("C:\\embree\\bin"):
                os.add_dll_directory("C:\\embree\\bin")
            
            import capacitor_cpp
            
            # Create engine configuration
            config = capacitor_cpp.CapacitorConfig()
            config.verbose_logging = False
            config.max_ray_distance = 5.0  # 5mm max distance
            config.bidirectional_rays = True
            config.ray_density = 1  # Use all triangles
            
            self.cpp_engine = capacitor_cpp.CapacitorEngine(config)
            logger.info("C++ ray tracing engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize C++ engine: {e}")
            raise
    
    def _load_data(self):
        """Load all CSV and model data"""
        logger.info("Loading simulation data...")
        start_time = time.time()
        
        self.data_loader = DataLoader(use_cache=self.use_cache)
        self.loaded_data = self.data_loader.load_all_data(self.data_path)
        
        load_time = time.time() - start_time
        logger.info(f"Data loaded in {load_time:.2f}s")
        
        # Log data summary
        logger.info("Data summary:")
        logger.info(f"  CSV files: {list(self.loaded_data['csv_data'].keys())}")
        logger.info(f"  Models: {list(self.loaded_data['model_data'].keys())}")
        logger.info(f"  Transformations: {list(self.loaded_data['transformations'].keys())}")
        
        for sensor, transforms in self.loaded_data['transformations'].items():
            logger.info(f"  Sensor {sensor}: {transforms.shape[0]} time steps")
    
    def _load_models_into_engine(self):
        """Load all 3D models into the C++ engine"""
        logger.info("Loading models into C++ engine...")
        
        models_loaded = 0
        for model_name, model_data in self.loaded_data['model_data'].items():
            vertices = model_data['vertices'].flatten().astype(float).tolist()
            faces = model_data['faces'].flatten().astype(int).tolist()
            
            success = self.cpp_engine.load_mesh(model_name, vertices, faces)
            
            if success:
                triangles = len(model_data['faces'])
                logger.info(f"  {model_name}: {triangles} triangles")
                models_loaded += 1
            else:
                logger.error(f"  Failed to load {model_name}")
                
        logger.info(f"Loaded {models_loaded} models into ray tracing engine")
        self.cpp_engine.print_statistics()
    
    def _convert_transformations(self, transforms: np.ndarray):
        """Convert numpy transformation matrices to C++ format"""
        import capacitor_cpp
        
        cpp_transforms = []
        for transform in transforms:
            cpp_matrix = capacitor_cpp.Matrix4x4()
            cpp_matrix.set_from_numpy(transform.astype(float))
            cpp_transforms.append(cpp_matrix)
        
        return cpp_transforms
    
    def _process_sensor(self, sensor_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process all models for a single sensor"""
        logger.info(f"Processing sensor {sensor_name}...")
        
        sensor_results = {}
        
        # Get transformations for this sensor
        transforms = self.loaded_data['transformations'][sensor_name]
        cpp_transforms = self._convert_transformations(transforms)
        
        logger.info(f"  Processing {len(transforms)} time steps")
        
        # Process both models for this sensor (e.g., A1 and A2)
        for model_num in ['1', '2']:
            model_name = f"{sensor_name}{model_num}"
            logger.info(f"  Processing {model_name}...")
            
            try:
                # Calculate capacitance for all time steps
                start_time = time.time()
                cpp_results = self.cpp_engine.calculate_capacitance_batch(
                    model_name, 
                    'stationary_negative', 
                    cpp_transforms
                )
                calc_time = time.time() - start_time
                
                # Convert C++ results to Python dictionaries
                python_results = []
                for cpp_result in cpp_results:
                    result = {
                        'step': cpp_result.step,
                        'capacitance_pF': cpp_result.capacitance_pF,
                        'minDistance_mm': cpp_result.minDistance_mm,
                        'maxDistance_mm': cpp_result.maxDistance_mm,
                        'totalArea_mm2': cpp_result.totalArea_mm2,
                        'hits': cpp_result.hits,
                        'misses': cpp_result.misses,
                        'computation_time_ms': cpp_result.computation_time_ms,
                        'rays_traced': cpp_result.rays_traced,
                        'translation_x': cpp_result.translation.x,
                        'translation_y': cpp_result.translation.y,
                        'translation_z': cpp_result.translation.z,
                        'valid': cpp_result.is_valid()
                    }
                    python_results.append(result)
                
                sensor_results[model_name] = python_results
                
                # Log summary
                valid_results = [r for r in python_results if r['valid']]
                if valid_results:
                    capacitances = [r['capacitance_pF'] for r in valid_results]
                    logger.info(f"    {len(valid_results)}/{len(python_results)} valid results")
                    logger.info(f"    Capacitance range: {min(capacitances):.1f} - {max(capacitances):.1f} pF")
                    logger.info(f"    Computation time: {calc_time:.2f}s ({calc_time/len(python_results)*1000:.1f}ms per step)")
                else:
                    logger.warning(f"    No valid results for {model_name}")
                    
            except Exception as e:
                logger.error(f"  Failed to process {model_name}: {e}")
                sensor_results[model_name] = []
        
        return sensor_results
    
    def run_simulation(self, steps: str = None, debug_rays: bool = False, profile: bool = False) -> SimulationResults:
        """
        Run the complete simulation
        
        Args:
            steps: Time steps to process (e.g., "0-100" or "0,10,20")
            debug_rays: Whether to export ray data for debugging
            profile: Whether to enable performance profiling
            
        Returns:
            SimulationResults object containing all results
        """
        logger.info("Starting capacitor simulation...")
        total_start_time = time.time()
        
        # Initialize components
        self._initialize_cpp_engine()
        self._load_data()
        self._load_models_into_engine()
        
        # Process each sensor
        all_sensor_results = {}
        total_calculations = 0
        
        for sensor_name in self.sensors:
            if sensor_name in self.loaded_data['transformations']:
                sensor_results = self._process_sensor(sensor_name)
                all_sensor_results[sensor_name] = sensor_results
                
                # Count calculations
                for model_results in sensor_results.values():
                    total_calculations += len(model_results)
            else:
                logger.warning(f"No transformation data found for sensor {sensor_name}")
        
        total_time = time.time() - total_start_time
        
        # Create results object
        results = SimulationResults(
            sensor_results=all_sensor_results,
            computation_time=total_time,
            total_steps=len(list(self.loaded_data['transformations'].values())[0]) if self.loaded_data['transformations'] else 0,
            total_calculations=total_calculations,
            data_info={
                'data_path': str(self.data_path),
                'sensors_processed': list(all_sensor_results.keys()),
                'models_loaded': len(self.loaded_data['model_data']),
                'total_transformations': sum(len(t) for t in self.loaded_data['transformations'].values())
            }
        )
        
        logger.info("Simulation completed!")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Total calculations: {total_calculations}")
        logger.info(f"  Average time per calculation: {total_time/total_calculations*1000:.1f}ms")
        
        return results
    
    def export_csv(self, output_path: Union[str, Path] = None, results: SimulationResults = None):
        """Export results to CSV files (legacy method name for compatibility)"""
        if results is None:
            logger.error("No results provided to export_csv. Call run_simulation() first.")
            return
        
        return self.export_results_csv(results, output_path)
    
    def export_results_csv(self, results: SimulationResults, output_path: Union[str, Path] = None):
        """Export simulation results to CSV files"""
        if output_path is None:
            output_path = self.output_path / "results"
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting results to CSV files...")
        
        files_created = []
        
        # Export individual sensor/model results
        for sensor, models in results.sensor_results.items():
            for model, model_results in models.items():
                if model_results:
                    # Create DataFrame
                    df = pd.DataFrame(model_results)
                    
                    # Export to CSV
                    filename = f"capacitance_{model}.csv"
                    filepath = output_path / filename
                    df.to_csv(filepath, index=False)
                    
                    files_created.append(filename)
                    logger.info(f"  Created {filename} ({len(model_results)} rows)")
        
        # Export combined summary
        flat_results = results.get_all_results_flat()
        if flat_results:
            summary_df = pd.DataFrame(flat_results)
            summary_file = output_path / "capacitance_summary_all_sensors.csv"
            summary_df.to_csv(summary_file, index=False)
            files_created.append("capacitance_summary_all_sensors.csv")
            logger.info(f"  Created summary file ({len(flat_results)} total rows)")
        
        # Export metadata
        metadata = {
            'simulation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': results.computation_time,
            'total_calculations': results.total_calculations,
            'sensors_processed': results.data_info['sensors_processed'],
            'files_created': files_created
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_file = output_path / "simulation_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        logger.info(f"Results exported: {len(files_created)} data files + metadata")
        return files_created