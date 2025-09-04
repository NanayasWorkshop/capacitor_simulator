#!/usr/bin/env python3
"""
Capacitor Simulator - Main Entry Point
High-performance multi-sensor capacitor simulation with dynamic visualization

Usage:
    python main.py --data data/ --visualize
    python main.py --data data/ --sensors A,B --export-csv
    python main.py --help
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

def setup_logging(verbose=False, log_dir="logs"):
    """Setup logging with both file and console output"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_{timestamp}.log"
    
    # Configure logging level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Setup formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def validate_paths(args):
    """Validate that required paths exist"""
    errors = []
    
    # Check data directory
    data_path = Path(args.data)
    if not data_path.exists():
        errors.append(f"Data directory not found: {data_path}")
    else:
        # Check for required subdirectories
        fem_path = data_path / "fem"
        models_path = data_path / "models"
        
        if not fem_path.exists():
            errors.append(f"FEM data directory not found: {fem_path}")
        if not models_path.exists():
            errors.append(f"Models directory not found: {models_path}")
        
        # Check for CSV files
        required_csv = ['A1Sample.csv', 'A2Sample.csv', 'B1Sample.csv', 
                       'B2Sample.csv', 'C1Sample.csv', 'C2Sample.csv']
        for csv_file in required_csv:
            csv_path = fem_path / csv_file
            if not csv_path.exists():
                errors.append(f"Required CSV file not found: {csv_path}")
        
        # Check for model files
        required_models = ['A1_model.obj', 'A2_model.obj', 'B1_model.obj',
                          'B2_model.obj', 'C1_model.obj', 'C2_model.obj',
                          'stationary_negative.obj']
        for model_file in required_models:
            model_path = models_path / model_file
            if not model_path.exists():
                errors.append(f"Required model file not found: {model_path}")
    
    # Check output directory (create if needed)
    output_path = Path(args.output)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories
        (output_path / "results").mkdir(exist_ok=True)
        (output_path / "visualization").mkdir(exist_ok=True)
        (output_path / "cache").mkdir(exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory {output_path}: {e}")
    
    return errors

def check_dependencies():
    """Check that all required dependencies are available"""
    missing_deps = []
    
    # Check Python packages
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('open3d', 'open3d'),
        ('tqdm', 'tqdm')
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(package_name)
    
    # Check C++ extension (will be built later)
    try:
        import capacitor_cpp
        print("C++ extension already built")
    except ImportError:
        print("C++ extension not built yet - will build automatically")
    
    return missing_deps

def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Multi-Sensor Capacitor Simulator with Ray Tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data data/                           # Run with visualization
  %(prog)s --data data/ --no-visualize           # Compute only, no display
  %(prog)s --data data/ --sensors A,B            # Process only sensors A and B
  %(prog)s --data data/ --export-csv --export-png # Export results and images
  %(prog)s --data data/ --steps 0-100            # Process only steps 0 to 100
  %(prog)s --data data/ --config config/custom.yaml # Use custom configuration
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data', 
        required=True,
        help='Path to data directory containing fem/ and models/ subdirectories'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output',
        default='output',
        help='Output directory for results (default: output/)'
    )
    
    parser.add_argument(
        '--sensors',
        default='A,B,C',
        help='Comma-separated list of sensors to process (default: A,B,C)'
    )
    
    parser.add_argument(
        '--steps',
        help='Time steps to process, e.g., "0-100" or "0,10,20" (default: all)'
    )
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Show 3D visualization (default: True)'
    )
    viz_group.add_argument(
        '--no-visualize',
        action='store_false',
        dest='visualize',
        help='Disable visualization, compute only'
    )
    viz_group.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Animation frame rate (default: 30)'
    )
    viz_group.add_argument(
        '--realtime',
        action='store_true',
        help='Show real-time animation during computation'
    )
    
    # Export options
    export_group = parser.add_argument_group('Export Options')
    export_group.add_argument(
        '--export-csv',
        action='store_true',
        help='Export capacitance results to CSV files'
    )
    export_group.add_argument(
        '--export-png',
        action='store_true',
        help='Export screenshots of visualization'
    )
    export_group.add_argument(
        '--export-video',
        action='store_true',
        help='Export animation as MP4 video'
    )
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--threads',
        type=int,
        help='Number of threads for parallel processing (default: auto)'
    )
    perf_group.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching, force recomputation'
    )
    perf_group.add_argument(
        '--memory-limit',
        type=str,
        help='Memory limit, e.g., "4GB" (default: no limit)'
    )
    
    # Debug options
    debug_group = parser.add_argument_group('Debug Options')
    debug_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    debug_group.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    debug_group.add_argument(
        '--debug-rays',
        action='store_true',
        help='Export ray tracing debug information'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        help='Configuration file (YAML format)'
    )
    
    return parser

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              Multi-Sensor Capacitor Simulator               ║
    ║                   with Real-time Visualization               ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main application entry point"""
    # Print banner
    print_banner()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Log startup information
    logger.info("="*60)
    logger.info("Capacitor Simulator Starting")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    logger.info("="*60)
    
    try:
        # Check dependencies
        logger.info("Checking dependencies...")
        missing_deps = check_dependencies()
        if missing_deps:
            logger.error(f"Missing required packages: {missing_deps}")
            logger.error("Please install with: pip install -r requirements.txt")
            return 1
        logger.info("All dependencies available")
        
        # Validate paths
        logger.info("Validating input paths...")
        path_errors = validate_paths(args)
        if path_errors:
            logger.error("Path validation failed:")
            for error in path_errors:
                logger.error(f"  - {error}")
            return 1
        logger.info("All paths validated")
        
        # Check C++ extension
        logger.info("Checking C++ extension...")
        try:
            # Add DLL directory for Embree on Windows
            import os
            if os.path.exists("C:\\embree\\bin"):
                os.add_dll_directory("C:\\embree\\bin")
            
            import capacitor_cpp
            logger.info("C++ extension loaded successfully")
        except ImportError:
            logger.info("Building C++ extension...")
            import subprocess
            result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Failed to build C++ extension:")
                logger.error(result.stderr)
                return 1
            logger.info("C++ extension built successfully")
        
        # Import core modules (after C++ extension is ready)
        logger.info("Loading simulation modules...")
        from data_loader import DataLoader
        from simulator import CapacitorSimulator
        if args.visualize:
            from visualizer import CapacitorVisualizer
        
        # Initialize simulator
        logger.info("Initializing simulator...")
        simulator = CapacitorSimulator(
            data_path=args.data,
            output_path=args.output,
            sensors=args.sensors.split(','),
            use_cache=not args.no_cache,
            num_threads=args.threads
        )
        
        # Run simulation
        start_time = time.time()
        logger.info("Starting simulation...")
        
        results = simulator.run_simulation(
            steps=args.steps,
            debug_rays=args.debug_rays,
            profile=args.profile
        )
        
        compute_time = time.time() - start_time
        logger.info(f"Simulation completed in {compute_time:.2f} seconds")
        
        # Export results
        if args.export_csv:
            logger.info("Exporting CSV results...")
            simulator.export_results_csv(results, args.output)
            logger.info("CSV export completed")
        
        # Visualization
        if args.visualize and results:
            logger.info("Starting visualization...")
            visualizer = CapacitorVisualizer(
                results=results,
                fps=args.fps,
                export_png=args.export_png,
                export_video=args.export_video
            )
            visualizer.show_interactive()
        
        logger.info("="*60)
        logger.info("Simulation completed successfully")
        logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to: {args.output}")
        logger.info("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Simulation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())