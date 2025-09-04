"""
Setup script for Capacitor Simulator C++ extensions
Builds the high-performance ray tracing engine with Embree integration
"""
import os
import sys
import platform
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension

# Detect platform and set appropriate paths
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_LINUX = SYSTEM == "Linux"
IS_MAC = SYSTEM == "Darwin"

# Default Embree installation paths by platform
DEFAULT_EMBREE_PATHS = {
    "Windows": [
        r"C:\embree",
        r"C:\Program Files\Intel\Embree4",
        r"C:\embree4",
        Path.home() / "embree"
    ],
    "Linux": [
        "/usr/local",
        "/opt/embree",
        Path.home() / "embree"
    ],
    "Darwin": [  # macOS
        "/usr/local",
        "/opt/homebrew",
        "/opt/embree"
    ]
}

def find_embree_installation():
    """Auto-detect Embree installation directory"""
    # Check environment variable first
    embree_root = os.environ.get('EMBREE_ROOT')
    if embree_root and Path(embree_root).exists():
        print(f"Using Embree from environment: {embree_root}")
        return Path(embree_root)
    
    # Check default paths for current platform
    possible_paths = DEFAULT_EMBREE_PATHS.get(SYSTEM, [])
    
    for path in possible_paths:
        path = Path(path)
        include_path = path / "include" / "embree4"
        lib_path = path / "lib"
        
        # Check if both include and lib directories exist
        if include_path.exists() and lib_path.exists():
            print(f"Found Embree installation: {path}")
            return path
    
    # Last resort: check if embree4 headers are in system include paths
    if IS_LINUX or IS_MAC:
        system_include = Path("/usr/include/embree4")
        if system_include.exists():
            print("Using system-installed Embree")
            return Path("/usr")
    
    return None

def get_embree_config():
    """Get Embree include paths, library paths, and libraries"""
    embree_root = find_embree_installation()
    
    if not embree_root:
        print("ERROR: Embree installation not found!")
        print("Please install Embree 4.x or set EMBREE_ROOT environment variable")
        print("Download from: https://github.com/embree/embree/releases")
        sys.exit(1)
    
    include_dirs = [str(embree_root / "include")]
    library_dirs = [str(embree_root / "lib")]
    
    # Platform-specific library names
    if IS_WINDOWS:
        libraries = ["embree4"]
    else:
        libraries = ["embree4"]
    
    return include_dirs, library_dirs, libraries

def get_compiler_flags():
    """Get platform-specific compiler flags for optimization"""
    if IS_WINDOWS:
        # Force MinGW/GCC flags for Windows (since user has MinGW)
        compile_args = ["-O3", "-std=c++17", "-march=native"]
        link_args = []
    else:
        # GCC/Clang flags for Linux/Mac
        compile_args = ["-O3", "-std=c++17", "-march=native", "-ffast-math"]
        link_args = ["-Wl,-rpath,$ORIGIN"]
    
    return compile_args, link_args

def create_extension():
    """Create the pybind11 extension module"""
    # Get Embree configuration
    include_dirs, library_dirs, libraries = get_embree_config()
    compile_args, link_args = get_compiler_flags()
    
    # Source files
    cpp_files = [
        "src/cpp/capacitor_engine.cpp",
        "src/cpp/python_bindings.cpp"
    ]
    
    # Check that source files exist
    for cpp_file in cpp_files:
        if not Path(cpp_file).exists():
            print(f"ERROR: Source file not found: {cpp_file}")
            sys.exit(1)
    
    # Create extension
    ext = Pybind11Extension(
        "capacitor_cpp",  # Module name
        sources=cpp_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        cxx_std=17,  # Require C++17
    )
    
    print("Extension configuration:")
    print(f"  Sources: {cpp_files}")
    print(f"  Include dirs: {include_dirs}")
    print(f"  Library dirs: {library_dirs}")
    print(f"  Libraries: {libraries}")
    print(f"  Compile args: {compile_args}")
    
    return ext

# Create the extension
try:
    capacitor_extension = create_extension()
except SystemExit:
    raise
except Exception as e:
    print(f"ERROR: Failed to create extension: {e}")
    sys.exit(1)

# Setup configuration
setup(
    name="capacitor_simulator",
    version="0.1.0",
    author="Capacitor Simulation Team",
    description="High-performance capacitor simulation with ray tracing",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    ext_modules=[capacitor_extension],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pybind11>=2.10.0"
    ],
)

if __name__ == "__main__":
    print("Capacitor Simulator C++ Extension Builder")
    print("=" * 50)
    print(f"Platform: {SYSTEM}")
    print(f"Python: {sys.version}")
    print("=" * 50)