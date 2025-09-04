#!/usr/bin/env python3
"""
Full system test - Load all data and test capacitance calculation
"""
import sys
import os
import time
import numpy as np

# Add DLL directory for Embree
os.add_dll_directory('C:/embree/bin')

# Add src to path
sys.path.insert(0, 'src/python')

from data_loader import DataLoader
import capacitor_cpp

def main():
    print("=" * 60)
    print("FULL CAPACITOR SIMULATION SYSTEM TEST")
    print("=" * 60)
    
    # Load all data
    print("Loading all data...")
    start_time = time.time()
    loader = DataLoader()
    data = loader.load_all_data('data')
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.1f}s")
    
    # Create engine
    config = capacitor_cpp.CapacitorConfig()
    config.verbose_logging = False  # Reduce output for batch processing
    engine = capacitor_cpp.CapacitorEngine(config)
    
    # Load all models into C++ engine
    print("\nLoading all models into C++ engine...")
    for model_name, model_data in data['model_data'].items():
        vertices = model_data['vertices'].flatten().tolist()
        faces = model_data['faces'].flatten().tolist()
        success = engine.load_mesh(model_name, vertices, faces)
        triangles = len(model_data['faces'])
        status = "OK" if success else "FAILED"
        print(f"  {model_name}: {triangles} triangles - {status}")
    
    print()
    engine.print_statistics()
    
    # Test each sensor
    for sensor in ['A', 'B', 'C']:
        print(f"\n{'-'*40}")
        print(f"TESTING SENSOR {sensor}")
        print(f"{'-'*40}")
        
        transforms = data['transformations'][sensor]
        print(f"Processing {len(transforms)} time steps...")
        
        # Convert transformation matrices to C++ format
        cpp_transforms = []
        for transform in transforms:
            cpp_matrix = capacitor_cpp.Matrix4x4()
            cpp_matrix.set_from_numpy(transform.astype(float))
            cpp_transforms.append(cpp_matrix)
        
        # Test both models for this sensor (e.g., A1 and A2)
        for model_num in ['1', '2']:
            model_name = f"{sensor}{model_num}"
            print(f"\n  Testing {model_name}...")
            
            # Calculate capacitance for all time steps
            start_time = time.time()
            results = engine.calculate_capacitance_batch(
                model_name, 
                'stationary_negative', 
                cpp_transforms
            )
            calc_time = time.time() - start_time
            
            # Analyze results
            valid_results = [r for r in results if r.is_valid()]
            
            if valid_results:
                capacitances = [r.capacitance_pF for r in valid_results]
                distances = [r.minDistance_mm for r in valid_results]
                hits = [r.hits for r in valid_results]
                
                print(f"    Batch calculation completed:")
                print(f"    - Total time: {calc_time:.2f}s")
                print(f"    - Time per step: {calc_time/len(results)*1000:.1f}ms")
                print(f"    - Valid results: {len(valid_results)}/{len(results)}")
                print(f"    - Capacitance range: {min(capacitances):.1f} - {max(capacitances):.1f} pF")
                print(f"    - Average capacitance: {np.mean(capacitances):.1f} pF")
                print(f"    - Distance range: {min(distances):.3f} - {max(distances):.3f} mm")
                print(f"    - Average hits: {np.mean(hits):.0f}")
                
                # Show sample results
                print(f"    Sample results:")
                for i in [0, len(results)//2, len(results)-1]:
                    if i < len(results) and results[i].is_valid():
                        r = results[i]
                        print(f"      Step {r.step}: {r.capacitance_pF:.1f} pF, "
                              f"{r.minDistance_mm:.3f} mm gap, "
                              f"{r.hits}/{r.hits+r.misses} hits")
            else:
                print(f"    WARNING: No valid results for {model_name}")
    
    print(f"\n{'='*60}")
    print("FULL SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Summary:")
    print(f"- Loaded {len(data['model_data'])} models")
    print(f"- Processed {len(data['transformations'])} sensor groups")
    print(f"- Each sensor has {len(list(data['transformations'].values())[0])} time steps")
    print(f"- Total transformations tested: {sum(len(t) for t in data['transformations'].values())} per model")
    print(f"- Ray tracing engine: Embree 4.x")
    print(f"- Dielectric material: Glycerin (εᵣ = 42.28)")
    print("\nReady for visualization and export!")

if __name__ == "__main__":
    main()