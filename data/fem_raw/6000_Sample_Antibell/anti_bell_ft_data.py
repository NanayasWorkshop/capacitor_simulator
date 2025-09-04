import pandas as pd
import numpy as np
import random
from scipy import stats

# Use true randomness
np.random.seed(None)
random.seed(None)

# Define the FT sensor maximum values
sensor_limits = {
    'Fx': 5000,    # N
    'Fy': 5000,    # N
    'Fz': 25000,   # N
    'Mx': 350,     # Nm
    'My': 350,     # Nm
    'Mz': 250      # Nm
}

n_total = 6000

def is_bell_curve(data, axis_name):
    """Detect if data distribution is bell-curve-like"""
    # Create histogram with 20 bins
    hist, bins = np.histogram(data, bins=20)
    
    # Check if center bins have significantly more data than edges
    center_start = len(hist) // 3
    center_end = 2 * len(hist) // 3
    center_sum = np.sum(hist[center_start:center_end])
    total_sum = np.sum(hist)
    
    center_ratio = center_sum / total_sum if total_sum > 0 else 0
    
    # Check skewness (should be close to 0 for bell curve)
    skewness = abs(stats.skew(data))
    
    # Bell curve detection: high center concentration + low skewness
    is_bell = (center_ratio > 0.5 and skewness < 1.5)
    
    print(f"  {axis_name}: center_ratio={center_ratio:.2f}, skewness={skewness:.2f} - {'🔔 Bell!' if is_bell else '✅ Even'}")
    
    return is_bell, center_ratio

def generate_even_distribution(axis_name, n_points):
    """Generate evenly distributed values for one axis"""
    max_val = sensor_limits[axis_name]
    
    # Use multiple strategies to ensure even distribution
    strategies = [
        'uniform_segments',    # Divide range into segments
        'pure_random',         # Pure uniform random
        'grid_with_noise',     # Grid points with noise
        'inverse_cumulative',  # Inverse cumulative for flatness
    ]
    
    # Generate points using mixed strategies
    points = []
    points_per_strategy = n_points // len(strategies)
    
    for strategy in strategies:
        strategy_points = []
        
        if strategy == 'uniform_segments':
            # Divide the range into segments and sample evenly
            segments = 20
            for i in range(points_per_strategy):
                segment = i % segments
                segment_min = -max_val + (segment * 2 * max_val / segments)
                segment_max = -max_val + ((segment + 1) * 2 * max_val / segments)
                point = random.uniform(segment_min, segment_max)
                strategy_points.append(point)
                
        elif strategy == 'pure_random':
            # Pure uniform distribution
            for i in range(points_per_strategy):
                point = random.uniform(-max_val, max_val)
                strategy_points.append(point)
                
        elif strategy == 'grid_with_noise':
            # Grid points with random noise
            grid_points = np.linspace(-max_val, max_val, points_per_strategy)
            for grid_point in grid_points:
                noise = random.uniform(-max_val*0.1, max_val*0.1)
                point = np.clip(grid_point + noise, -max_val, max_val)
                strategy_points.append(point)
                
        elif strategy == 'inverse_cumulative':
            # Use inverse cumulative distribution for perfect flatness
            for i in range(points_per_strategy):
                u = random.uniform(0, 1)
                point = -max_val + (2 * max_val * u)
                strategy_points.append(point)
        
        points.extend(strategy_points)
    
    # Fill remaining points with pure uniform
    while len(points) < n_points:
        point = random.uniform(-max_val, max_val)
        points.append(point)
    
    return points[:n_points]

def remove_center_bias(data, removal_fraction=0.3):
    """Remove data points from center region to reduce bell curve"""
    data_array = np.array(data)
    
    # Define center region (middle 40% of range)
    data_min, data_max = data_array.min(), data_array.max()
    data_range = data_max - data_min
    center_min = data_min + 0.3 * data_range
    center_max = data_max - 0.3 * data_range
    
    # Find center points
    center_mask = (data_array >= center_min) & (data_array <= center_max)
    center_indices = np.where(center_mask)[0]
    
    # Remove random fraction of center points
    num_to_remove = int(len(center_indices) * removal_fraction)
    if num_to_remove > 0:
        indices_to_remove = np.random.choice(center_indices, num_to_remove, replace=False)
        return indices_to_remove
    
    return []

def fix_bell_curves(df):
    """Fix bell curve patterns in each axis separately"""
    print(f"\n🔍 Checking for bell curves in each axis...")
    
    axes = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    max_iterations = 10
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        any_bell_detected = False
        
        for axis in axes:
            is_bell, center_ratio = is_bell_curve(df[axis].values, axis)
            
            if is_bell:
                any_bell_detected = True
                print(f"  🔄 Fixing {axis} distribution...")
                
                # Remove center-biased points
                indices_to_remove = remove_center_bias(df[axis].values, 
                                                     removal_fraction=0.25)
                
                if len(indices_to_remove) > 0:
                    # Generate new evenly distributed values for this axis only
                    new_values = generate_even_distribution(axis, len(indices_to_remove))
                    
                    # Replace the values for this axis
                    df.loc[indices_to_remove, axis] = new_values
                    print(f"    Replaced {len(indices_to_remove)} values in {axis}")
        
        if not any_bell_detected:
            print(f"  ✅ All axes have even distributions!")
            break
        
        if iteration == max_iterations - 1:
            print(f"  ⚠️  Reached maximum iterations. Some axes may still have slight bell curves.")
    
    return df

# Generate initial dataset
print(f"🎲 Generating {n_total} evenly distributed FT sensor data points...")

data = []
axes = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

print(f"📊 Generating even distributions for each axis...")

for i in range(n_total):
    row = []
    for axis in axes:
        # Generate one evenly distributed value for each axis
        max_val = sensor_limits[axis]
        value = random.uniform(-max_val, max_val)
        row.append(value)
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data, columns=axes)

print(f"✅ Generated {len(df)} initial data points")

# Fix any bell curve patterns
df = fix_bell_curves(df)

# Final shuffle to randomize order
df = df.sample(frac=1).reset_index(drop=True)

# Add Test_ID
df.insert(0, 'Test_ID', range(1, len(df) + 1))

# Calculate magnitudes for analysis
df['Force_Magnitude'] = np.sqrt(df['Fx']**2 + df['Fy']**2 + df['Fz']**2)
df['Torque_Magnitude'] = np.sqrt(df['Mx']**2 + df['My']**2 + df['Mz']**2)

# Save to CSV
filename = 'even_distribution_ft_data.csv'
df.to_csv(filename, index=False, float_format='%.2f')

# Final verification
print(f"\n" + "="*60)
print("🎯 FINAL DISTRIBUTION VERIFICATION")
print("="*60)

print(f"\n🔍 Final bell curve check:")
for axis in axes:
    is_bell, center_ratio = is_bell_curve(df[axis].values, axis)

print(f"\n📊 SUMMARY:")
print(f"   Total data points: {len(df)}")
print(f"   Each axis range:")
for axis in axes:
    unit = 'N' if axis.startswith('F') else 'Nm'
    axis_data = df[axis]
    print(f"     {axis}: {axis_data.min():.1f} to {axis_data.max():.1f} {unit}")

print(f"\n📐 MAGNITUDE STATISTICS:")
print(f"   Force magnitude: {df['Force_Magnitude'].min():.1f} - {df['Force_Magnitude'].max():.1f} N")
print(f"   Torque magnitude: {df['Torque_Magnitude'].min():.1f} - {df['Torque_Magnitude'].max():.1f} Nm")

print(f"\n📁 FILE CREATED: {filename}")
print(f"🎯 {len(df)} points with evenly distributed individual axes!")
print(f"✅ Ready for FEM simulation!")