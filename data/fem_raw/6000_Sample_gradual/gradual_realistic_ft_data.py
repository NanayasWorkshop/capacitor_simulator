import pandas as pd
import numpy as np
import random
import math

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

def calculate_combined_load_fraction(fx, fy, fz, mx, my, mz):
    """Calculate the combined load as fraction of maximum possible"""
    force_fraction = np.sqrt((fx/sensor_limits['Fx'])**2 + 
                           (fy/sensor_limits['Fy'])**2 + 
                           (fz/sensor_limits['Fz'])**2)
    
    torque_fraction = np.sqrt((mx/sensor_limits['Mx'])**2 + 
                            (my/sensor_limits['My'])**2 + 
                            (mz/sensor_limits['Mz'])**2)
    
    return np.sqrt(force_fraction**2 + torque_fraction**2)

def probability_acceptance(combined_load_fraction):
    """
    Probability of accepting a load case based on combined load.
    Higher probability for lower loads, gradually decreasing for higher loads.
    """
    if combined_load_fraction <= 0.1:
        return 1.0  # Always accept very small loads
    elif combined_load_fraction <= 0.3:
        return 0.9  # High probability for small loads
    elif combined_load_fraction <= 0.5:
        return 0.7  # Good probability for medium loads
    elif combined_load_fraction <= 0.7:
        return 0.4  # Lower probability for high loads
    elif combined_load_fraction <= 0.85:
        return 0.2  # Low probability for very high loads
    else:
        return 0.05  # Very rare extreme loads

def generate_truly_random_value(axis_name):
    """Generate truly random values using various strategies"""
    
    max_val = sensor_limits[axis_name]
    
    # Choose random generation strategy
    strategies = [
        'pure_uniform',     # Completely uniform
        'exponential_decay', # Exponential falloff from zero
        'power_law',        # Power law distribution
        'random_spikes',    # Random spikes at various levels
        'clustered',        # Clustered around random points
        'bimodal',          # Two peaks
        'random_walk',      # Random walk from zero
        'sine_modulated',   # Sine wave modulated random
    ]
    
    strategy = random.choice(strategies)
    
    if strategy == 'pure_uniform':
        return np.random.uniform(-max_val, max_val)
    
    elif strategy == 'exponential_decay':
        # Exponential decay from zero, with random sign
        val = np.random.exponential(scale=max_val/4)
        return random.choice([-1, 1]) * min(val, max_val)
    
    elif strategy == 'power_law':
        # Power law distribution (more low values, fewer high values)
        exponent = random.uniform(0.3, 2.0)
        val = np.random.power(exponent) * max_val
        return random.choice([-1, 1]) * val
    
    elif strategy == 'random_spikes':
        # Random spikes at different intensity levels
        intensity_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        intensity = random.choice(intensity_levels)
        noise_factor = random.uniform(0.8, 1.2)
        val = intensity * max_val * noise_factor
        return random.choice([-1, 1]) * min(val, max_val)
    
    elif strategy == 'clustered':
        # Cluster around random center points
        center_fraction = random.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
        center = random.choice([-1, 1]) * center_fraction * max_val
        spread = random.uniform(0.1, 0.3) * max_val
        val = np.random.normal(center, spread)
        return np.clip(val, -max_val, max_val)
    
    elif strategy == 'bimodal':
        # Two peaks at random locations
        peak1 = random.uniform(0.1, 0.4) * max_val
        peak2 = random.uniform(0.6, 0.9) * max_val
        if random.random() < 0.5:
            val = np.random.normal(peak1, peak1*0.2)
        else:
            val = np.random.normal(peak2, peak2*0.2)
        return random.choice([-1, 1]) * np.clip(abs(val), 0, max_val)
    
    elif strategy == 'random_walk':
        # Random walk from zero
        steps = random.randint(5, 20)
        val = 0
        step_size = max_val / steps / 2
        for _ in range(steps):
            val += np.random.normal(0, step_size)
        return np.clip(val, -max_val, max_val)
    
    elif strategy == 'sine_modulated':
        # Sine wave modulated random values
        frequency = random.uniform(0.1, 2.0)
        amplitude = random.uniform(0.3, 0.8) * max_val
        phase = random.uniform(0, 2*math.pi)
        base = amplitude * math.sin(frequency + phase)
        noise = np.random.normal(0, amplitude * 0.3)
        return np.clip(base + noise, -max_val, max_val)

def generate_realistic_loads():
    """Generate loads with gradual falloff probability"""
    data = []
    attempts = 0
    max_attempts = n_total * 10  # Allow many attempts
    
    print(f"🎲 Generating {n_total} realistic load cases with gradual falloff...")
    
    while len(data) < n_total and attempts < max_attempts:
        attempts += 1
        
        # Generate completely random values for each axis
        fx = generate_truly_random_value('Fx')
        fy = generate_truly_random_value('Fy')
        fz = generate_truly_random_value('Fz')
        mx = generate_truly_random_value('Mx')
        my = generate_truly_random_value('My')
        mz = generate_truly_random_value('Mz')
        
        # Sometimes zero out some axes for more realistic combinations
        if random.random() < 0.15:  # 15% chance of zeroing some axes
            axes_to_zero = random.sample(['fx', 'fy', 'fz', 'mx', 'my', 'mz'], 
                                       random.randint(1, 3))
            if 'fx' in axes_to_zero: fx = 0
            if 'fy' in axes_to_zero: fy = 0
            if 'fz' in axes_to_zero: fz = 0
            if 'mx' in axes_to_zero: mx = 0
            if 'my' in axes_to_zero: my = 0
            if 'mz' in axes_to_zero: mz = 0
        
        # Calculate combined load
        combined_fraction = calculate_combined_load_fraction(fx, fy, fz, mx, my, mz)
        
        # Skip if combined load is too extreme (> 0.9)
        if combined_fraction > 0.9:
            continue
            
        # Use probability acceptance based on combined load
        acceptance_prob = probability_acceptance(combined_fraction)
        
        if random.random() < acceptance_prob:
            # Determine load type based on combined fraction (for coloring only)
            if combined_fraction <= 0.4:
                load_type = 'Normal'
            else:
                load_type = 'Strong'
                
            data.append([fx, fy, fz, mx, my, mz, load_type])
        
        if len(data) % 500 == 0 and len(data) > 0:
            current_avg_load = np.mean([calculate_combined_load_fraction(d[0], d[1], d[2], d[3], d[4], d[5]) 
                                      for d in data])
            print(f"  Generated {len(data)}/{n_total} cases (avg load: {current_avg_load:.3f})...")
    
    print(f"✅ Generated {len(data)} cases in {attempts} attempts")
    return data

# Generate the dataset
print("🌊 Generating truly random FT sensor data with gradual probability falloff...")
all_data = generate_realistic_loads()

# Shuffle thoroughly
for _ in range(10):  # Multiple shuffles
    random.shuffle(all_data)

# Create DataFrame
columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Load_Type']
df = pd.DataFrame(all_data, columns=columns)

# Add Test_ID
df.insert(0, 'Test_ID', range(1, len(df) + 1))

# Calculate magnitudes and combined load fraction
df['Force_Magnitude'] = np.sqrt(df['Fx']**2 + df['Fy']**2 + df['Fz']**2)
df['Torque_Magnitude'] = np.sqrt(df['Mx']**2 + df['My']**2 + df['Mz']**2)
df['Combined_Load_Fraction'] = [calculate_combined_load_fraction(row['Fx'], row['Fy'], row['Fz'], 
                                                               row['Mx'], row['My'], row['Mz']) 
                                for _, row in df.iterrows()]

# Save to CSV
filename = 'gradual_realistic_ft_sensor_data.csv'
df.to_csv(filename, index=False, float_format='%.2f')

# Analysis of the distribution
load_bins = {
    'Very Low (0-0.1)': len(df[df['Combined_Load_Fraction'] <= 0.1]),
    'Low (0.1-0.3)': len(df[(df['Combined_Load_Fraction'] > 0.1) & (df['Combined_Load_Fraction'] <= 0.3)]),
    'Medium (0.3-0.5)': len(df[(df['Combined_Load_Fraction'] > 0.3) & (df['Combined_Load_Fraction'] <= 0.5)]),
    'High (0.5-0.7)': len(df[(df['Combined_Load_Fraction'] > 0.5) & (df['Combined_Load_Fraction'] <= 0.7)]),
    'Very High (0.7-0.85)': len(df[(df['Combined_Load_Fraction'] > 0.7) & (df['Combined_Load_Fraction'] <= 0.85)]),
    'Extreme (>0.85)': len(df[df['Combined_Load_Fraction'] > 0.85])
}

# Display summary
print("\n" + "="*70)
print("🌊 GRADUAL REALISTIC FT SENSOR TEST DATA")
print("="*70)
print(f"📊 Total test cases generated: {len(df)}")

normal_count = len(df[df['Load_Type'] == 'Normal'])
strong_count = len(df[df['Load_Type'] == 'Strong'])
print(f"🟦 Normal loads (≤0.4): {normal_count} ({normal_count/len(df)*100:.1f}%)")
print(f"🟥 Strong loads (>0.4): {strong_count} ({strong_count/len(df)*100:.1f}%)")

print(f"\n🎯 GRADUAL DISTRIBUTION BY LOAD LEVEL:")
for level, count in load_bins.items():
    percentage = count / len(df) * 100
    bar = "█" * max(1, int(percentage / 2))
    print(f"   {level:20s}: {count:4d} ({percentage:5.1f}%) {bar}")

print(f"\n⚖️  COMBINED LOAD STATISTICS:")
print(f"   Range: {df['Combined_Load_Fraction'].min():.3f} - {df['Combined_Load_Fraction'].max():.3f}")
print(f"   Average: {df['Combined_Load_Fraction'].mean():.3f}")
print(f"   Median: {df['Combined_Load_Fraction'].median():.3f}")
print(f"   Std Dev: {df['Combined_Load_Fraction'].std():.3f}")

print(f"\n💪 FORCE STATISTICS:")
print(f"   Range: {df['Force_Magnitude'].min():.1f} - {df['Force_Magnitude'].max():.1f} N")
print(f"   Average: {df['Force_Magnitude'].mean():.1f} N")

print(f"\n🔄 TORQUE STATISTICS:")
print(f"   Range: {df['Torque_Magnitude'].min():.1f} - {df['Torque_Magnitude'].max():.1f} Nm")
print(f"   Average: {df['Torque_Magnitude'].mean():.1f} Nm")

print(f"\n📐 INDIVIDUAL AXIS RANGES:")
for axis in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
    unit = 'N' if axis.startswith('F') else 'Nm'
    axis_data = df[axis]
    print(f"   {axis}: {axis_data.min():.1f} to {axis_data.max():.1f} {unit} (std: {axis_data.std():.1f})")

print(f"\n🎲 RANDOMIZATION STRATEGIES USED:")
print("   - Pure uniform, exponential decay, power law")
print("   - Random spikes, clustered, bimodal")  
print("   - Random walk, sine-modulated distributions")
print("   - Gradual probability falloff (not harsh cutoff)")
print("   - 15% chance of zeroing some axes for realism")

print(f"\n📁 FILES CREATED:")
print(f"   📄 {filename}")
print(f"   🌊 {len(df)} data points with natural gradual distribution")

print(f"\n✅ Update your visualizer to use: '{filename}'")
print("   This should show much more realistic, non-bell-curve patterns!")