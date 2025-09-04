import pandas as pd
import numpy as np
from scipy import stats
import random

# Set random seed for reproducibility (remove this line for completely random each time)
np.random.seed(None)  # Use system time for true randomness
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

# Define thresholds for combined loading
normal_combined_limit = 0.4      # Normal operations stay under 40% of combined max
strong_combined_limit = 0.8      # Strong cases can go up to 80% of combined max
n_total = 6000
n_normal = int(n_total * 0.8)    # 80% = 4800
n_strong = int(n_total * 0.2)    # 20% = 1200

def calculate_combined_load_fraction(fx, fy, fz, mx, my, mz):
    """Calculate the combined load as fraction of maximum possible"""
    force_fraction = np.sqrt((fx/sensor_limits['Fx'])**2 + 
                           (fy/sensor_limits['Fy'])**2 + 
                           (fz/sensor_limits['Fz'])**2)
    
    torque_fraction = np.sqrt((mx/sensor_limits['Mx'])**2 + 
                            (my/sensor_limits['My'])**2 + 
                            (mz/sensor_limits['Mz'])**2)
    
    return np.sqrt(force_fraction**2 + torque_fraction**2)

def random_distribution_generator(min_val, max_val, size=1):
    """Generate truly random values using various distributions"""
    distribution_types = [
        'uniform',      # Flat distribution
        'exponential',  # Exponential decay
        'beta_low',     # Skewed towards low values
        'beta_high',    # Skewed towards high values
        'triangular',   # Triangular distribution
        'gamma',        # Gamma distribution
        'weibull',      # Weibull distribution
        'lognormal',    # Log-normal distribution
    ]
    
    values = []
    for _ in range(size):
        dist_type = random.choice(distribution_types)
        
        if dist_type == 'uniform':
            val = np.random.uniform(min_val, max_val)
        elif dist_type == 'exponential':
            # Exponential distribution, scaled and shifted
            val = np.random.exponential(scale=(max_val-min_val)/3) + min_val
            val = np.clip(val, min_val, max_val)
        elif dist_type == 'beta_low':
            # Beta distribution skewed towards low values
            val = np.random.beta(0.5, 2) * (max_val - min_val) + min_val
        elif dist_type == 'beta_high':
            # Beta distribution skewed towards high values
            val = np.random.beta(2, 0.5) * (max_val - min_val) + min_val
        elif dist_type == 'triangular':
            # Triangular distribution with random mode
            mode = np.random.uniform(min_val, max_val)
            val = np.random.triangular(min_val, mode, max_val)
        elif dist_type == 'gamma':
            # Gamma distribution
            val = np.random.gamma(2, scale=(max_val-min_val)/4) + min_val
            val = np.clip(val, min_val, max_val)
        elif dist_type == 'weibull':
            # Weibull distribution
            val = np.random.weibull(1.5) * (max_val - min_val) + min_val
            val = np.clip(val, min_val, max_val)
        elif dist_type == 'lognormal':
            # Log-normal distribution
            val = np.random.lognormal(0, 0.5) * (max_val - min_val) / 3 + min_val
            val = np.clip(val, min_val, max_val)
        
        values.append(val)
    
    return values[0] if size == 1 else values

def generate_realistic_normal_loads():
    """Generate realistic normal operation loads with truly random distributions"""
    data = []
    attempts = 0
    max_attempts = n_normal * 3  # Allow more attempts
    
    print(f"Generating {n_normal} normal load cases...")
    
    while len(data) < n_normal and attempts < max_attempts:
        attempts += 1
        
        # Use random distributions instead of normal distributions
        # Forces - mix of small and medium values
        fx = random_distribution_generator(-sensor_limits['Fx'] * 0.3, sensor_limits['Fx'] * 0.3)
        fy = random_distribution_generator(-sensor_limits['Fy'] * 0.3, sensor_limits['Fy'] * 0.3)
        
        # Fz can be larger (gravity effects, but still random)
        fz_range = random.choice([0.2, 0.3, 0.4])  # Random range selection
        fz = random_distribution_generator(-sensor_limits['Fz'] * 0.1, sensor_limits['Fz'] * fz_range)
        
        # Torques - generally smaller but random
        mx = random_distribution_generator(-sensor_limits['Mx'] * 0.25, sensor_limits['Mx'] * 0.25)
        my = random_distribution_generator(-sensor_limits['My'] * 0.25, sensor_limits['My'] * 0.25)
        mz = random_distribution_generator(-sensor_limits['Mz'] * 0.2, sensor_limits['Mz'] * 0.2)
        
        # Check combined load
        combined_fraction = calculate_combined_load_fraction(fx, fy, fz, mx, my, mz)
        
        if combined_fraction <= normal_combined_limit:
            data.append([fx, fy, fz, mx, my, mz, 'Normal'])
        elif combined_fraction <= normal_combined_limit * 1.1:  # Allow slight overage and scale down
            scale_factor = normal_combined_limit / combined_fraction * 0.95
            data.append([fx*scale_factor, fy*scale_factor, fz*scale_factor, 
                        mx*scale_factor, my*scale_factor, mz*scale_factor, 'Normal'])
        
        if len(data) % 500 == 0:
            print(f"  Generated {len(data)}/{n_normal} normal cases...")
    
    print(f"✅ Generated {len(data)} normal cases in {attempts} attempts")
    return data

def generate_realistic_strong_loads():
    """Generate strong loads with truly random distributions"""
    data = []
    attempts = 0
    max_attempts = n_strong * 5  # Allow more attempts for strong loads
    
    print(f"Generating {n_strong} strong load cases...")
    
    while len(data) < n_strong and attempts < max_attempts:
        attempts += 1
        
        # Strong loads - use wider ranges and more random distributions
        load_scenario = random.choice(['force_dominant', 'torque_dominant', 'mixed', 'extreme'])
        
        if load_scenario == 'force_dominant':
            # High forces, lower torques
            fx = random_distribution_generator(-sensor_limits['Fx'] * 0.8, sensor_limits['Fx'] * 0.8)
            fy = random_distribution_generator(-sensor_limits['Fy'] * 0.8, sensor_limits['Fy'] * 0.8)
            fz = random_distribution_generator(-sensor_limits['Fz'] * 0.9, sensor_limits['Fz'] * 0.9)
            mx = random_distribution_generator(-sensor_limits['Mx'] * 0.3, sensor_limits['Mx'] * 0.3)
            my = random_distribution_generator(-sensor_limits['My'] * 0.3, sensor_limits['My'] * 0.3)
            mz = random_distribution_generator(-sensor_limits['Mz'] * 0.2, sensor_limits['Mz'] * 0.2)
            
        elif load_scenario == 'torque_dominant':
            # High torques, lower forces
            fx = random_distribution_generator(-sensor_limits['Fx'] * 0.4, sensor_limits['Fx'] * 0.4)
            fy = random_distribution_generator(-sensor_limits['Fy'] * 0.4, sensor_limits['Fy'] * 0.4)
            fz = random_distribution_generator(-sensor_limits['Fz'] * 0.5, sensor_limits['Fz'] * 0.5)
            mx = random_distribution_generator(-sensor_limits['Mx'] * 0.8, sensor_limits['Mx'] * 0.8)
            my = random_distribution_generator(-sensor_limits['My'] * 0.8, sensor_limits['My'] * 0.8)
            mz = random_distribution_generator(-sensor_limits['Mz'] * 0.7, sensor_limits['Mz'] * 0.7)
            
        elif load_scenario == 'mixed':
            # Balanced strong loads
            fx = random_distribution_generator(-sensor_limits['Fx'] * 0.6, sensor_limits['Fx'] * 0.6)
            fy = random_distribution_generator(-sensor_limits['Fy'] * 0.6, sensor_limits['Fy'] * 0.6)
            fz = random_distribution_generator(-sensor_limits['Fz'] * 0.7, sensor_limits['Fz'] * 0.7)
            mx = random_distribution_generator(-sensor_limits['Mx'] * 0.6, sensor_limits['Mx'] * 0.6)
            my = random_distribution_generator(-sensor_limits['My'] * 0.6, sensor_limits['My'] * 0.6)
            mz = random_distribution_generator(-sensor_limits['Mz'] * 0.5, sensor_limits['Mz'] * 0.5)
            
        else:  # extreme
            # Very high loads on random axes
            axes_to_max = random.sample(['fx', 'fy', 'fz', 'mx', 'my', 'mz'], random.randint(1, 3))
            fx = random_distribution_generator(-sensor_limits['Fx'] * (0.9 if 'fx' in axes_to_max else 0.3), 
                                            sensor_limits['Fx'] * (0.9 if 'fx' in axes_to_max else 0.3))
            fy = random_distribution_generator(-sensor_limits['Fy'] * (0.9 if 'fy' in axes_to_max else 0.3), 
                                            sensor_limits['Fy'] * (0.9 if 'fy' in axes_to_max else 0.3))
            fz = random_distribution_generator(-sensor_limits['Fz'] * (0.9 if 'fz' in axes_to_max else 0.4), 
                                            sensor_limits['Fz'] * (0.9 if 'fz' in axes_to_max else 0.4))
            mx = random_distribution_generator(-sensor_limits['Mx'] * (0.8 if 'mx' in axes_to_max else 0.2), 
                                            sensor_limits['Mx'] * (0.8 if 'mx' in axes_to_max else 0.2))
            my = random_distribution_generator(-sensor_limits['My'] * (0.8 if 'my' in axes_to_max else 0.2), 
                                            sensor_limits['My'] * (0.8 if 'my' in axes_to_max else 0.2))
            mz = random_distribution_generator(-sensor_limits['Mz'] * (0.7 if 'mz' in axes_to_max else 0.2), 
                                            sensor_limits['Mz'] * (0.7 if 'mz' in axes_to_max else 0.2))
        
        combined_fraction = calculate_combined_load_fraction(fx, fy, fz, mx, my, mz)
        
        # Accept if it's in the strong range
        if normal_combined_limit < combined_fraction <= strong_combined_limit:
            data.append([fx, fy, fz, mx, my, mz, 'Strong'])
        elif combined_fraction > strong_combined_limit:
            # Scale down to fit within strong limit
            scale_factor = strong_combined_limit / combined_fraction * 0.98
            scaled_fx = fx * scale_factor
            scaled_fy = fy * scale_factor
            scaled_fz = fz * scale_factor
            scaled_mx = mx * scale_factor
            scaled_my = my * scale_factor
            scaled_mz = mz * scale_factor
            
            # Verify the scaled version is still in strong range
            scaled_combined = calculate_combined_load_fraction(scaled_fx, scaled_fy, scaled_fz, 
                                                             scaled_mx, scaled_my, scaled_mz)
            if scaled_combined > normal_combined_limit:
                data.append([scaled_fx, scaled_fy, scaled_fz, scaled_mx, scaled_my, scaled_mz, 'Strong'])
        
        if len(data) % 200 == 0:
            print(f"  Generated {len(data)}/{n_strong} strong cases...")
    
    print(f"✅ Generated {len(data)} strong cases in {attempts} attempts")
    return data

# Generate the datasets
print("🎲 Generating truly random FT sensor test data...")
print(f"Target: {n_total} total points ({n_normal} normal + {n_strong} strong)")

normal_data = generate_realistic_normal_loads()
strong_data = generate_realistic_strong_loads()

# Combine and shuffle thoroughly multiple times
all_data = normal_data + strong_data
for _ in range(5):  # Multiple shuffles for better randomization
    np.random.shuffle(all_data)

# Convert to DataFrame and shuffle again
temp_df = pd.DataFrame(all_data, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Load_Type'])
temp_df = temp_df.sample(frac=1).reset_index(drop=True)  # Sample without fixed seed for true randomness
all_data = temp_df.values.tolist()

# Create final DataFrame
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
filename = 'truly_random_ft_sensor_data.csv'
df.to_csv(filename, index=False, float_format='%.2f')

# Display summary
print("\n" + "="*60)
print("🎯 TRULY RANDOM FT SENSOR TEST DATA")
print("="*60)
print(f"📊 Total test cases generated: {len(df)}")
print(f"🟦 Normal loads: {len(df[df['Load_Type'] == 'Normal'])} ({len(df[df['Load_Type'] == 'Normal'])/len(df)*100:.1f}%)")
print(f"🟥 Strong loads: {len(df[df['Load_Type'] == 'Strong'])} ({len(df[df['Load_Type'] == 'Strong'])/len(df)*100:.1f}%)")

print(f"\n⚖️  COMBINED LOAD VERIFICATION:")
normal_loads = df[df['Load_Type'] == 'Normal']['Combined_Load_Fraction']
strong_loads = df[df['Load_Type'] == 'Strong']['Combined_Load_Fraction']
print(f"   Normal range: {normal_loads.min():.3f} - {normal_loads.max():.3f} (should be ≤ {normal_combined_limit})")
print(f"   Strong range: {strong_loads.min():.3f} - {strong_loads.max():.3f} (should be {normal_combined_limit}-{strong_combined_limit})")

print(f"\n💪 FORCE STATISTICS:")
print(f"   Range: {df['Force_Magnitude'].min():.1f} - {df['Force_Magnitude'].max():.1f} N")
print(f"   Average: {df['Force_Magnitude'].mean():.1f} N")
print(f"   Std Dev: {df['Force_Magnitude'].std():.1f} N")

print(f"\n🔄 TORQUE STATISTICS:")
print(f"   Range: {df['Torque_Magnitude'].min():.1f} - {df['Torque_Magnitude'].max():.1f} Nm")
print(f"   Average: {df['Torque_Magnitude'].mean():.1f} Nm")
print(f"   Std Dev: {df['Torque_Magnitude'].std():.1f} Nm")

print(f"\n📐 INDIVIDUAL AXIS RANGES:")
for axis in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
    unit = 'N' if axis.startswith('F') else 'Nm'
    print(f"   {axis}: {df[axis].min():.1f} to {df[axis].max():.1f} {unit}")

print(f"\n🎯 DISTRIBUTION ANALYSIS:")
print("   This data uses 8 different random distributions:")
print("   - Uniform, Exponential, Beta (low/high), Triangular")
print("   - Gamma, Weibull, Log-normal distributions")
print("   - Should show much more realistic, non-bell-curve patterns!")

print(f"\n📁 FILES CREATED:")
print(f"   📄 {filename} - Ready for FEM simulation")
print(f"   🎲 {len(df)} truly random test cases with realistic load scenarios")

print(f"\n✅ Ready for visualization! Run the clean_ft_visualizer.py script to see the improved distributions.")