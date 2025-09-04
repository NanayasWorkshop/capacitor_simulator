import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

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
n_total = 1000
n_normal = 800  # 80%
n_strong = 200  # 20%

def calculate_combined_load_fraction(fx, fy, fz, mx, my, mz):
    """Calculate the combined load as fraction of maximum possible"""
    force_fraction = np.sqrt((fx/sensor_limits['Fx'])**2 + 
                           (fy/sensor_limits['Fy'])**2 + 
                           (fz/sensor_limits['Fz'])**2)
    
    torque_fraction = np.sqrt((mx/sensor_limits['Mx'])**2 + 
                            (my/sensor_limits['My'])**2 + 
                            (mz/sensor_limits['Mz'])**2)
    
    # Combined load (you can adjust this weighting)
    return np.sqrt(force_fraction**2 + torque_fraction**2)

def generate_normal_loads():
    """Generate realistic normal operation loads (80% of dataset) with combined limit"""
    data = []
    
    for i in range(n_normal):
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            # Generate initial values
            fx = np.random.normal(0, sensor_limits['Fx'] * 0.15)
            fy = np.random.normal(0, sensor_limits['Fy'] * 0.15)  
            fz = np.random.uniform(-sensor_limits['Fz'] * 0.1, sensor_limits['Fz'] * 0.3)
            
            mx = np.random.normal(0, sensor_limits['Mx'] * 0.2)
            my = np.random.normal(0, sensor_limits['My'] * 0.2)
            mz = np.random.normal(0, sensor_limits['Mz'] * 0.15)
            
            # Check combined load
            combined_fraction = calculate_combined_load_fraction(fx, fy, fz, mx, my, mz)
            
            if combined_fraction <= normal_combined_limit:
                data.append([fx, fy, fz, mx, my, mz, 'Normal'])
                break
            else:
                # Scale down if too large
                scale_factor = normal_combined_limit / combined_fraction * 0.9
                fx *= scale_factor
                fy *= scale_factor
                fz *= scale_factor
                mx *= scale_factor
                my *= scale_factor
                mz *= scale_factor
                data.append([fx, fy, fz, mx, my, mz, 'Normal'])
                break
            
            attempts += 1
    
    return data

def generate_strong_loads():
    """Generate strong loads (20% of dataset) with combined limit"""
    data = []
    
    for i in range(n_strong):
        attempts = 0
        while attempts < 100:
            # Generate stronger initial values
            fx = np.random.uniform(-sensor_limits['Fx'] * 0.7, sensor_limits['Fx'] * 0.7)
            fy = np.random.uniform(-sensor_limits['Fy'] * 0.7, sensor_limits['Fy'] * 0.7)
            fz = np.random.uniform(-sensor_limits['Fz'] * 0.8, sensor_limits['Fz'] * 0.8)
            
            mx = np.random.uniform(-sensor_limits['Mx'] * 0.6, sensor_limits['Mx'] * 0.6)
            my = np.random.uniform(-sensor_limits['My'] * 0.6, sensor_limits['My'] * 0.6)
            mz = np.random.uniform(-sensor_limits['Mz'] * 0.5, sensor_limits['Mz'] * 0.5)
            
            combined_fraction = calculate_combined_load_fraction(fx, fy, fz, mx, my, mz)
            
            # For strong loads, we want them between normal_limit and strong_limit
            if normal_combined_limit < combined_fraction <= strong_combined_limit:
                data.append([fx, fy, fz, mx, my, mz, 'Strong'])
                break
            elif combined_fraction > strong_combined_limit:
                # Scale down to fit within strong limit
                scale_factor = strong_combined_limit / combined_fraction * 0.95
                fx *= scale_factor
                fy *= scale_factor
                fz *= scale_factor
                mx *= scale_factor
                my *= scale_factor
                mz *= scale_factor
                data.append([fx, fy, fz, mx, my, mz, 'Strong'])
                break
            
            attempts += 1
            
        if attempts >= 100:  # Fallback if we can't generate valid strong load
            # Create a moderate strong load
            scale = np.random.uniform(0.5, 0.7)
            fx = np.random.uniform(-sensor_limits['Fx'] * scale, sensor_limits['Fx'] * scale)
            fy = np.random.uniform(-sensor_limits['Fy'] * scale, sensor_limits['Fy'] * scale)
            fz = np.random.uniform(-sensor_limits['Fz'] * scale, sensor_limits['Fz'] * scale)
            mx = my = mz = 0  # Zero torques for this fallback case
            data.append([fx, fy, fz, mx, my, mz, 'Strong'])
    
    return data

# Generate the datasets
print("Generating realistic FT sensor test data...")
normal_data = generate_normal_loads()
strong_data = generate_strong_loads()

# Combine and shuffle thoroughly
all_data = normal_data + strong_data
np.random.shuffle(all_data)  # First shuffle

# Convert to DataFrame for easier shuffling
temp_df = pd.DataFrame(all_data, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Load_Type'])
temp_df = temp_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Second shuffle with fixed seed
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
filename = 'realistic_ft_sensor_data.csv'
df.to_csv(filename, index=False, float_format='%.2f')

# Display summary
print("=== REALISTIC FT SENSOR TEST DATA ===")
print(f"Total test cases: {len(df)}")
print(f"Normal loads (≤{normal_combined_limit*100:.0f}% combined): {len(df[df['Load_Type'] == 'Normal'])}")
print(f"Strong loads ({normal_combined_limit*100:.0f}-{strong_combined_limit*100:.0f}% combined): {len(df[df['Load_Type'] == 'Strong'])}")
print()

print("COMBINED LOAD STATISTICS:")
print(f"Overall range: {df['Combined_Load_Fraction'].min():.3f} to {df['Combined_Load_Fraction'].max():.3f}")
print(f"Normal loads range: {df[df['Load_Type'] == 'Normal']['Combined_Load_Fraction'].min():.3f} to {df[df['Load_Type'] == 'Normal']['Combined_Load_Fraction'].max():.3f}")
print(f"Strong loads range: {df[df['Load_Type'] == 'Strong']['Combined_Load_Fraction'].min():.3f} to {df[df['Load_Type'] == 'Strong']['Combined_Load_Fraction'].max():.3f}")
print()

print("FIRST 10 TEST CASES:")
print(df[['Test_ID', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Combined_Load_Fraction', 'Load_Type']].head(10))
print()

print("SAMPLE STRONG CASES:")
strong_sample = df[df['Load_Type'] == 'Strong'].head(5)
print(strong_sample[['Test_ID', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Combined_Load_Fraction']])
print()

print(f"Data saved to: {filename}")
print("Ready for FEM simulation!")