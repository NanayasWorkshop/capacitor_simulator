import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the data
try:
    df = pd.read_csv('even_distribution_ft_data.csv')
    print(f"✅ Loaded {len(df)} data points successfully!")
except FileNotFoundError:
    print("❌ Error: even_distribution_ft_data.csv not found!")
    print("Please run the data generation script first.")
    exit()

# Calculate combined magnitude for color mapping
df['Combined_Magnitude'] = np.sqrt(df['Force_Magnitude']**2 + (df['Torque_Magnitude']*10)**2)

# Define strength-based color palette
strength_colors = [
    "#000046",  # Dark blue (weakest)
    "#2E008C",  # Purple blue
    "#B70395",  # Magenta
    "#EE5E05",  # Orange
    "#FEC201",  # Yellow
    "#FFF5A0"   # Light yellow (strongest)
]

print("🎨 Creating strength-based color mapping...")

# Create subplots
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        'Force Magnitude Distribution', 'Torque Magnitude Distribution', 'Fx Distribution',
        'Fy Distribution', 'Fz Distribution', 'Mx Distribution', 
        'My Distribution', 'Mz Distribution', '3D Force Plot'
    ],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter3d"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Create colorscale for plotly
plotly_colorscale = [[i/(len(strength_colors)-1), color] for i, color in enumerate(strength_colors)]

# 1. Force Magnitude Distribution with strength colors
fig.add_trace(
    go.Scatter(
        x=df['Force_Magnitude'],
        y=np.random.normal(0, 0.1, len(df)),
        mode='markers',
        marker=dict(
            color=df['Force_Magnitude'],
            colorscale=plotly_colorscale,
            size=4,
            opacity=0.7,
            colorbar=dict(title="Force Magnitude (N)", x=0.15, len=0.3)
        ),
        name='Force Magnitude',
        showlegend=False
    ),
    row=1, col=1
)

# 2. Torque Magnitude Distribution with strength colors
fig.add_trace(
    go.Scatter(
        x=df['Torque_Magnitude'],
        y=np.random.normal(0, 0.1, len(df)),
        mode='markers',
        marker=dict(
            color=df['Torque_Magnitude'],
            colorscale=plotly_colorscale,
            size=4,
            opacity=0.7,
            colorbar=dict(title="Torque Magnitude (Nm)", x=0.48, len=0.3)
        ),
        name='Torque Magnitude',
        showlegend=False
    ),
    row=1, col=2
)

# 3-8. Individual Axis Distributions with strength colors
axes_info = [
    ('Fx', 1, 3), ('Fy', 2, 1), ('Fz', 2, 2), 
    ('Mx', 2, 3), ('My', 3, 1), ('Mz', 3, 2)
]

for axis, row, col in axes_info:
    fig.add_trace(
        go.Scatter(
            x=df[axis],
            y=np.random.normal(0, 0.1, len(df)),
            mode='markers',
            marker=dict(
                color=abs(df[axis]),
                colorscale=plotly_colorscale,
                size=3,
                opacity=0.6
            ),
            name=f'{axis} Distribution',
            showlegend=False
        ),
        row=row, col=col
    )

# 9. 3D Force Plot with combined strength colors
fig.add_trace(
    go.Scatter3d(
        x=df['Fx'],
        y=df['Fy'],
        z=df['Fz'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['Combined_Magnitude'],
            colorscale=plotly_colorscale,
            opacity=0.8,
            colorbar=dict(title="Combined Magnitude", x=0.85, len=0.5)
        ),
        name='3D Forces',
        showlegend=False
    ),
    row=3, col=3
)

# Update layout
fig.update_layout(
    height=1000,
    width=1400,
    title_text="FT Sensor Even Distribution Analysis (Strength-Colored)",
    title_x=0.5,
    showlegend=False
)

# Update x-axis labels
fig.update_xaxes(title_text="Force Magnitude (N)", row=1, col=1)
fig.update_xaxes(title_text="Torque Magnitude (Nm)", row=1, col=2)
fig.update_xaxes(title_text="Fx (N)", row=1, col=3)
fig.update_xaxes(title_text="Fy (N)", row=2, col=1)
fig.update_xaxes(title_text="Fz (N)", row=2, col=2)
fig.update_xaxes(title_text="Mx (Nm)", row=2, col=3)
fig.update_xaxes(title_text="My (Nm)", row=3, col=1)
fig.update_xaxes(title_text="Mz (Nm)", row=3, col=2)

# Update y-axis labels
fig.update_yaxes(title_text="Distribution", row=1, col=1)
fig.update_yaxes(title_text="Distribution", row=1, col=2)
fig.update_yaxes(title_text="Distribution", row=1, col=3)
fig.update_yaxes(title_text="Distribution", row=2, col=1)
fig.update_yaxes(title_text="Distribution", row=2, col=2)
fig.update_yaxes(title_text="Distribution", row=2, col=3)
fig.update_yaxes(title_text="Distribution", row=3, col=1)
fig.update_yaxes(title_text="Distribution", row=3, col=2)

# Update 3D scene
fig.update_scenes(
    xaxis_title="Fx (N)",
    yaxis_title="Fy (N)",
    zaxis_title="Fz (N)",
    row=3, col=3
)

# Save as HTML file
filename = "strength_colored_ft_analysis.html"
fig.write_html(filename)
print(f"✅ Visualization saved as '{filename}'")
print("📂 Open this file in your web browser to view the interactive plots!")

# Print summary statistics
print("\n" + "="*60)
print("📊 STRENGTH-COLORED ANALYSIS")
print("="*60)
print(f"📈 Total data points: {len(df)}")

print(f"\n🎨 STRENGTH-BASED COLOR MAPPING:")
print("   Dark blue (#000046) = Weakest forces/torques")
print("   Purple blue (#2E008C) = Low strength") 
print("   Magenta (#B70395) = Medium-low strength")
print("   Orange (#EE5E05) = Medium-high strength")
print("   Yellow (#FEC201) = High strength")
print("   Light yellow (#FFF5A0) = Strongest forces/torques")

print(f"\n💪 STRENGTH STATISTICS:")
print(f"   Combined magnitude range: {df['Combined_Magnitude'].min():.1f} - {df['Combined_Magnitude'].max():.1f}")
print(f"   Force magnitude range: {df['Force_Magnitude'].min():.1f} - {df['Force_Magnitude'].max():.1f} N")
print(f"   Average force magnitude: {df['Force_Magnitude'].mean():.1f} N")
print(f"   Standard deviation: {df['Force_Magnitude'].std():.1f} N")

print(f"\n🔄 TORQUE STATISTICS:")
print(f"   Torque magnitude range: {df['Torque_Magnitude'].min():.1f} - {df['Torque_Magnitude'].max():.1f} Nm")
print(f"   Average torque magnitude: {df['Torque_Magnitude'].mean():.1f} Nm")
print(f"   Standard deviation: {df['Torque_Magnitude'].std():.1f} Nm")

print(f"\n📐 INDIVIDUAL AXIS STATISTICS:")
for axis in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
    unit = 'N' if axis.startswith('F') else 'Nm'
    axis_data = df[axis]
    print(f"   {axis}: {axis_data.min():.1f} to {axis_data.max():.1f} {unit} (std: {axis_data.std():.1f})")

# Distribution evenness check
print(f"\n🎯 DISTRIBUTION EVENNESS CHECK:")
print("   (Lower standard deviation = more even distribution)")
for axis in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
    axis_data = df[axis]
    hist, bins = np.histogram(axis_data, bins=20)
    hist_std = np.std(hist)
    hist_mean = np.mean(hist)
    evenness_ratio = hist_std / hist_mean if hist_mean > 0 else 0
    
    status = "✅ Even" if evenness_ratio < 0.3 else "⚠️  Uneven" if evenness_ratio < 0.6 else "❌ Very uneven"
    print(f"   {axis}: {status} (evenness ratio: {evenness_ratio:.2f})")

print(f"\n🎯 Next steps:")
print(f"   1. Open '{filename}' in your web browser")
print(f"   2. Check the beautiful strength-based color gradients")
print(f"   3. Dark blue = weak, Light yellow = strong forces")
print(f"   4. Use 'even_distribution_ft_data.csv' for FEM simulation")
print(f"   5. All {len(df)} data points are ready!")