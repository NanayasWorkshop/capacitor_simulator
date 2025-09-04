import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the data
try:
    df = pd.read_csv('anti_bell_ft_sensor_data.csv')
    print(f"✅ Loaded {len(df)} data points successfully!")
except FileNotFoundError:
    print("❌ Error: realistic_ft_sensor_data.csv not found!")
    print("Please run the data generation script first.")
    exit()

# Create color mapping for load types - more vibrant colors
color_map = {'Normal': 'darkblue', 'Strong': 'red'}

print("🔄 Creating visualizations...")

# Create subplots with multiple rows and columns
fig = make_subplots(
    rows=4, cols=3,
    subplot_titles=[
        'Force Magnitude Distribution', 'Torque Magnitude Distribution', 'Combined Load Distribution',
        'Fx Distribution', 'Fy Distribution', 'Fz Distribution', 
        'Mx Distribution', 'My Distribution', 'Mz Distribution',
        '3D Force Plot', 'Force vs Torque Scatter', 'Load Type Distribution'
    ],
    specs=[
        [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
        [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
        [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
        [{"type": "scatter3d"}, {"type": "scatter"}, {"type": "bar"}]
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.08
)

# 1. Force Magnitude Distribution
for load_type in ['Normal', 'Strong']:
    data_subset = df[df['Load_Type'] == load_type]
    fig.add_trace(
        go.Histogram(
            x=data_subset['Force_Magnitude'],
            name=f'{load_type} Forces',
            opacity=0.7,
            marker_color=color_map[load_type],
            showlegend=True
        ),
        row=1, col=1
    )

# 2. Torque Magnitude Distribution
for load_type in ['Normal', 'Strong']:
    data_subset = df[df['Load_Type'] == load_type]
    fig.add_trace(
        go.Histogram(
            x=data_subset['Torque_Magnitude'],
            name=f'{load_type} Torques',
            opacity=0.7,
            marker_color=color_map[load_type],
            showlegend=False
        ),
        row=1, col=2
    )

# 3. Combined Load Distribution
for load_type in ['Normal', 'Strong']:
    data_subset = df[df['Load_Type'] == load_type]
    fig.add_trace(
        go.Histogram(
            x=data_subset['Combined_Load_Fraction'],
            name=f'{load_type} Combined',
            opacity=0.7,
            marker_color=color_map[load_type],
            showlegend=False
        ),
        row=1, col=3
    )

# 4-6. Individual Force Components
force_axes = ['Fx', 'Fy', 'Fz']
for i, axis in enumerate(force_axes):
    for load_type in ['Normal', 'Strong']:
        data_subset = df[df['Load_Type'] == load_type]
        fig.add_trace(
            go.Histogram(
                x=data_subset[axis],
                name=f'{load_type} {axis}',
                opacity=0.7,
                marker_color=color_map[load_type],
                showlegend=False
            ),
            row=2, col=i+1
        )

# 7-9. Individual Torque Components
torque_axes = ['Mx', 'My', 'Mz']
for i, axis in enumerate(torque_axes):
    for load_type in ['Normal', 'Strong']:
        data_subset = df[df['Load_Type'] == load_type]
        fig.add_trace(
            go.Histogram(
                x=data_subset[axis],
                name=f'{load_type} {axis}',
                opacity=0.7,
                marker_color=color_map[load_type],
                showlegend=False
            ),
            row=3, col=i+1
        )

# 10. 3D Force Plot
for load_type in ['Normal', 'Strong']:
    data_subset = df[df['Load_Type'] == load_type]
    fig.add_trace(
        go.Scatter3d(
            x=data_subset['Fx'],
            y=data_subset['Fy'],
            z=data_subset['Fz'],
            mode='markers',
            marker=dict(
                size=3,
                color=color_map[load_type],
                opacity=0.6
            ),
            name=f'{load_type} 3D Forces',
            showlegend=False
        ),
        row=4, col=1
    )

# 11. Force vs Torque Scatter
for load_type in ['Normal', 'Strong']:
    data_subset = df[df['Load_Type'] == load_type]
    fig.add_trace(
        go.Scatter(
            x=data_subset['Force_Magnitude'],
            y=data_subset['Torque_Magnitude'],
            mode='markers',
            marker=dict(
                size=4,
                color=color_map[load_type],
                opacity=0.6
            ),
            name=f'{load_type} F vs T',
            showlegend=False
        ),
        row=4, col=2
    )

# 12. Load Type Distribution (Bar Chart)
load_counts = df['Load_Type'].value_counts()
fig.add_trace(
    go.Bar(
        x=load_counts.index,
        y=load_counts.values,
        marker_color=[color_map[load_type] for load_type in load_counts.index],
        showlegend=False,
        text=load_counts.values,
        textposition='auto'
    ),
    row=4, col=3
)

# Update layout
fig.update_layout(
    height=1200,
    width=1400,
    title_text="FT Sensor Data Analysis Dashboard",
    title_x=0.5,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update x-axis labels
fig.update_xaxes(title_text="Force Magnitude (N)", row=1, col=1)
fig.update_xaxes(title_text="Torque Magnitude (Nm)", row=1, col=2)
fig.update_xaxes(title_text="Combined Load Fraction", row=1, col=3)
fig.update_xaxes(title_text="Fx (N)", row=2, col=1)
fig.update_xaxes(title_text="Fy (N)", row=2, col=2)
fig.update_xaxes(title_text="Fz (N)", row=2, col=3)
fig.update_xaxes(title_text="Mx (Nm)", row=3, col=1)
fig.update_xaxes(title_text="My (Nm)", row=3, col=2)
fig.update_xaxes(title_text="Mz (Nm)", row=3, col=3)
fig.update_xaxes(title_text="Force Magnitude (N)", row=4, col=2)
fig.update_xaxes(title_text="Load Type", row=4, col=3)

# Update y-axis labels
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=3)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=2)
fig.update_yaxes(title_text="Count", row=2, col=3)
fig.update_yaxes(title_text="Count", row=3, col=1)
fig.update_yaxes(title_text="Count", row=3, col=2)
fig.update_yaxes(title_text="Count", row=3, col=3)
fig.update_yaxes(title_text="Torque Magnitude (Nm)", row=4, col=2)
fig.update_yaxes(title_text="Count", row=4, col=3)

# Update 3D scene
fig.update_scenes(
    xaxis_title="Fx (N)",
    yaxis_title="Fy (N)",
    zaxis_title="Fz (N)",
    row=4, col=1
)

# Save as HTML file (don't show in terminal)
filename = "ft_sensor_analysis.html"
fig.write_html(filename)
print(f"✅ Visualization saved as '{filename}'")
print("📂 Open this file in your web browser to view the interactive plots!")

# Print summary statistics
print("\n" + "="*50)
print("📊 SUMMARY STATISTICS")
print("="*50)
print(f"📈 Total data points: {len(df)}")
print(f"🟦 Normal loads: {len(df[df['Load_Type'] == 'Normal'])} ({len(df[df['Load_Type'] == 'Normal'])/len(df)*100:.1f}%)")
print(f"🟥 Strong loads: {len(df[df['Load_Type'] == 'Strong'])} ({len(df[df['Load_Type'] == 'Strong'])/len(df)*100:.1f}%)")

print(f"\n💪 FORCE STATISTICS:")
print(f"   Range: {df['Force_Magnitude'].min():.1f} - {df['Force_Magnitude'].max():.1f} N")
print(f"   Average: {df['Force_Magnitude'].mean():.1f} N")

print(f"\n🔄 TORQUE STATISTICS:")
print(f"   Range: {df['Torque_Magnitude'].min():.1f} - {df['Torque_Magnitude'].max():.1f} Nm")
print(f"   Average: {df['Torque_Magnitude'].mean():.1f} Nm")

print(f"\n⚖️  COMBINED LOAD STATISTICS:")
print(f"   Range: {df['Combined_Load_Fraction'].min():.3f} - {df['Combined_Load_Fraction'].max():.3f}")
print(f"   Average: {df['Combined_Load_Fraction'].mean():.3f}")

print(f"\n📐 INDIVIDUAL AXIS RANGES:")
for axis in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
    unit = 'N' if axis.startswith('F') else 'Nm'
    print(f"   {axis}: {df[axis].min():.1f} to {df[axis].max():.1f} {unit}")

print(f"\n🎯 Next steps:")
print(f"   1. Open '{filename}' in your web browser")
print(f"   2. Explore the interactive plots")
print(f"   3. Verify the distributions look realistic")
print(f"   4. Use 'realistic_ft_sensor_data.csv' for FEM simulation")