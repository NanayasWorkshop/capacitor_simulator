#!/usr/bin/env python3
"""
Test script to visualize triangle movement and tilting using Plotly
This will help debug the transformation calculations by showing actual movement
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math

class TransformationVisualizer:
    """Visualize triangle transformations and movements"""
    
    def __init__(self, radius=1.0):
        self.radius = radius
        self.reference_triangle = self._create_reference_triangle()
        
    def _create_reference_triangle(self):
        """Create perfect equilateral triangle"""
        angle_offset = 2 * math.pi / 3  # 120 degrees
        points = []
        
        for i in range(3):
            angle = i * angle_offset
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            z = 0.0
            points.append([x, y, z])
            
        return np.array(points)
    
    def _calculate_center_and_normal(self, points):
        """Calculate center and normal vector from 3 points"""
        points = np.array(points)
        
        # Center is average of points
        center = np.mean(points, axis=0)
        
        # Calculate normal using cross product
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        
        # Normalize
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length
        else:
            normal = np.array([0, 0, 1])  # Default to Z-up
            
        return center, normal
    
    def load_test_data(self):
        """Load the test data you provided"""
        # A1 data (converted from meters to mm)
        a1_raw = [
            [-1.659770575E-07*1000, -1.179934358E-06*1000, 1.000620481E-06*1000, -1.145295825E-06*1000, -1.209201956E-06*1000, 1.829048635E-06*1000, 8.054139686E-07*1000, -1.173652267E-06*1000, 1.756289879E-06*1000],
            [2.541475455E-07*1000, 2.24608538E-07*1000, 5.085159354E-07*1000, -1.169215344E-07*1000, 2.655541908E-07*1000, 3.575213955E-07*1000, 6.640083985E-07*1000, 2.826868613E-07*1000, 4.335876492E-07*1000],
            [-2.858283306E-06*1000, 1.658352678E-06*1000, -1.770424673E-06*1000, -1.398418752E-06*1000, 1.495303825E-06*1000, -2.243533932E-06*1000, -4.749502094E-06*1000, 1.786999263E-06*1000, -3.58251152E-06*1000],
            [3.15327019E-06*1000, 5.583605359E-07*1000, 7.260665235E-07*1000, 2.856105291E-06*1000, 9.889849648E-07*1000, -5.263666967E-07*1000, 3.903975407E-06*1000, 2.93911285E-07*1000, 1.353088891E-06*1000],
            [-7.905245341E-07*1000, 1.499798259E-07*1000, 2.911600669E-07*1000, -1.082551169E-06*1000, 1.586509667E-07*1000, 4.066395983E-07*1000, -6.44201702E-07*1000, 1.954640271E-07*1000, 1.312451657E-08*1000]
        ]
        
        # A2 data (converted from meters to mm)
        a2_raw = [
            [-1.072588656E-06*1000, -7.157275721E-06*1000, 2.141914568E-05*1000, -1.932420485E-06*1000, -7.175575232E-06*1000, 2.472454911E-05*1000, -9.428906168E-08*1000, -7.142134035E-06*1000, 2.463533256E-05*1000],
            [2.16541971E-06*1000, 3.358715593E-06*1000, 1.146611294E-05*1000, 1.688264016E-06*1000, 3.246442172E-06*1000, 1.049847623E-05*1000, 2.821284955E-06*1000, 3.330457746E-06*1000, 1.059972235E-05*1000],
            [-1.456349657E-05*1000, 8.905987358E-06*1000, -3.897739888E-05*1000, -1.335573796E-05*1000, 9.065539877E-06*1000, -4.311049048E-05*1000, -1.687256969E-05*1000, 8.878732099E-06*1000, -4.52986756E-05*1000],
            [8.334655985E-06*1000, 6.693400566E-06*1000, 1.623774227E-05*1000, 7.747718489E-06*1000, 6.790950304E-06*1000, 1.242091664E-05*1000, 9.373856412E-06*1000, 6.368407844E-06*1000, 1.571419862E-05*1000],
            [-2.822230226E-06*1000, 2.073867292E-06*1000, 6.319916326E-06*1000, -3.561275491E-06*1000, 2.143575665E-06*1000, 5.807055593E-06*1000, -2.960212438E-06*1000, 1.937899659E-06*1000, 5.118607902E-06*1000]
        ]
        
        # Convert to proper format: [step][node][xyz]
        a1_data = []
        a2_data = []
        
        for step in range(5):
            # A1: Extract 3 nodes with X,Y,Z each
            a1_step = []
            for node in range(3):
                x = a1_raw[step][node*3 + 0]
                y = a1_raw[step][node*3 + 1] 
                z = a1_raw[step][node*3 + 2]
                a1_step.append([x, y, z])
            a1_data.append(np.array(a1_step))
            
            # A2: Extract 3 nodes with X,Y,Z each
            a2_step = []
            for node in range(3):
                x = a2_raw[step][node*3 + 0]
                y = a2_raw[step][node*3 + 1]
                z = a2_raw[step][node*3 + 2]
                a2_step.append([x, y, z])
            a2_data.append(np.array(a2_step))
        
        return a1_data, a2_data
    
    def calculate_actual_positions(self, displacement_data):
        """Calculate actual positions by adding displacements to reference triangle"""
        actual_positions = []
        
        for displacements in displacement_data:
            actual = self.reference_triangle + displacements
            actual_positions.append(actual)
            
        return actual_positions
    
    def create_3d_visualization(self, a1_positions, a2_positions):
        """Create 3D visualization of triangle movement"""
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d', 'colspan': 3}, None, None]],
            subplot_titles=('A1 Triangles', 'A2 Triangles', 'A1 vs A2 Combined', 'Movement Analysis'),
            vertical_spacing=0.1
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Plot A1 triangles
        for i, pos in enumerate(a1_positions):
            center, normal = self._calculate_center_and_normal(pos)
            
            # Add triangle edges
            for j in range(3):
                k = (j + 1) % 3
                fig.add_trace(go.Scatter3d(
                    x=[pos[j][0], pos[k][0]], 
                    y=[pos[j][1], pos[k][1]], 
                    z=[pos[j][2], pos[k][2]],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=4),
                    marker=dict(size=6),
                    name=f'A1 Step {i}',
                    showlegend=(j==0)
                ), row=1, col=1)
            
            # Add center point
            fig.add_trace(go.Scatter3d(
                x=[center[0]], 
                y=[center[1]], 
                z=[center[2]],
                mode='markers',
                marker=dict(size=10, color=colors[i], symbol='circle'),
                name=f'A1 Center {i}',
                showlegend=False
            ), row=1, col=1)
            
            # Add dotted normal vector from center
            normal_scale = 0.005  # Length of normal line
            fig.add_trace(go.Scatter3d(
                x=[center[0], center[0] + normal[0] * normal_scale], 
                y=[center[1], center[1] + normal[1] * normal_scale], 
                z=[center[2], center[2] + normal[2] * normal_scale],
                mode='lines',
                line=dict(color=colors[i], width=6, dash='dot'),
                name=f'A1 Normal {i}',
                showlegend=False
            ), row=1, col=1)
        
        # Plot A2 triangles  
        for i, pos in enumerate(a2_positions):
            center, normal = self._calculate_center_and_normal(pos)
            
            # Add triangle edges
            for j in range(3):
                k = (j + 1) % 3
                fig.add_trace(go.Scatter3d(
                    x=[pos[j][0], pos[k][0]], 
                    y=[pos[j][1], pos[k][1]], 
                    z=[pos[j][2], pos[k][2]],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=4),
                    marker=dict(size=6),
                    name=f'A2 Step {i}',
                    showlegend=(j==0)
                ), row=1, col=2)
            
            # Add center point
            fig.add_trace(go.Scatter3d(
                x=[center[0]], 
                y=[center[1]], 
                z=[center[2]],
                mode='markers',
                marker=dict(size=10, color=colors[i], symbol='circle'),
                name=f'A2 Center {i}',
                showlegend=False
            ), row=1, col=2)
            
            # Add dotted normal vector from center
            normal_scale = 0.005  # Length of normal line
            fig.add_trace(go.Scatter3d(
                x=[center[0], center[0] + normal[0] * normal_scale], 
                y=[center[1], center[1] + normal[1] * normal_scale], 
                z=[center[2], center[2] + normal[2] * normal_scale],
                mode='lines',
                line=dict(color=colors[i], width=6, dash='dot'),
                name=f'A2 Normal {i}',
                showlegend=False
            ), row=1, col=2)
        
        # Combined view with amplified movement
        amplification = 10  # Amplify movement for visibility
        
        for i, (pos1, pos2) in enumerate(zip(a1_positions, a2_positions)):
            # A1 amplified
            pos1_amp = pos1 * amplification
            for j in range(3):
                k = (j + 1) % 3
                fig.add_trace(go.Scatter3d(
                    x=[pos1_amp[j][0], pos1_amp[k][0]], 
                    y=[pos1_amp[j][1], pos1_amp[k][1]], 
                    z=[pos1_amp[j][2], pos1_amp[k][2]],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=3, dash='solid'),
                    marker=dict(size=5),
                    name=f'A1 Step {i} (×{amplification})',
                    showlegend=(j==0)
                ), row=1, col=3)
            
            # A2 amplified
            pos2_amp = pos2 * amplification
            for j in range(3):
                k = (j + 1) % 3
                fig.add_trace(go.Scatter3d(
                    x=[pos2_amp[j][0], pos2_amp[k][0]], 
                    y=[pos2_amp[j][1], pos2_amp[k][1]], 
                    z=[pos2_amp[j][2], pos2_amp[k][2]],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=3, dash='dash'),
                    marker=dict(size=5),
                    name=f'A2 Step {i} (×{amplification})',
                    showlegend=(j==0)
                ), row=1, col=3)
        
        # Centers and normals analysis
        for i, (pos1, pos2) in enumerate(zip(a1_positions, a2_positions)):
            center1, normal1 = self._calculate_center_and_normal(pos1)
            center2, normal2 = self._calculate_center_and_normal(pos2)
            
            # Plot centers
            fig.add_trace(go.Scatter3d(
                x=[center1[0] * amplification], 
                y=[center1[1] * amplification], 
                z=[center1[2] * amplification],
                mode='markers',
                marker=dict(size=10, color=colors[i], symbol='circle'),
                name=f'A1 Center {i}',
                showlegend=(i==0)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter3d(
                x=[center2[0] * amplification], 
                y=[center2[1] * amplification], 
                z=[center2[2] * amplification],
                mode='markers',
                marker=dict(size=10, color=colors[i], symbol='square'),
                name=f'A2 Center {i}',
                showlegend=(i==0)
            ), row=2, col=1)
            
            # Plot normal vectors (scaled)
            normal_scale = 0.01 * amplification
            fig.add_trace(go.Scatter3d(
                x=[center1[0] * amplification, (center1[0] + normal1[0] * normal_scale)], 
                y=[center1[1] * amplification, (center1[1] + normal1[1] * normal_scale)], 
                z=[center1[2] * amplification, (center1[2] + normal1[2] * normal_scale)],
                mode='lines',
                line=dict(color=colors[i], width=6),
                name=f'A1 Normal {i}',
                showlegend=(i==0)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter3d(
                x=[center2[0] * amplification, (center2[0] + normal2[0] * normal_scale)], 
                y=[center2[1] * amplification, (center2[1] + normal2[1] * normal_scale)], 
                z=[center2[2] * amplification, (center2[2] + normal2[2] * normal_scale)],
                mode='lines',
                line=dict(color=colors[i], width=6, dash='dash'),
                name=f'A2 Normal {i}',
                showlegend=(i==0)
            ), row=2, col=1)
        
        fig.update_layout(
            title_text="Triangle Movement Analysis - First 5 Steps",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def print_analysis(self, a1_positions, a2_positions):
        """Print numerical analysis of the movement"""
        print("="*60)
        print("TRIANGLE MOVEMENT ANALYSIS")
        print("="*60)
        
        for i, (pos1, pos2) in enumerate(zip(a1_positions, a2_positions)):
            center1, normal1 = self._calculate_center_and_normal(pos1)
            center2, normal2 = self._calculate_center_and_normal(pos2)
            
            print(f"\nStep {i}:")
            print(f"  A1 Center: [{center1[0]:.6f}, {center1[1]:.6f}, {center1[2]:.6f}] mm")
            print(f"  A2 Center: [{center2[0]:.6f}, {center2[1]:.6f}, {center2[2]:.6f}] mm")
            print(f"  A1 Normal: [{normal1[0]:.6f}, {normal1[1]:.6f}, {normal1[2]:.6f}]")
            print(f"  A2 Normal: [{normal2[0]:.6f}, {normal2[1]:.6f}, {normal2[2]:.6f}]")
            
            # Calculate relative movement
            center_diff = center2 - center1
            print(f"  Center Diff: [{center_diff[0]:.6f}, {center_diff[1]:.6f}, {center_diff[2]:.6f}] mm")
            
            # Calculate angle between normals
            dot_product = np.dot(normal1, normal2)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            print(f"  Normal Angle: {angle_deg:.3f} degrees")
        
        print("\n" + "="*60)
        print("DISPLACEMENT MAGNITUDES")
        print("="*60)
        
        for i, (pos1, pos2) in enumerate(zip(a1_positions, a2_positions)):
            print(f"\nStep {i}:")
            for j in range(3):
                disp1 = pos1[j] - self.reference_triangle[j]
                disp2 = pos2[j] - self.reference_triangle[j]
                mag1 = np.linalg.norm(disp1)
                mag2 = np.linalg.norm(disp2)
                print(f"  Node {j}: A1 displacement = {mag1:.6f} mm, A2 displacement = {mag2:.6f} mm")

def main():
    """Main test function"""
    print("Triangle Movement Visualization Test")
    print("="*50)
    
    # Create visualizer
    viz = TransformationVisualizer(radius=1.0)
    
    # Load test data
    print("Loading test data...")
    a1_displacements, a2_displacements = viz.load_test_data()
    
    # Calculate actual positions
    print("Calculating actual positions...")
    a1_positions = viz.calculate_actual_positions(a1_displacements)
    a2_positions = viz.calculate_actual_positions(a2_displacements)
    
    # Print numerical analysis
    viz.print_analysis(a1_positions, a2_positions)
    
    # Create visualization
    print("Creating 3D visualization...")
    fig = viz.create_3d_visualization(a1_positions, a2_positions)
    
    # Save to HTML file
    html_filename = "triangle_movement_analysis.html"
    fig.write_html(html_filename)
    
    print(f"\nVisualization complete!")
    print(f"HTML file saved as: {html_filename}")
    print("Note: Movement is amplified 1000× in the combined view for visibility")

if __name__ == "__main__":
    main()