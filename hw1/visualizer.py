#!pip install open3d

import open3d as o3d
import plotly.graph_objects as go
import numpy as np

pcd = o3d.io.read_point_cloud("assets/points3D.ply")

points = np.asarray(pcd.points)

colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors, opacity=0.8)
        )
    ]
)

fig.update_layout(
    title="Visualization of assets/points3D.ply",
    scene=dict(
        xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)"),
        yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)"),
        zaxis=dict(title='Z', backgroundcolor="rgb(230, 230,230)"),
        aspectmode='data'  
    )
)

fig.show()