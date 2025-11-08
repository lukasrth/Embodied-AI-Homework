# 2. Import all libraries
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

# 4. Define the correct file path
file_path = "/content/Embodied-AI-Homework/hw1/assets/points3D.ply"

print(f"Loading file: {file_path}")
pcd = o3d.io.read_point_cloud(file_path)

if not pcd.has_points():
    print("ERROR: FILE LOADED BUT HAS NO POINTS.")
else:
    print(f"Loaded {len(pcd.points)} points. Plotting...")
    
    # 5. Get points and colors
    points = np.asarray(pcd.points)
    # Get colors, normalize from [0, 255] to [0, 1] for matplotlib
    colors = np.asarray(pcd.colors) 

    # 6. Create a 3D Matplotlib plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    # We use a subset of points (e.g., 20000) for speed
    subset_size = min(len(points), 20000)
    indices = np.random.choice(len(points), subset_size, replace=False)
    
    ax.scatter(
        points[indices, 0],  # X
        points[indices, 1],  # Y
        points[indices, 2],  # Z
        c=colors[indices],   # Colors
        s=0.1                # Size of points
    )
    
    ax.set_title("Visualization of assets/points3D.ply")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # 7. Show the plot
    plt.show()