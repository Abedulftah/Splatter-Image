import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def show_point_cloud(xyz, colors, path):
    
    colors = colors.squeeze().squeeze(1)
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = colors.cpu().numpy()
    xyz = xyz.squeeze().cpu().numpy()

    # Open the .ply file for writing
    save_files(xyz, (colors * 255).astype(np.uint8), path)

    # app = gui.Application.instance
    # app.initialize()
    # # point_cloud = o3d.io.read_point_cloud("output_with_colors.ply")
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(xyz)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # vis = o3d.visualization.O3DVisualizer("Gaussians", 1024, 768)
    # vis.show_settings = True
    # vis.add_geometry("Points", point_cloud)

    # vis.reset_camera_to_default()

    # app.add_window(vis)
    # app.run()

def save_files(xyz, colors, path):
    with open(path, "w") as f:
            # Write the PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write the xyz and color data
            for point, color in zip(xyz, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

def vis_point_cloud():
    for name in ['front', 'back', 'both']:
        app = gui.Application.instance
        app.initialize()
        point_cloud = o3d.io.read_point_cloud(f"{name}.ply")

        vis = o3d.visualization.O3DVisualizer('Gaussians', 1024, 768)
        vis.show_settings = True
        vis.add_geometry("Points", point_cloud)

        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()