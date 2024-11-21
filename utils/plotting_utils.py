import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from plyfile import PlyData, PlyElement
import torch
import seaborn as sns

def show_point_cloud(xyz, colors, path, sigma=None):
    
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

def vis_point_cloud(path=''):
    for name in ['back', 'front', 'both', 'rb_both']:
        app = gui.Application.instance
        app.initialize()
        point_cloud = o3d.io.read_point_cloud(path + f"{name}.ply")

        vis = o3d.visualization.O3DVisualizer('Gaussians', 1024, 768)
        vis.show_settings = True
        vis.add_geometry("Points", point_cloud)

        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

def make_vid(path='', name=''):

    point_cloud = o3d.io.read_point_cloud(path + f"{name}.ply")

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, visible=True)

    vis.add_geometry(point_cloud)
    vis.update_renderer()
    
    view_control = vis.get_view_control()
    view_control.rotate(0.0, 1000.0)

    lst = []
    for frame_count in range(200):
        rot = frame_count // 50
        if rot % 2 == 0:
            view_control.rotate(15.0, 0.0)
        else:
            view_control.rotate(0.0, 15.0)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        
        # Convert float buffer to uint8 image
        image = (255 * np.asarray(image)).astype(np.uint8)
        
        # Append the frame to the video writer
        lst.append(image)
    imageio.mimsave(path + f'{name}_pc.gif', lst, fps=20)  # Adjust fps as needed
    vis.destroy_window()


def write_gaussian_splatting_ply(filename, features_dc, features_rest, opacity, rotation, scaling, xyz):
    features_dc = features_dc.cpu().numpy()
    opacity = opacity.cpu().numpy()
    rotation = rotation.cpu().numpy()
    scaling = scaling.cpu().numpy()
    xyz = xyz.cpu().numpy()

    print("features_dc", features_dc.shape)
    print("features_rest", features_rest.shape)
    print("opacity", opacity.shape)
    print("rotation", rotation.shape)
    print("scaling", scaling.shape)
    print("xyz", xyz.shape)

    # Number of vertices
    num_vertices = xyz.shape[0]

    # Convert features to float arrays and handle the data as needed for PLY
    vertex_data = [
        (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],                       # Position
            features_dc[i, 0, 0], features_dc[i, 0, 1], features_dc[i, 0, 2],  # Spherical harmonics (4x3)
            *features_rest[i, 0], *features_rest[i, 1], *features_rest[i, 2], # Rest of the features (4x3)
            opacity[i],                                            # Opacity
            rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3],  # Rotation (quaternion)
            scaling[i, 0], scaling[i, 1], scaling[i, 2]            # Scaling (3 components)
        )
        for i in range(num_vertices)
    ]

    # Define vertex properties in PLY format
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('f_rest_00', 'f4'), ('f_rest_01', 'f4'), ('f_rest_02', 'f4'),
        ('f_rest_10', 'f4'), ('f_rest_11', 'f4'), ('f_rest_12', 'f4'),
        ('f_rest_20', 'f4'), ('f_rest_21', 'f4'), ('f_rest_22', 'f4'),
        ('opacity', 'f4'),
        ('rotation_0', 'f4'), ('rotation_1', 'f4'), ('rotation_2', 'f4'), ('rotation_3', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4')
    ]
    

    # Create the structured array
    vertex_array = np.array(vertex_data, dtype=vertex_dtype)

    # Create the PLY element and write to file
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    PlyData([vertex_element]).write(filename)
    print(f"File saved as {filename}")


def showSigma(sigma, xyz, path):
    N = sigma.shape[0]
    np.random.seed(42)

    # Random means (Nx3)
    mu = xyz
    sigma = np.einsum("nij,nkj->nik", sigma, sigma)  # Ensures positive-definiteness

    # Create a list to hold all ellipsoids
    ellipsoids = []

    for i in range(N):
        # Eigen decomposition for each covariance matrix
        eigvals, eigvecs = np.linalg.eigh(sigma[i])

        # Create a sphere mesh
        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)

        # Scaling and rotation
        scaling = np.sqrt(eigvals)
        scaling_matrix = np.diag(scaling)
        rotation_matrix = eigvecs

        # Transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix @ scaling_matrix  # Scale and rotate
        transform[:3, 3] = mu[i]  # Translate to mean

        # Apply transformation
        ellipsoid.transform(transform)

        # Assign a random color for visualization
        color = np.random.uniform(0.2, 1.0, 3)
        ellipsoid.paint_uniform_color(color)

        # Add to list
        ellipsoids.append(ellipsoid)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, visible=True)

    for i in range(len(ellipsoids)):
        vis.add_geometry(ellipsoids[i])
    vis.update_renderer()
    
    view_control = vis.get_view_control()
    view_control.rotate(0.0, 1000.0)

    lst = []
    for frame_count in range(200):
        rot = frame_count // 50
        if rot % 2 == 0:
            view_control.rotate(15.0, 0.0)
        else:
            view_control.rotate(0.0, 15.0)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        
        # Convert float buffer to uint8 image
        image = (255 * np.asarray(image)).astype(np.uint8)
        
        # Append the frame to the video writer
        lst.append(image)
    imageio.mimsave(path + f'back_shape.gif', lst, fps=20)  # Adjust fps as needed
    vis.destroy_window()
    # Visualize all ellipsoids
    # o3d.visualization.draw_geometries(ellipsoids, window_name="Multiple 3D Gaussians")


def heatMap(sigma, save_path):

    det = torch.linalg.det(sigma)  # Shape: (16384,)

    # Compute the volume for all Gaussians at once
    dimension = sigma.shape[2]
    volume = (2 * np.pi) ** (dimension / 2) * torch.sqrt(det)
    plt.figure(figsize=(8, 6))
    sns.heatmap(volume.view((128, 128)))
    plt.title("Determinant of Covariance Matrices")
    plt.show()

    # plt.savefig(save_path, dpi=300, bbox_inches="tight")

# vis_point_cloud('C:\\Users\\User\\SRN\\srn_cars\\out\\0_1079efee042629d4ce28f0f1b509eda\\')
# for name in ['back', 'front', 'both', 'rb_both']:
#     make_vid(f'C:\\Users\\User\\SRN\\srn_cars\\out\\0_1079efee042629d4ce28f0f1b509eda\\', name)