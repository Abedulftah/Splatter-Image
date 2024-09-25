import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def show_point_cloud(xyz, colors, camera_centers):
    app = gui.Application.instance
    app.initialize()
    
    colors = colors.squeeze().squeeze(1)
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = colors.cpu().numpy()
    xyz = xyz.squeeze().cpu().numpy()

    if not os.path.exists('front_xyz_latest_wo_off.npy'):
        np.save("front_xyz_latest_wo_off.npy", xyz)
        np.save("front_colors_latest_wo_off.npy", colors)
    elif not os.path.exists('back_xyz_latest_wo_off.npy'):
        np.save("back_xyz_latest_wo_off.npy", xyz)
        np.save("back_colors_latest_wo_off.npy", colors)
    else:
        np.save("both_xyz_latest_wo_off.npy", xyz)
        np.save("both_colors_latest_wo_off.npy", colors)


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.O3DVisualizer("Gaussians", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", point_cloud)

    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()



# list_xyz, list_colors, titles = [], [], ['front', 'back', 'both']
# list_xyz.append(np.load("front_xyz_latest.npy"))
# list_colors.append(np.load("front_colors_latest.npy"))

# list_xyz.append(np.load("back_xyz_latest.npy"))
# list_colors.append(np.load("back_colors_latest.npy"))

# list_xyz.append(np.load("both_xyz_latest.npy"))
# list_colors.append(np.load("both_colors_latest.npy"))

# list_xyz_wo_off, list_colors_wo_off, titles_wo_off = [], [], ['front', 'back', 'both']
# list_xyz_wo_off.append(np.load("front_xyz_latest_wo_off.npy"))
# list_colors_wo_off.append(np.load("front_colors_latest_wo_off.npy"))

# list_xyz_wo_off.append(np.load("back_xyz_latest_wo_off.npy"))
# list_colors_wo_off.append(np.load("back_colors_latest_wo_off.npy"))

# list_xyz_wo_off.append(np.load("both_xyz_latest_wo_off.npy"))
# list_colors_wo_off.append(np.load("both_colors_latest_wo_off.npy"))

# for xyz, color, title, xyz_wo_off, color_wo_off, title_wo_off in zip(list_xyz, list_colors, titles, list_xyz_wo_off, list_colors_wo_off, titles_wo_off):
#     app = gui.Application.instance
#     app.initialize()
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(xyz)
#     point_cloud.colors = o3d.utility.Vector3dVector(color)

#     point_cloud_wo_off = o3d.geometry.PointCloud()
#     point_cloud_wo_off.points = o3d.utility.Vector3dVector(xyz_wo_off + 0.5)
#     point_cloud_wo_off.colors = o3d.utility.Vector3dVector(color_wo_off)

#     vis = o3d.visualization.O3DVisualizer(title, 1024, 768)
#     vis.show_settings = True
    
#     # Add both point clouds to the same visualizer
#     vis.add_geometry("Points", point_cloud)
#     vis.add_geometry("Points Without Offset", point_cloud_wo_off)
#     for point in xyz_wo_off:
#         if point[0] != 0:
#             vis.add_3d_label(point + 0.5, "point cloud without offset")
#             break

#     vis.reset_camera_to_default()

#     app.add_window(vis)
#     app.run()