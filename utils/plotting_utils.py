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

    if not os.path.exists('front_xyz_latest.npy'):
        np.save("front_xyz_latest.npy", xyz)
        np.save("front_colors_latest.npy", colors)
    elif not os.path.exists('back_xyz_latest.npy'):
        np.save("back_xyz_latest.npy", xyz)
        np.save("back_colors_latest.npy", colors)
    else:
        np.save("both_xyz_latest.npy", xyz)
        np.save("both_colors_latest.npy", colors)


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
# list_xyz.append(np.load("front_xyz.npy"))
# list_colors.append(np.load("front_colors.npy"))

# list_xyz.append(np.load("back_xyz.npy"))
# list_colors.append(np.load("back_colors.npy"))

# list_xyz.append(np.load("both_xyz.npy"))
# list_colors.append(np.load("both_colors.npy"))

# for xyz, color, title in zip(list_xyz, list_colors, titles):
#     app = gui.Application.instance
#     app.initialize()
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(xyz)
#     point_cloud.colors = o3d.utility.Vector3dVector(color)

#     vis = o3d.visualization.O3DVisualizer(title, 1024, 768)
#     vis.show_settings = True
#     vis.add_geometry("Points", point_cloud)

#     vis.reset_camera_to_default()

#     app.add_window(vis)
#     app.run()