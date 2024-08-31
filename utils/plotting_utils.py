import open3d as o3d

def show_point_cloud(xyz, colors):

    colors = colors.squeeze().squeeze(1)
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = colors.cpu().numpy()
    xyz = xyz.squeeze().cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])