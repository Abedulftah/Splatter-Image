# Adapted from https://github.com/graphdeco-inria/gaussian-splatting/tree/main
# to take in a predicted dictionary with 3D Gaussian parameters.

import math
import torch
import numpy as np

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import focal2fov

def render_predicted(pc : dict, 
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor, 
                     cfg, 
                     scaling_modifier = 1.0, 
                     override_color = None,
                     focals_pixels = None):
    """
    Render the scene as specified by pc dictionary. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc1, pc2 = pc['forward'], pc['back']
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    forward_screenspace_points = torch.zeros_like(pc1["xyz"], dtype=pc1["xyz"].dtype, requires_grad=True, device=pc1["xyz"].device) + 0
    back_screenspace_points = torch.zeros_like(pc2["xyz"], dtype=pc2["xyz"].dtype, requires_grad=True, device=pc2["xyz"].device) + 0

    try:
        forward_screenspace_points.retain_grad()
        back_screenspace_points.retain_grad()
    except:
        pass

    if focals_pixels == None:
        tanfovx = math.tan(cfg.data.fov * np.pi / 360)
        tanfovy = math.tan(cfg.data.fov * np.pi / 360)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg.data.training_resolution))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg.data.training_resolution))
    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cfg.data.training_resolution),
        image_width=int(cfg.data.training_resolution),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg.model.max_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    forward_means3D, back_means3D = pc1["xyz"], pc2['xyz']
    forward_means2D, back_means2D = forward_screenspace_points, back_screenspace_points
    forward_opacity, back_opacity = pc1["opacity"], pc2["opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    forward_scales, back_scales = None, None
    forward_rotations, back_rotations = None, None
    forward_cov3D_precomp, back_cov3D_precomp = None, None

    forward_scales, back_scales = pc1["scaling"], pc2["scaling"]
    forward_rotations, back_rotations = pc1["rotation"], pc2["rotation"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    forward_shs, back_shs = None, None
    forward_colors_precomp, back_colors_precomp = None, None
    if override_color is None:
        if "features_rest" in pc1.keys():
            forward_shs = torch.cat([pc1["features_dc"], pc1["features_rest"]], dim=1).contiguous()
            back_shs = torch.cat([pc2["features_dc"], pc2["features_rest"]], dim=1).contiguous()
        else:
            forward_shs = pc1["features_dc"]
            back_shs = pc2["features_dc"]
    else:
        forward_colors_precomp = override_color
        back_colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    forward_rendered_image, forward_radii = rasterizer(
        means3D = forward_means3D,
        means2D = forward_means2D,
        shs = forward_shs,
        colors_precomp = forward_colors_precomp,
        opacities = forward_opacity,
        scales = forward_scales,
        rotations = forward_rotations,
        cov3D_precomp = forward_cov3D_precomp)
    
    back_rendered_image, back_radii = rasterizer(
        means3D = back_means3D,
        means2D = back_means2D,
        shs = back_shs,
        colors_precomp = back_colors_precomp,
        opacities = back_opacity,
        scales = back_scales,
        rotations = back_rotations,
        cov3D_precomp = back_cov3D_precomp)
    
    forward_radii = forward_radii.reshape(64, 64).unsqueeze(0)
    back_radii = back_radii.reshape(64, 64).unsqueeze(0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = forward_rendered_image * (forward_radii > 0) + back_rendered_image * (forward_radii <= 0)

    return {"render": rendered_image,
            "viewspace_points": forward_screenspace_points,
            "visibility_filter" : forward_radii > 0,
            "radii": forward_radii}
