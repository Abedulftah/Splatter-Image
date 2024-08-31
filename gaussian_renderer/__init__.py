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
    means3D = torch.cat((pc1['xyz'], pc2['xyz']), dim=0)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
    try:
        screenspace_points.retain_grad()
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

    means2D = screenspace_points
    opacity = torch.cat((pc1['opacity'], pc2['opacity']), dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = torch.cat((pc1['scaling'], pc2['scaling']), dim=0)
    rotations = torch.cat((pc1['rotation'], pc2['rotation']), dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc1.keys():
            shs = torch.cat((torch.cat([pc1["features_dc"], pc1["features_rest"]], dim=1).contiguous(), torch.cat([pc2["features_dc"], pc2["features_rest"]], dim=1).contiguous()), dim=0)
        else:
            shs = torch.cat((pc1['features_dc'], pc2['features_dc']), dim=0)
    else:
        # mean we need to cat
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}