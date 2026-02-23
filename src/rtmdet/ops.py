"""Point and geometry operations for object detection."""

from typing import Sequence

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def get_image_shape_after_stride(
    image_shape: tuple[int, int], stride: int
) -> tuple[int, int]:
    """Compute the downsampled shape for a given stride."""
    return (image_shape[0] + stride - 1) // stride, (
        image_shape[1] + stride - 1
    ) // stride


def points_inside_oriented_boxes(
    point: Float[Tensor, "N 2"], box: Float[Tensor, "M 5"]
) -> Bool[Tensor, "N M"]:
    """
    Check if points are inside oriented boxes.

    Args:
        point: Points to be checked,  (x,y)
        box: Oriented boxes, with format (cx, cy, w, h, angle).
    Returns:
        boolean tensor of shape indicating whether each point is inside each box.
    """
    cxcy_gt = box[:, :2]
    wh_gt = box[:, 2:4]
    rot_gt = box[:, 4]

    cos_r = torch.cos(-rot_gt)
    sin_r = torch.sin(-rot_gt)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)

    #  (N, M, 2) = (N, 1, 2) - (1, M, 2)
    shifted_points: Float[Tensor, "N M 2"] = point[:, None] - cxcy_gt[None, :, :]
    rotated_points = torch.einsum("nmi,mij->nmj", shifted_points, R)

    half_wh = wh_gt / 2
    inside_xy = rotated_points.abs() <= half_wh[:, None, :]
    inside = inside_xy.all(dim=-1)

    return inside


def distance_between_points(
    points_a: Float[Tensor, "N 2"], points_b: Float[Tensor, "M 2"]
) -> Float[Tensor, "N M"]:
    """
    Compute the pairwise distance between two sets of points.

    Args:
        points_a: First set of points, shape (N, 2).
        points_b: Second set of points, shape (M, 2).
    Returns:
        A tensor of shape (N, M) where each element [i, j] is the distance between points_a[i] and points_b[j].
    """
    # (N, M, 2) = (N, 1, 2) - (1, M, 2)
    diff = points_a[:, None] - points_b[None, :, :]
    dist_squared = (diff**2).sum(dim=-1)
    return torch.sqrt(dist_squared)


def within_certain_region(
    points_a: Float[Tensor, "N 2"],
    points_b: Float[Tensor, "M 2"],
    radius: float,
) -> Bool[Tensor, "N M"]:
    """
    Check if points_a are within a certain radius of points_b.

    Args:
        points_a: Points to be checked, shape (N, 2).
        points_b: Reference points, shape (M, 2).
        radius: The radius within which points_a should be to points_b.
    Returns:
        boolean tensor of shape (N, M) indicating whether each point in points_a is within the radius of each point in points_b.
    """
    distances = distance_between_points(points_a, points_b)
    return distances <= radius


def compute_multiple_priors(
    image_shape: tuple[int, int],
    strides: Sequence[int],
) -> tuple[Float[Tensor, "total_priors 2"], Float[Tensor, "total_priors"]]:
    """Compute priors for multiple FPN levels."""
    priors_list = [compute_priors(image_shape, stride) for stride in strides]
    stride_flat_tensor = torch.cat(
        [
            torch.full((priors.shape[0],), stride)
            for priors, stride in zip(priors_list, strides)
        ],
        dim=0,
    )
    priors_flat_tensor = torch.cat(priors_list, dim=0)
    return priors_flat_tensor, stride_flat_tensor


def decode_xywh_from_ltbr_and_priors(
    priors: Float[Tensor, "num_priors 2"],
    reg_preds: Float[Tensor, "num_priors 4"],
) -> Float[Tensor, "num_priors 4"]:
    """
    Decodes into (cx, cy, w, h) format.
    """
    l_shift = reg_preds[:, 0]
    t_shift = reg_preds[:, 1]
    r_shift = reg_preds[:, 2]
    b_shift = reg_preds[:, 3]

    w = l_shift + r_shift
    h = t_shift + b_shift

    x_prior = priors[:, 0]
    y_prior = priors[:, 1]

    cx = x_prior + (r_shift - l_shift) / 2
    cy = y_prior + (b_shift - t_shift) / 2

    return torch.stack([cx, cy, w, h], dim=-1)


def compute_priors(
    image_shape: tuple[int, int],
    stride: int = 1,
) -> Float[Tensor, "num_priors 2"]:
    """Compute priors for a single FPN level."""
    downsampled_shape = get_image_shape_after_stride(image_shape, stride)
    priors = get_center_grid(downsampled_shape) * stride
    priors = priors.view(-1, 2)
    return priors


def get_center_grid(
    shape: tuple[int, int],
) -> Float[Tensor, "h w 2"]:
    height, width = shape
    x_shift = torch.arange(0, width) + 0.5
    y_shift = torch.arange(0, height) + 0.5

    xx_shift, yy_shift = torch.meshgrid(x_shift, y_shift, indexing="xy")

    priors = torch.stack((xx_shift, yy_shift), dim=-1)
    return priors
