import math
from typing import Any

import torch
from jaxtyping import Float, Int
import torch.nn.functional as F

from rtmdet.assigner import DynamicSoftLabelAssigner
from rtmdet.dataset import OrientedBoundingBoxBatch
from rtmdet.model import RotatedRTMDetOutput
from rtmdet.ops import compute_multiple_priors, ltbr_angle_priors2xywhr


def _get_covariance_matrix(
    boxes: Float[torch.Tensor, "N 5"],
) -> tuple[
    Float[torch.Tensor, "N 1"],
    Float[torch.Tensor, "N 1"],
    Float[torch.Tensor, "N 1"],
]:
    """Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Covariance matrix components (a, b, c) where the covariance
            matrix is [[a, c], [c, b]], each of shape (N, 1).
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(
    obb1: Float[torch.Tensor, "N 5"],
    obb2: Float[torch.Tensor, "N 5"],
    CIoU: bool = False,
    eps: float = 1e-7,
) -> Float[torch.Tensor, "N 1"]:
    """Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(
    obb1: Float[torch.Tensor, "N 5"] | Any,
    obb2: Float[torch.Tensor, "M 5"] | Any,
    eps: float = 1e-7,
) -> Float[torch.Tensor, "N M"]:
    """Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = obb1 if isinstance(obb1, torch.Tensor) else torch.as_tensor(obb1)
    obb2 = obb2 if isinstance(obb2, torch.Tensor) else torch.as_tensor(obb2)

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def quality_focal_loss(
    pred: Float[torch.Tensor, "num_priors num_classes"],
    target: Float[torch.Tensor, "num_priors num_classes"],
    weight: Float[torch.Tensor, "num_priors"] | None = None,
    alpha: float = 0.25,
    beta: float = 2.0,
    eps: float = 1e-7,
) -> Float[torch.Tensor, ""]:
    """Quality Focal Loss for classification.

    Args:
        pred: Predicted logits (before sigmoid)
        target: Soft labels (one-hot * IoU weight)
        weight: Optional per-prior weight
        alpha: Alpha parameter
        beta: Beta parameter

    Returns:
        Scalar loss
    """
    pred_sigmoid = pred.sigmoid()
    pt = (pred_sigmoid - target).abs().pow(beta)
    focal_weight = ((1 - target) * alpha + target * (1 - alpha)) * pt
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    loss = focal_weight * bce

    if weight is not None:
        loss = loss.sum(dim=-1) * weight
    else:
        loss = loss.sum(dim=-1)

    return loss.mean()


class RotatedTDMDetLoss(torch.nn.Module):
    image_shape: tuple[int, int]
    strides: list[int]
    assigner: DynamicSoftLabelAssigner

    def __init__(
        self,
        image_shape: tuple[int, int],
        k: int = 13,
        eps: float = 1e-7,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self.assigner = DynamicSoftLabelAssigner(k=k)
        self.eps = eps
        self.strides = strides or [8, 16, 32]
        self.image_shape = image_shape

    def forward_pass_single(
        self,
        gt_boxes: Float[torch.Tensor, "N 5"],
        gt_labels: Int[torch.Tensor, "N"],
        pred_boxes: Float[torch.Tensor, "num_priors 5"],
        pred_cls_logits: Float[torch.Tensor, "num_priors num_classes"],
        priors: Float[torch.Tensor, "num_priors 4"],
        strides: torch.Tensor,
        device: torch.device,
        num_classes: int,
    ) -> tuple[torch.Tensor, int]:
        """Process loss computation for a single image.

        Args:
            gt_boxes: Ground truth boxes, shape (N, 5) xywhr format.
            gt_labels: Ground truth labels, shape (N,).
            pred_boxes: Predicted boxes for all priors, shape (num_priors, 5).
            pred_cls_logits: Predicted class logits, shape (num_priors, num_classes).
            priors: Anchor priors.
            strides: Feature strides.
            device: Device for tensor creation.
            num_classes: Number of classes.

        Returns:
            Tuple of (loss, num_positive_samples).
        """
        num_priors = pred_boxes.shape[0]

        gt_cls = F.one_hot(gt_labels, num_classes=num_classes)

        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            neg_target = torch.zeros(num_priors, num_classes, device=device)
            cls_loss = quality_focal_loss(
                pred=pred_cls_logits,
                target=neg_target,
                weight=None,
            )
            return cls_loss, 0

        pred_cls = pred_cls_logits.softmax(dim=-1)
        assigned_labels, assigned_ious = self.assigner.assign(
            gt_boxes=gt_boxes,
            gt_cls=gt_cls,
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            strides=strides,
        )

        pos_mask = assigned_labels >= 0
        num_pos = pos_mask.sum().item()

        gt_cls_full = torch.zeros(num_priors, num_classes, device=device)
        gt_cls_full[assigned_labels[pos_mask]] = gt_cls[assigned_labels[pos_mask]]

        cls_weight = torch.ones_like(assigned_ious)
        cls_weight[~pos_mask] = 0.1

        cls_loss = quality_focal_loss(
            pred=pred_cls_logits,
            target=gt_cls_full,
            weight=cls_weight,
        )

        if num_pos > 0:
            pos_idx = pos_mask.nonzero().squeeze(-1)
            matched_gt_idx = assigned_labels[pos_idx]
            pos_gt_boxes = gt_boxes[matched_gt_idx]
            pos_pred_boxes = pred_boxes[pos_idx]

            pos_ious = probiou(pos_pred_boxes, pos_gt_boxes, CIoU=True)
            bbox_loss = (1 - pos_ious).mean()

            return cls_loss + bbox_loss, num_pos

        return cls_loss, 0

    def forward(
        self, x: OrientedBoundingBoxBatch, y: RotatedRTMDetOutput
    ) -> torch.Tensor:
        """Calculate the loss for a batch of oriented bounding boxes.

        Args:
            x (OrientedBoundingBoxBatch): A batch of oriented bounding boxes, containing images, boxes, and labels.
            y (RotatedRTMDetOutput): The output from the RotatedRTMDet model, containing predicted boxes and class scores.

        Returns:
            (torch.Tensor): The calculated loss for the batch.
        """
        device = x.images.device
        batch_size = x.images.shape[0]

        priors, strides = compute_multiple_priors(
            image_shape=self.image_shape, strides=self.strides, device=device
        )

        pred_boxes = ltbr_angle_priors2xywhr(y.ltbr_reg, y.angle_preds, priors)
        pred_cls_logits = y.cls_logits
        num_classes = pred_cls_logits.shape[-1]

        total_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            loss_b, _ = self.forward_pass_single(
                gt_boxes=x.boxes[b],
                gt_labels=x.labels[b],
                pred_boxes=pred_boxes,
                pred_cls_logits=pred_cls_logits,
                priors=priors,
                strides=strides,
                device=device,
                num_classes=num_classes,
            )
            total_loss = total_loss + loss_b

        return total_loss
