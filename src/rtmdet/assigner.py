"""Assignment strategy for object detection."""

import torch
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor, nn
from torch.nn import functional as F
from beartype import beartype as typechecker

from rtmdet.probiou import batch_probiou
from rtmdet.ops import (
    distance_between_points,
    points_inside_oriented_boxes,
    within_certain_region,
)


class DynamicSoftLabelAssigner(nn.Module):
    def __init__(
        self,
        k: int = 13,
        lambda_1: float = 1,
        lambda_2: float = 3,
        lambda_3: float = 1,
        center_radius: float = 2.5,
    ):
        super().__init__()
        self.k = k
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.alpha = 10
        self.beta = 3
        self.center_radius = center_radius

    def compute_cost_from_cost_parts(
        self,
        cost_center: Float[Tensor, "num_gt num_priors"],
        cost_cls: Float[Tensor, "num_gt num_priors"],
        cost_iou: Float[Tensor, "num_gt num_priors"],
    ) -> Float[Tensor, "num_gt num_priors"]:
        return (
            self.lambda_1 * cost_iou
            + self.lambda_2 * cost_cls
            + self.lambda_3 * cost_center
        )

    def geometry_constrain(
        self,
        pred_boxes: Float[Tensor, "num_priors 5"],
        gt_boxes: Float[Tensor, "num_gt 5"],
    ) -> Bool[Tensor, "num_priors"]:
        valid_radius_mask = within_certain_region(
            points_a=pred_boxes[:, :2],
            points_b=gt_boxes[:, :2],
            radius=self.center_radius,
        )

        valid_inside_mask = points_inside_oriented_boxes(
            point=pred_boxes[:, :2],
            box=gt_boxes,
        )

        valid_mask = valid_radius_mask | valid_inside_mask

        return valid_mask.any(dim=0)

    def compute_pairwise_iou(
        self,
        *,
        gt_boxes: Float[Tensor, "num_gt 5"],
        pred_boxes: Float[Tensor, "num_priors 5"],
    ) -> Float[Tensor, "num_gt num_priors"]:
        """
        Compute IoU between predicted boxes and ground truth boxes.

        Args:
            pred_boxes: Predicted boxes in (cx, cy, w, h, angle) format, shape (num_priors, 5).
            gt_boxes: Ground truth boxes in (cx, cy, w, h, angle) format, shape (num_gt, 5).

        Returns:
            IoU matrix of shape (num_gt, num_priors) where each element [i, j] is the IoU between pred_boxes[j] and gt_boxes[i].
        """
        return batch_probiou(gt_boxes, pred_boxes)

    def compute_pairwise_iou_cost(
        self,
        pairwise_iou: Float[Tensor, "num_gt num_priors"],
        eps: float = 1e-7,
    ) -> Float[Tensor, "num_gt num_priors"]:
        return -torch.log(pairwise_iou + eps)

    def compute_pairwise_cls_cost(
        self,
        *,
        pred_cls: Float[Tensor, "num_priors num_classes"],
        gt_cls: Float[Tensor, "num_gt num_classes"],
        pairwise_iou: Float[Tensor, "num_gt num_priors"],
    ) -> Float[Tensor, "num_gt num_priors"]:
        """
        Compute classification cost between predicted class scores and ground truth class labels.

        Args:
            pred_cls: Predicted class scores (logits).
            gt_cls: Ground truth class labels in one-hot format.
            pairwise_iou: IoU matrix to weight the classification cost based on localization quality.

        Returns:
            Classification cost matrix of shape (num_gt, num_priors) where each element [i, j] is the cost of assigning pred_boxes[j] to gt_boxes[i] based on their class predictions
        """
        pred_cls_expanded = pred_cls[None, :, :]  # [1, num_priors, num_classes]
        gt_cls_expanded = gt_cls[:, None, :]  # [num_gt, 1, num_classes]
        iou_expanded = pairwise_iou[..., None]  # [num_gt, num_priors, 1]

        # Create the Soft Label (Target = 1 * IoU)
        soft_label = gt_cls_expanded * iou_expanded

        # Quality Focal Loss calculation
        scale_factor = soft_label - pred_cls_expanded.sigmoid()
        cost_cls = F.binary_cross_entropy_with_logits(
            pred_cls_expanded,
            soft_label,
            reduction="none",
        ) * scale_factor.abs().pow(2.0)

        return cost_cls.sum(dim=-1)

    def compute_pairwise_center_prior_cost(
        self,
        pred_boxes: Float[Tensor, "num_priors 5"],
        gt_boxes: Float[Tensor, "num_gt 5"],
        strides: Float[Tensor, "num_priors"],
    ) -> Float[Tensor, "num_gt num_priors"]:
        """
        Compute regression cost between predicted boxes and ground truth boxes.
        Args:
            pred_boxes: Predicted boxes in (cx, cy, w, h, angle) format, shape (num_priors, 5).
            gt_boxes: Ground truth boxes in (cx, cy, w, h, angle)
                format, shape (num_gt, 5).
        Returns:
            Regression cost matrix of shape (num_gt, num_priors) where each
            element [i, j] is the cost of assigning pred_boxes[j] to
            gt_boxes[i] based on their box regression predictions.
        """
        pred_centers = pred_boxes[:, :2]
        gt_centers = gt_boxes[:, :2]
        center_distances = distance_between_points(gt_centers, pred_centers)
        center_distances_normalized = center_distances / strides[None, :]
        return self.alpha ** (center_distances_normalized - self.beta)

    def dynamic_k_per_gt(
        self, pairwise_iou: Float[Tensor, "num_gt num_priors"]
    ) -> Int[Tensor, "num_gt"]:
        safe_k = min(self.k, pairwise_iou.shape[1])
        topk_ious, _ = pairwise_iou.topk(safe_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(dim=1).long(), min=1)
        return dynamic_ks

    def compute_dynamic_k_mask(
        self,
        *,
        pairwise_cost: Float[Tensor, "num_gt num_priors"],
        pairwise_iou: Float[Tensor, "num_gt num_priors"],
    ) -> Bool[Tensor, "num_gt num_priors"]:
        dynamic_ks = self.dynamic_k_per_gt(pairwise_iou)
        max_dynamic_k = int(dynamic_ks.max().item())
        safe_max_dynamic_k = min(max_dynamic_k, pairwise_cost.shape[1])
        topk_costs, topk_indices = pairwise_cost.topk(
            k=safe_max_dynamic_k, dim=1, largest=False
        )
        mask = torch.zeros_like(pairwise_cost, dtype=torch.bool)
        for gt_idx in range(pairwise_cost.shape[0]):
            k = dynamic_ks[gt_idx].item()
            dynamic_k_indices = topk_indices[gt_idx, :k]
            mask[gt_idx, dynamic_k_indices] = True

        return mask

    def handle_conflicts_in_mask(
        self,
        *,
        mask: Bool[Tensor, "num_gt num_priors"],
        pairwise_cost: Float[Tensor, "num_gt num_priors"],
    ) -> Bool[Tensor, "num_gt num_priors"]:
        # Find conflicts (multiple ground truths assigned to the same prior)
        num_priors = mask.shape[1]
        prior_assign_count = mask.sum(dim=0)
        conflicts = prior_assign_count > 1

        if not conflicts.any():
            return mask

        mask_dst = mask.clone()
        # For each conflicting prior, keep the assignment with lowest cost
        conflicting_prior_indices = torch.where(conflicts)[0]
        for prior_idx in conflicting_prior_indices:
            # Find all ground truths assigned to this prior
            assigned_gts = torch.where(mask[:, prior_idx])[0]
            # Keep the one with lowest cost
            costs = pairwise_cost[assigned_gts, prior_idx]
            max_cost_idx = torch.argmin(costs)
            # Remove all other assignments to this prior
            mask_dst[assigned_gts, prior_idx] = False
            # Keep the assignment with lowest cost
            mask_dst[assigned_gts[max_cost_idx], prior_idx] = True

        return mask_dst

    @torch.no_grad
    @jaxtyped(typechecker=typechecker)
    def assign(
        self,
        pred_boxes: Float[Tensor, "num_priors 5"],
        pred_cls: Float[Tensor, "num_priors num_classes"],
        gt_boxes: Float[Tensor, "num_gt 5"],
        gt_cls: Int[Tensor, "num_gt num_classes"],
        strides: Float[Tensor, "num_priors"],
    ):
        num_priors = pred_boxes.shape[0]
        valid_mask = self.geometry_constrain(pred_boxes, gt_boxes)

        valid_pred_boxes = pred_boxes[valid_mask]
        valid_pred_cls = pred_cls[valid_mask]
        valid_strides = strides[valid_mask]

        pairwise_iou: Float[Tensor, "num_gt num_valid_priors"] = (
            self.compute_pairwise_iou(pred_boxes=valid_pred_boxes, gt_boxes=gt_boxes)
        )

        pairwise_iou_cost = self.compute_pairwise_iou_cost(pairwise_iou)
        pairwise_cls_cost = self.compute_pairwise_cls_cost(
            pred_cls=valid_pred_cls, gt_cls=gt_cls, pairwise_iou=pairwise_iou
        )
        pairwise_reg_cost = self.compute_pairwise_center_prior_cost(
            valid_pred_boxes, gt_boxes, valid_strides
        )
        pairwise_cost = self.compute_cost_from_cost_parts(
            pairwise_reg_cost, pairwise_cls_cost, pairwise_iou_cost
        )

        dynamic_k_mask_unresolved = self.compute_dynamic_k_mask(
            pairwise_cost=pairwise_cost,
            pairwise_iou=pairwise_iou,
        )

        dynamic_k_mask = self.handle_conflicts_in_mask(
            mask=dynamic_k_mask_unresolved,
            pairwise_cost=pairwise_cost,
        )

        assigned_gt_inds = torch.full(
            (num_priors,), -1, dtype=torch.long, device=pred_boxes.device
        )
        assigned_ious = torch.zeros(
            (num_priors,), dtype=torch.float32, device=pred_boxes.device
        )

        if dynamic_k_mask.any():
            # Get the indices from the [num_gt, num_valid] mask
            matched_gt_idx, matched_valid_prior_idx = torch.where(dynamic_k_mask)

            # Translate valid indices back to global indices
            global_valid_indices = torch.where(valid_mask)[0]
            global_prior_idx = global_valid_indices[matched_valid_prior_idx]

            # Fill the return tensors
            assigned_gt_inds[global_prior_idx] = matched_gt_idx
            assigned_ious[global_prior_idx] = pairwise_iou[
                matched_gt_idx, matched_valid_prior_idx
            ]

        return assigned_gt_inds, assigned_ious
