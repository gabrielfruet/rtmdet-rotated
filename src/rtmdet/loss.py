import torch
from jaxtyping import Float, Int
import torch.nn.functional as F

from rtmdet.assigner import DynamicSoftLabelAssigner
from rtmdet.dataset import OrientedBoundingBoxBatch
from rtmdet.model import RotatedRTMDetOutput
from rtmdet.ops import compute_multiple_priors, ltbr_angle_priors2xywhr

from rtmdet.probiou import probiou
from rtmdet.typecheck import typechecker


@typechecker
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


class RotatedRTMDetLoss(torch.nn.Module):
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
        strides: torch.Tensor,
        num_classes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """Process loss computation for a single image.

        Args:
            gt_boxes: Ground truth boxes, shape (N, 5) xywhr format.
            gt_labels: Ground truth labels, shape (N,).
            pred_boxes: Predicted boxes for all priors, shape (num_priors, 5).
            pred_cls_logits: Predicted class logits, shape (num_priors, num_classes).
            strides: Feature strides.
            num_classes: Number of classes.
            device: Device for tensor creation.

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

        gt_cls_full = torch.zeros(
            num_priors, num_classes, device=device, dtype=torch.float32
        )
        gt_cls_full[assigned_labels[pos_mask]] = (
            gt_cls[assigned_labels[pos_mask]] * assigned_ious
        )

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

            return cls_loss + bbox_loss, int(num_pos)

        return cls_loss, 0

    def forward(
        self, target: OrientedBoundingBoxBatch, pred: RotatedRTMDetOutput
    ) -> torch.Tensor:
        """Calculate the loss for a batch of oriented bounding boxes.

        Args:
            target (OrientedBoundingBoxBatch): A batch of oriented bounding boxes, containing images, boxes, and labels.
            pred (RotatedRTMDetOutput): The output from the RotatedRTMDet model, containing predicted boxes and class scores.

        Returns:
            (torch.Tensor): The calculated loss for the batch.
        """
        device = target.images.device
        batch_size = target.images.shape[0]

        priors, strides = compute_multiple_priors(
            image_shape=self.image_shape, strides=self.strides, device=device
        )

        pred_boxes = ltbr_angle_priors2xywhr(pred.ltbr_reg, pred.angle_preds, priors)
        pred_cls_logits = pred.cls_logits
        num_classes = pred_cls_logits.shape[-1]

        total_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            loss_b, _ = self.forward_pass_single(
                gt_boxes=target.boxes[b],
                gt_labels=target.labels[b],
                pred_boxes=pred_boxes,
                pred_cls_logits=pred_cls_logits,
                strides=strides,
                device=device,
                num_classes=num_classes,
            )
            total_loss = total_loss + loss_b

        return total_loss

    def __call__(
        self, target: OrientedBoundingBoxBatch, pred: RotatedRTMDetOutput
    ) -> torch.Tensor:
        return super.__call__(target, pred)
