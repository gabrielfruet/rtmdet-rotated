from rtmdet.dataset import OrientedBoundingBoxBatch
import pytest
import torch
from torchvision import tv_tensors

from rtmdet.loss import RotatedTDMDetLoss


class TestRotatedTDMDetLoss:
    @pytest.fixture
    def example_rtmdet_loss(self) -> RotatedTDMDetLoss:
        from rtmdet.loss import RotatedTDMDetLoss

        return RotatedTDMDetLoss(
            image_shape=(512, 512),
        )

    def test__forward_single_pass__simple(self, example_rtmdet_loss: RotatedTDMDetLoss):
        gt_boxes = torch.tensor([[5, 5, 10, 10, 0]], dtype=torch.float32)
        gt_labels = torch.tensor([1])

        pred_boxes = torch.tensor([[5, 5, 10, 10, 0]], dtype=torch.float32)

        pred_cls_logits = torch.tensor([[0.1, 0.9]])

        strides = torch.tensor([8], dtype=torch.float32)

        example_rtmdet_loss.forward_pass_single(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_cls_logits=pred_cls_logits,
            strides=strides,
            num_classes=2,
            device=gt_boxes.device,
        )

    def test__forward_single_pass__simple_close_to_zero(
        self, example_rtmdet_loss: RotatedTDMDetLoss
    ):
        gt_boxes = torch.tensor([[5, 5, 10, 10, 0]], dtype=torch.float32)
        gt_labels = torch.tensor([1])

        pred_boxes = torch.tensor([[5, 5, 10, 10, 0]], dtype=torch.float32)

        pred_cls_logits = torch.tensor([[-100.0, 100.0]])

        strides = torch.tensor([32], dtype=torch.float32)

        loss, _ = example_rtmdet_loss.forward_pass_single(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_cls_logits=pred_cls_logits,
            strides=strides,
            num_classes=2,
            device=gt_boxes.device,
        )

        assert torch.isclose(loss, torch.tensor(0.0), atol=0.001)

    def test__forward_single_pass__no_ground_truth(
        self, example_rtmdet_loss: RotatedTDMDetLoss
    ):
        gt_boxes = torch.empty((0, 5), dtype=torch.float32)
        gt_labels = torch.empty(0, dtype=torch.int64)

        pred_boxes = torch.tensor([[5, 5, 10, 10, 0]], dtype=torch.float32)

        pred_cls_logits = torch.tensor([[-100.0, 100.0]])

        strides = torch.tensor([32], dtype=torch.float32)

        example_rtmdet_loss.forward_pass_single(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_cls_logits=pred_cls_logits,
            strides=strides,
            num_classes=2,
            device=gt_boxes.device,
        )

    def test__forward_single_pass__no_pred_raises_error(
        self, example_rtmdet_loss: RotatedTDMDetLoss
    ):
        gt_boxes = torch.tensor([[5, 5, 10, 10, 0]], dtype=torch.float32)
        gt_labels = torch.tensor([1])

        pred_boxes = torch.empty((0, 5), dtype=torch.float32)

        pred_cls_logits = torch.empty((0, 2), dtype=torch.int64)

        strides = torch.empty(0, dtype=torch.float32)

        with pytest.raises((RuntimeError, IndexError)):
            example_rtmdet_loss.forward_pass_single(
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_cls_logits=pred_cls_logits,
                strides=strides,
                num_classes=2,
                device=gt_boxes.device,
            )

    def test__forward_single_pass__multiple(
        self, example_rtmdet_loss: RotatedTDMDetLoss
    ):
        gt_boxes = torch.tensor(
            [[5, 5, 10, 10, 0], [5, 5, 10, 10, 0]], dtype=torch.float32
        )
        gt_labels = torch.tensor([1, 1])

        pred_boxes = torch.tensor(
            [[5, 5, 10, 10, 0], [5, 5, 10, 10, 0]], dtype=torch.float32
        )

        pred_cls_logits = torch.tensor([[0.1, 0.9], [0.1, 0.9]])

        strides = torch.tensor([8, 32], dtype=torch.float32)

        example_rtmdet_loss.forward_pass_single(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_cls_logits=pred_cls_logits,
            strides=strides,
            num_classes=2,
            device=gt_boxes.device,
        )
