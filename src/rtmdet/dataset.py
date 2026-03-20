from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
from torchvision import tv_tensors
from jaxtyping import Float, Int
from PIL import Image as PILImage
import numpy.typing as npt
from torch.utils.data import Dataset

NDArray4Corners = Float[npt.NDArray, "N 8"]  # (x0, y0, x1, y1, x2, y2, x3, y3) format
NDArrayOBBoxes = Float[
    npt.NDArray, "N 5"
]  # (cx, cy, w, h, angle) format, normalized to [0,1]

DOTA_DEFAULT_CLASS_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]


class OrientedBoundingBoxSample(NamedTuple):
    image: Float[tv_tensors.Image, "C H W"]
    boxes: Float[torch.Tensor, "N 5"]  # xywhr format, absolute coords
    labels: Int[torch.Tensor, "N"]


class OrientedBoundingBoxBatch(NamedTuple):
    images: Float[tv_tensors.Image, "B C H W"]
    boxes: list[Float[torch.Tensor, "N 5"]]  # xywhr format, absolute coords
    labels: list[Int[torch.Tensor, "N"]]


def dota_collate_fn(batch: list[OrientedBoundingBoxSample]) -> OrientedBoundingBoxBatch:
    return OrientedBoundingBoxBatch(
        images=tv_tensors.Image(torch.stack([sample.image for sample in batch])),
        boxes=[sample.boxes for sample in batch],
        labels=[sample.labels for sample in batch],
    )


def oriented_bbox_from_corners(corners: NDArray4Corners) -> NDArrayOBBoxes:
    """Convert 4-corner format to (cx, cy, w, h, angle) format.

    Args:
        corners:
            Array of shape (n_boxes, 8) with coordinates
            (x0, y0, x1, y1, x2, y2, x3, y3)
            Corners should be in clockwise or counter-clockwise order.

    Returns:
        Array of shape (n_boxes, 5) with (cx, cy, w, h, angle)
        coordinates. Angle is in radians, in the range [-pi/2, pi/2].
    """
    if corners.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)

    pts = corners.reshape(-1, 4, 2)

    cx = (pts[:, 0, 0] + pts[:, 2, 0]) / 2
    cy = (pts[:, 0, 1] + pts[:, 2, 1]) / 2

    width = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    height = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    angle = np.arctan2(pts[:, 1, 1] - pts[:, 0, 1], pts[:, 1, 0] - pts[:, 0, 0])

    angle_over = angle > np.pi / 2
    angle_under = angle < -np.pi / 2

    angle[angle_over] -= np.pi
    angle[angle_under] += np.pi

    return np.stack([cx, cy, width, height, angle], axis=1)


def load_labels(
    label_path: Path, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load labels from a DOTA format label file.

    Args:
        label_path: Path to the label file.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of (boxes, labels) where boxes is (N, 5) tensor in CXCYWHR format
        and labels is (N,) tensor of class indices.
    """
    if not label_path.exists():
        return torch.empty(0, 5), torch.empty(0, dtype=torch.long)

    data = np.loadtxt(label_path, ndmin=2)
    if data.size == 0:
        return torch.empty(0, 5), torch.empty(0, dtype=torch.long)

    labels = torch.from_numpy(data[:, 0]).long()
    corners = data[:, 1:]  # shape (N, 8), normalized

    boxes_normalized = oriented_bbox_from_corners(corners)

    boxes_absolute = denormalize_boxes(boxes_normalized, width, height)
    boxes = torch.from_numpy(boxes_absolute).float()

    return boxes, labels


def denormalize_boxes(boxes: NDArrayOBBoxes, width: int, height: int) -> NDArrayOBBoxes:
    """Denormalize bounding boxes from [0,1] to absolute pixel coordinates.

    Args:
        boxes: Array of shape (N, 5) with normalized (cx, cy, w, h, angle).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Array of shape (N, 5) with absolute (cx, cy, w, h, angle).
    """
    boxes = boxes.copy()
    boxes[:, 0] *= width  # cx
    boxes[:, 2] *= width  # w
    boxes[:, 1] *= height  # cy
    boxes[:, 3] *= height  # h
    return boxes


def normalize_boxes(boxes: NDArrayOBBoxes, width: int, height: int) -> NDArrayOBBoxes:
    """Normalize bounding boxes from absolute pixel coordinates to [0,1].

    Args:
        boxes: Array of shape (N, 5) with absolute (cx, cy, w, h, angle).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Array of shape (N, 5) with normalized (cx, cy, w, h, angle).
    """
    boxes = boxes.copy()
    boxes[:, 0] /= width  # cx
    boxes[:, 2] /= width  # w
    boxes[:, 1] /= height  # cy
    boxes[:, 3] /= height  # h
    return boxes


class DOTADataset(Dataset[OrientedBoundingBoxSample]):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        class_names: list[str] | None = None,
        transform: Callable | None = None,
    ):
        self.root = root
        self.split = split
        self.class_names = class_names or DOTA_DEFAULT_CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform

        # DOTA128 structure: images/{split}/ and labels/{split}/
        img_dir = root / "images" / split
        self.label_dir = root / "labels" / split

        # Scan for image files
        self.image_paths = []
        if img_dir.exists():
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                self.image_paths.extend(sorted(img_dir.glob(ext)))
                self.image_paths.extend(sorted(img_dir.glob(ext.upper())))
        # Deduplicate
        self.image_paths = list(set(self.image_paths))
        self.image_paths.sort()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> OrientedBoundingBoxSample:
        img_path = self.image_paths[idx]

        # Load image as PIL for torchvision v2 transforms compatibility
        img = PILImage.open(img_path).convert("RGB")
        W, H = img.size

        # Convert to tv_tensors.Image for v2 transforms compatibility
        img = tv_tensors.Image(img)

        # Load annotations
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes, labels = load_labels(label_path, W, H)

        boxes = tv_tensors.BoundingBoxes(  # type: ignore
            boxes,
            format=tv_tensors.BoundingBoxFormat.CXCYWHR,
            canvas_size=(H, W),
        )

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)

        sample = OrientedBoundingBoxSample(image=img, boxes=boxes, labels=labels)

        return sample
