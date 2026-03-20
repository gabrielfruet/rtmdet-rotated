from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from PIL import Image
from torch.utils.data import Dataset

NDArray4Corners = npt.NDArray[np.floating]
NDArrayOBBoxes = npt.NDArray[np.floating]

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
    image: Float[torch.Tensor, "C H W"]
    boxes: Float[torch.Tensor, "N 5"]  # xywhr format
    labels: Int[torch.Tensor, "N"]


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


class DOTADataset(Dataset[OrientedBoundingBoxSample]):
    def __init__(
        self, root: Path, split: str = "train", class_names: list[str] | None = None
    ):
        self.root = root
        self.split = split
        self.class_names = class_names or DOTA_DEFAULT_CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

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

        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Load annotations
        label_path = self.label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            data = np.loadtxt(label_path)
            if data.size == 0:
                boxes = torch.empty(0, 5)
                labels = torch.empty(0, dtype=torch.long)
            else:
                # Handle single bbox case (no leading dimension)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                labels = torch.from_numpy(data[:, 0]).long()
                corners = data[:, 1:]  # shape (N, 8), normalized

                # Convert corners to xywh
                boxes_normalized = oriented_bbox_from_corners(corners)  # (N, 5)

                # Denormalize: cx, w use W; cy, h use H
                boxes_normalized[:, 0] *= W  # cx
                boxes_normalized[:, 2] *= W  # w
                boxes_normalized[:, 1] *= H  # cy
                boxes_normalized[:, 3] *= H  # h

                boxes = torch.from_numpy(boxes_normalized).float()
        else:
            boxes = torch.empty(0, 5)
            labels = torch.empty(0, dtype=torch.long)

        return OrientedBoundingBoxSample(image=image, boxes=boxes, labels=labels)
