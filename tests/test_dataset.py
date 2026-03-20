import tempfile
import uuid
from pathlib import Path

import numpy as np
import numpy.testing as npt
import torch
from torchvision import tv_tensors

from rtmdet.dataset import (
    DOTADataset,
    denormalize_boxes,
    load_labels,
    normalize_boxes,
    oriented_bbox_from_corners,
)


def test_oriented_bbox_from_corners_empty() -> None:
    corners = np.array([], dtype=np.float64).reshape(0, 8)
    result = oriented_bbox_from_corners(corners)
    assert result.shape == (0, 5)
    assert result.dtype == np.float64


def test_oriented_bbox_from_corners_single_box_horizontal() -> None:
    # Horizontal box: corners (0,0), (4,0), (4,2), (0,2)
    # Width = 4, Height = 2, angle = 0
    # Center = (2, 1)
    corners = np.array([[0, 0, 4, 0, 4, 2, 0, 2]], dtype=np.float64)
    result = oriented_bbox_from_corners(corners)

    expected = np.array([[2.0, 1.0, 4.0, 2.0, 0.0]])
    npt.assert_allclose(result, expected, rtol=1e-5)


def test_oriented_bbox_from_corners_single_box_vertical() -> None:
    # Vertical box: corners (0,0), (0,4), (2,4), (2,0)
    # Width = 4 (edge 0->1), Height = 2 (edge 1->2)
    # angle = atan2(4-0, 0-0) = atan2(4, 0) = pi/2 (after normalization)
    # Center = (1, 2)
    corners = np.array([[0, 0, 0, 4, 2, 4, 2, 0]], dtype=np.float64)
    result = oriented_bbox_from_corners(corners)

    # Angle is at boundary, can be either pi/2 or -pi/2 (same orientation)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 2.0
    assert result[0, 2] == 4.0
    assert result[0, 3] == 2.0
    assert result[0, 4] in (-np.pi / 2, np.pi / 2)


def test_oriented_bbox_from_corners_rotated() -> None:
    # 45-degree rotated box: center at (0,0), w=4, h=2
    # corners for a box rotated 45 degrees
    # For simplicity, let's use a box at origin rotated 45 degrees
    # A box with width=4, height=2, angle=pi/4
    import math

    angle = math.pi / 4
    w, h = 4.0, 2.0
    cx, cy = 0.0, 0.0

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = np.array(
        [
            [
                cx + (-w / 2) * cos_a - (-h / 2) * sin_a,
                cy + (-w / 2) * sin_a + (-h / 2) * cos_a,
                cx + (w / 2) * cos_a - (-h / 2) * sin_a,
                cy + (w / 2) * sin_a + (-h / 2) * cos_a,
                cx + (w / 2) * cos_a - (h / 2) * sin_a,
                cy + (w / 2) * sin_a + (h / 2) * cos_a,
                cx + (-w / 2) * cos_a - (h / 2) * sin_a,
                cy + (-w / 2) * sin_a + (h / 2) * cos_a,
            ]
        ],
        dtype=np.float64,
    )
    result = oriented_bbox_from_corners(corners)

    expected = np.array([[cx, cy, w, h, angle]])
    npt.assert_allclose(result, expected, rtol=1e-5)


def test_oriented_bbox_from_corners_angle_normalization_positive() -> None:
    # Create a box that would produce angle > pi/2
    # Edge 0->1 going in a direction that gives angle > pi/2
    # e.g., angle = 3*pi/4 should become -pi/4
    import math

    original_angle = 3 * math.pi / 4  # 135 degrees
    w, h = 4.0, 2.0
    cx, cy = 0.0, 0.0

    cos_a, sin_a = math.cos(original_angle), math.sin(original_angle)
    corners = np.array(
        [
            [
                cx + (-w / 2) * cos_a - (-h / 2) * sin_a,
                cy + (-w / 2) * sin_a + (-h / 2) * cos_a,
                cx + (w / 2) * cos_a - (-h / 2) * sin_a,
                cy + (w / 2) * sin_a + (-h / 2) * cos_a,
                cx + (w / 2) * cos_a - (h / 2) * sin_a,
                cy + (w / 2) * sin_a + (h / 2) * cos_a,
                cx + (-w / 2) * cos_a - (h / 2) * sin_a,
                cy + (-w / 2) * sin_a + (h / 2) * cos_a,
            ]
        ],
        dtype=np.float64,
    )
    result = oriented_bbox_from_corners(corners)

    # After normalization, angle should be in [-pi/2, pi/2]
    assert result[0, 4] <= np.pi / 2
    assert result[0, 4] >= -np.pi / 2
    # Since original was 3*pi/4, after subtract pi it becomes -pi/4
    expected_angle = -np.pi / 4
    npt.assert_allclose(result[0, 4], expected_angle, rtol=1e-5)


def test_oriented_bbox_from_corners_angle_normalization_negative() -> None:
    # Create a box that would produce angle < -pi/2
    # e.g., angle = -3*pi/4 should become pi/4
    import math

    original_angle = -3 * math.pi / 4  # -135 degrees
    w, h = 4.0, 2.0
    cx, cy = 0.0, 0.0

    cos_a, sin_a = math.cos(original_angle), math.sin(original_angle)
    corners = np.array(
        [
            [
                cx + (-w / 2) * cos_a - (-h / 2) * sin_a,
                cy + (-w / 2) * sin_a + (-h / 2) * cos_a,
                cx + (w / 2) * cos_a - (-h / 2) * sin_a,
                cy + (w / 2) * sin_a + (-h / 2) * cos_a,
                cx + (w / 2) * cos_a - (h / 2) * sin_a,
                cy + (w / 2) * sin_a + (h / 2) * cos_a,
                cx + (-w / 2) * cos_a - (h / 2) * sin_a,
                cy + (-w / 2) * sin_a + (h / 2) * cos_a,
            ]
        ],
        dtype=np.float64,
    )
    result = oriented_bbox_from_corners(corners)

    # After normalization, angle should be in [-pi/2, pi/2]
    assert result[0, 4] <= np.pi / 2
    assert result[0, 4] >= -np.pi / 2
    # Since original was -3*pi/4, after adding pi it becomes pi/4
    expected_angle = np.pi / 4
    npt.assert_allclose(result[0, 4], expected_angle, rtol=1e-5)


def test_oriented_bbox_from_corners_batch() -> None:
    # Multiple boxes
    # Box 1: horizontal, center (2,1), w=4, h=2
    # Box 2: vertical, center (1,2), w=4, h=2
    corners = np.array(
        [
            [0, 0, 4, 0, 4, 2, 0, 2],  # horizontal box
            [0, 0, 0, 4, 2, 4, 2, 0],  # vertical box
        ],
        dtype=np.float64,
    )
    result = oriented_bbox_from_corners(corners)

    assert result.shape == (2, 5)
    # Box 1
    npt.assert_allclose(result[0, :4], [2.0, 1.0, 4.0, 2.0], rtol=1e-5)
    assert result[0, 4] == 0.0
    # Box 2 - angle at boundary
    npt.assert_allclose(result[1, :4], [1.0, 2.0, 4.0, 2.0], rtol=1e-5)
    assert result[1, 4] in (-np.pi / 2, np.pi / 2)


def test_oriented_bbox_from_corners_center_calculation() -> None:
    # Test center calculation specifically
    # Corners at (1,1), (5,1), (5,3), (1,3) - center should be (3, 2)
    corners = np.array([[1, 1, 5, 1, 5, 3, 1, 3]], dtype=np.float64)
    result = oriented_bbox_from_corners(corners)

    npt.assert_allclose(result[0, 0], 3.0, rtol=1e-5)
    npt.assert_allclose(result[0, 1], 2.0, rtol=1e-5)


def test_denormalize_boxes_empty() -> None:
    boxes = np.array([], dtype=np.float64).reshape(0, 5)
    result = denormalize_boxes(boxes, width=1024, height=682)
    assert result.shape == (0, 5)
    assert result.dtype == np.float64


def test_denormalize_boxes_single_box() -> None:
    # Normalized box: cx=0.5, cy=0.5, w=0.2, h=0.3, angle=0.1
    # Image: 100x100
    # Expected: cx=50, cy=50, w=20, h=30, angle=0.1
    boxes = np.array([[0.5, 0.5, 0.2, 0.3, 0.1]], dtype=np.float64)
    result = denormalize_boxes(boxes, width=100, height=100)

    expected = np.array([[50.0, 50.0, 20.0, 30.0, 0.1]], dtype=np.float64)
    npt.assert_allclose(result, expected, rtol=1e-5)


def test_denormalize_boxes_different_width_height() -> None:
    # Normalized box in 1024x682 image
    # cx=0.5, cy=0.5, w=0.25, h=0.5, angle=0.3
    # Expected: cx=512, cy=341, w=256, h=341, angle=0.3
    boxes = np.array([[0.5, 0.5, 0.25, 0.5, 0.3]], dtype=np.float64)
    result = denormalize_boxes(boxes, width=1024, height=682)

    expected = np.array([[512.0, 341.0, 256.0, 341.0, 0.3]], dtype=np.float64)
    npt.assert_allclose(result, expected, rtol=1e-5)


def test_denormalize_boxes_batch() -> None:
    boxes = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.5, 0.1, 0.1, -0.3],
        ],
        dtype=np.float64,
    )
    result = denormalize_boxes(boxes, width=100, height=200)

    assert result.shape == (2, 5)
    # Box 1: cx=10, cy=40, w=30, h=80
    npt.assert_allclose(result[0, :4], [10.0, 40.0, 30.0, 80.0], rtol=1e-5)
    assert result[0, 4] == 0.5  # angle unchanged
    # Box 2: cx=50, cy=100, w=10, h=20
    npt.assert_allclose(result[1, :4], [50.0, 100.0, 10.0, 20.0], rtol=1e-5)
    assert result[1, 4] == -0.3  # angle unchanged


def test_denormalize_boxes_preserves_angle() -> None:
    angles = [0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2]
    for angle in angles:
        boxes = np.array([[0.5, 0.5, 0.2, 0.2, angle]], dtype=np.float64)
        result = denormalize_boxes(boxes, width=100, height=100)
        assert result[0, 4] == angle


def test_normalize_boxes_empty() -> None:
    boxes = np.array([], dtype=np.float64).reshape(0, 5)
    result = normalize_boxes(boxes, width=1024, height=682)
    assert result.shape == (0, 5)
    assert result.dtype == np.float64


def test_normalize_boxes_single_box() -> None:
    # Absolute box: cx=50, cy=50, w=20, h=30, angle=0.1
    # Image: 100x100
    # Expected: cx=0.5, cy=0.5, w=0.2, h=0.3, angle=0.1
    boxes = np.array([[50.0, 50.0, 20.0, 30.0, 0.1]], dtype=np.float64)
    result = normalize_boxes(boxes, width=100, height=100)

    expected = np.array([[0.5, 0.5, 0.2, 0.3, 0.1]], dtype=np.float64)
    npt.assert_allclose(result, expected, rtol=1e-5)


def test_normalize_boxes_batch() -> None:
    boxes = np.array(
        [
            [10.0, 40.0, 30.0, 80.0, 0.5],
            [50.0, 100.0, 10.0, 20.0, -0.3],
        ],
        dtype=np.float64,
    )
    result = normalize_boxes(boxes, width=100, height=200)

    assert result.shape == (2, 5)
    npt.assert_allclose(result[0, :4], [0.1, 0.2, 0.3, 0.4], rtol=1e-5)
    assert result[0, 4] == 0.5  # angle unchanged
    npt.assert_allclose(result[1, :4], [0.5, 0.5, 0.1, 0.1], rtol=1e-5)
    assert result[1, 4] == -0.3  # angle unchanged


def test_normalize_boxes_roundtrip_denormalize() -> None:
    # Start with normalized, denormalize, then normalize back
    original = np.array([[0.5, 0.5, 0.2, 0.3, 0.1]], dtype=np.float64)
    denorm = denormalize_boxes(original, width=1024, height=682)
    restored = normalize_boxes(denorm, width=1024, height=682)
    npt.assert_allclose(restored, original, rtol=1e-5)


def test_denormalize_boxes_roundtrip_normalize() -> None:
    # Start with absolute, normalize, then denormalize back
    original = np.array([[512.0, 341.0, 256.0, 341.0, 0.3]], dtype=np.float64)
    norm = normalize_boxes(original, width=1024, height=682)
    restored = denormalize_boxes(norm, width=1024, height=682)
    npt.assert_allclose(restored, original, rtol=1e-5)


def test_normalize_boxes_one_pixel_dimensions() -> None:
    # Edge case: width or height = 1 (avoid div by zero issues)
    boxes = np.array([[0.5, 0.5, 0.5, 0.5, 0.1]], dtype=np.float64)
    result = normalize_boxes(boxes, width=1, height=1)
    npt.assert_allclose(result[0, :4], [0.5, 0.5, 0.5, 0.5], rtol=1e-5)


def test_load_labels_nonexistent_file(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist.txt"
    boxes, labels = load_labels(nonexistent, width=1024, height=682)

    assert boxes.shape == (0, 5)
    assert labels.shape == (0,)
    assert boxes.dtype == torch.float32
    assert labels.dtype == torch.long


def test_load_labels_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    boxes, labels = load_labels(empty_file, width=1024, height=682)

    assert boxes.shape == (0, 5)
    assert labels.shape == (0,)


def test_load_labels_single_box(tmp_path: Path) -> None:
    # DOTA format: label + 8 normalized corner coordinates
    # Box: center (0.5, 0.5), w=0.2, h=0.2, horizontal (angle=0)
    # corners in clockwise order: (0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)
    label_file = tmp_path / "single.txt"
    label_file.write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")

    boxes, labels = load_labels(label_file, width=100, height=100)

    assert boxes.shape == (1, 5)
    assert labels.shape == (1,)
    assert labels[0].item() == 0
    # cx=50, cy=50, w=20, h=20, angle=0
    npt.assert_allclose(boxes[0].numpy(), [50.0, 50.0, 20.0, 20.0, 0.0], rtol=1e-4)


def test_load_labels_multiple_boxes(tmp_path: Path) -> None:
    label_file = tmp_path / "multi.txt"
    # Two boxes: class 0 at top-left, class 1 at bottom-right
    label_file.write_text(
        "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n1 0.7 0.7 0.9 0.7 0.9 0.9 0.7 0.9\n"
    )

    boxes, labels = load_labels(label_file, width=100, height=100)

    assert boxes.shape == (2, 5)
    assert labels.shape == (2,)
    assert labels[0].item() == 0
    assert labels[1].item() == 1
    # Box 1: cx=20, cy=20, w=20, h=20
    npt.assert_allclose(boxes[0].numpy(), [20.0, 20.0, 20.0, 20.0, 0.0], rtol=1e-4)
    # Box 2: cx=80, cy=80, w=20, h=20
    npt.assert_allclose(boxes[1].numpy(), [80.0, 80.0, 20.0, 20.0, 0.0], rtol=1e-4)


def test_load_labels_class_indices_are_long() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        label_file = tmp_path / "test.txt"
        label_file.write_text("5 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

        _, labels = load_labels(label_file, width=100, height=100)

        assert labels.dtype == torch.long
        assert labels[0].item() == 5


def test_dataset_initialization(tmp_path: Path) -> None:
    # Create minimal dataset structure
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create dummy image files
    for i in range(3):
        (img_dir / f"P000{i}__682__0___0.jpg").touch()
        (lbl_dir / f"P000{i}__682__0___0.txt").write_text("")

    dataset = DOTADataset(tmp_path, split="train")

    assert len(dataset) == 3
    assert len(dataset.image_paths) == 3


def test_dataset_len(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(5):
        (img_dir / f"img_{i}.jpg").touch()
        (lbl_dir / f"img_{i}.txt").write_text("")

    dataset = DOTADataset(tmp_path, split="train")
    assert len(dataset) == 5


def test_dataset_getitem_structure(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create a dummy image (10x10 pixels)
    from PIL import Image

    dummy_img = Image.new("RGB", (10, 10))
    dummy_img.save(img_dir / "test.jpg")

    # Create a label file
    lbl_dir / "test.txt"

    dataset = DOTADataset(tmp_path, split="train")
    sample = dataset[0]

    from rtmdet.dataset import OrientedBoundingBoxSample

    assert isinstance(sample, OrientedBoundingBoxSample)
    assert hasattr(sample, "image")
    assert isinstance(sample.image, tv_tensors.Image)
    assert hasattr(sample, "boxes")
    assert isinstance(sample.boxes, tv_tensors.BoundingBoxes)
    assert hasattr(sample, "labels")
    assert isinstance(sample.labels, torch.Tensor)


def test_dataset_with_real_images(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create real-ish 100x100 image
    from PIL import Image

    dummy_img = Image.new("RGB", (100, 100), color="red")
    dummy_img.save(img_dir / "real_test.jpg")

    # Create label with one box: class 0, corners (10,10)-(30,10)-(30,30)-(10,30)
    # This is a 20x20 box centered at (20,20)
    (lbl_dir / "real_test.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

    dataset = DOTADataset(tmp_path, split="train")
    sample = dataset[0]

    assert sample.image.shape[0] == 3  # 3 channels (RGB)


def test_dataset_custom_class_names(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    (img_dir / "test.jpg").touch()
    (lbl_dir / "test.txt").write_text("")

    custom_names = ["car", "person", "bike"]
    dataset = DOTADataset(tmp_path, split="train", class_names=custom_names)

    assert dataset.class_names == custom_names
    assert dataset.class_to_idx == {"car": 0, "person": 1, "bike": 2}


def test_dataset_transform_applied(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    from PIL import Image

    dummy_img = Image.new("RGB", (100, 100))
    dummy_img.save(img_dir / "transform_test.jpg")
    lbl_dir / "transform_test.txt"

    transform_called = False

    def dummy_transform(img, boxes, labels):
        nonlocal transform_called
        transform_called = True
        return img, boxes, labels

    dataset = DOTADataset(tmp_path, split="train", transform=dummy_transform)
    _ = dataset[0]

    assert transform_called


def test_dataset_empty_label_file(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    from PIL import Image

    dummy_img = Image.new("RGB", (100, 100))
    dummy_img.save(img_dir / "empty_labels.jpg")
    lbl_dir / "empty_labels.txt"

    dataset = DOTADataset(tmp_path, split="train")
    sample = dataset[0]

    assert sample.boxes.shape[0] == 0
    assert sample.labels.shape[0] == 0
