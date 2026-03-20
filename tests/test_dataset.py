import numpy as np
import numpy.testing as npt

from rtmdet.dataset import oriented_bbox_from_corners


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
