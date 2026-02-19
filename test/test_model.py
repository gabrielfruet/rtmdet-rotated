import torch

from rtmdet.model import compute_priors


def test_get_center_grid():
    from rtmdet.model import get_center_grid

    image_shape = (224, 224)
    center_grid = get_center_grid(image_shape)
    assert center_grid.shape == (224, 224, 2)
    assert center_grid[0, 0].tolist() == [0.5, 0.5]
    assert center_grid[0, 1].tolist() == [1.5, 0.5]
    assert center_grid[1, 0].tolist() == [0.5, 1.5]
    assert center_grid[1, 1].tolist() == [1.5, 1.5]
    assert center_grid[-1, -1].tolist() == [223.5, 223.5]


def test_get_image_shape_after_stride():
    from rtmdet.model import get_image_shape_after_stride

    image_shape = (224, 224)
    stride = 4
    downsampled_shape = get_image_shape_after_stride(image_shape, stride)
    assert downsampled_shape == (56, 56)


def test_compute_priors():
    from rtmdet.model import compute_priors

    image_shape = (224, 224)
    stride = 4
    downsample_shape = (56, 56)
    priors = compute_priors(image_shape, stride)
    assert priors.shape == (56 * 56, 2)
    assert priors[0].tolist() == [2.0, 2.0]
    assert priors[1].tolist() == [6.0, 2.0]
    assert priors[56].tolist() == [2.0, 6.0]
    assert priors[57].tolist() == [6.0, 6.0]
    assert priors[-1].tolist() == [222.0, 222.0]


def test_compute_multiple_priors():
    from rtmdet.model import compute_multiple_priors, compute_priors

    image_shape = (224, 224)
    strides = [4, 8]
    separate_priors = [compute_priors(image_shape, stride) for stride in strides]
    priors = compute_multiple_priors(image_shape, strides)
    assert priors.shape == (56 * 56 + 28 * 28, 2)

    assert torch.allclose(priors[: 56 * 56], separate_priors[0])
    assert torch.allclose(priors[56 * 56 :], separate_priors[1])
