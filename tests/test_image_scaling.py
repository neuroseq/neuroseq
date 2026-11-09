import numpy as np
import pytest

from neuroseq.image_scaling import display_limits
from neuroseq.image_scaling import scale_to_uint8


def test_a_hot_pixel_does_not_darken_the_image() -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(300, 30, (200, 200))
    image[5, 5] = 30000  # a hot pixel, as two-photon means often have
    scaled = scale_to_uint8(image)
    assert scaled.dtype == np.uint8
    assert scaled[5, 5] == 255  # saturates
    # The tissue spans the grey levels instead of a handful near black.
    assert len(np.unique(scaled)) > 200
    assert 100 < np.median(scaled) < 160


def test_limits_fall_back_to_min_and_max_and_reject_empty_images() -> None:
    two_values = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert display_limits(two_values) == (0.0, 10.0)
    np.testing.assert_array_equal(scale_to_uint8(two_values), [[0, 0], [0, 255]])
    np.testing.assert_array_equal(scale_to_uint8(np.full((2, 2), 7.0)), 0)
    with pytest.raises(ValueError):
        display_limits(np.array([np.nan]))
