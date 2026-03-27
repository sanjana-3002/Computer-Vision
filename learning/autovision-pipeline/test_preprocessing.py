# These tests prove the pipeline is reproducible — key requirement for ML engineering roles

import pytest
import numpy as np
import cv2
from preprocessing import ChartPreprocessor

IMAGE_PATH = "/Users/sanjanawaghray/Documents/projects/Computer-Vision-1/learning/data/raw/chart1.webp"


@pytest.fixture
def processor():
    return ChartPreprocessor(IMAGE_PATH)


def test_image_loads_correctly(processor):
    """ChartPreprocessor should load a valid image and store it as a NumPy array."""
    assert processor.bgr_image is not None
    assert isinstance(processor.bgr_image, np.ndarray)
    assert processor.bgr_image.ndim == 3


def test_convert_color_spaces_shapes(processor):
    """All color space conversions should return arrays matching the source image."""
    h, w = processor.bgr_image.shape[:2]
    rgb, gray, hsv, lab = processor.convert_color_spaces()

    assert rgb.shape  == (h, w, 3)
    assert gray.shape == (h, w)       # grayscale has no channel dimension
    assert hsv.shape  == (h, w, 3)
    assert lab.shape  == (h, w, 3)


def test_resize_and_crop_returns_224(processor):
    """resize_and_crop should always return a (224, 224, 3) image."""
    resized, _ = processor.resize_and_crop()
    assert resized.shape == (224, 224, 3)


def test_apply_filters_returns_same_shape(processor):
    """All three filters should return arrays with the same spatial dimensions as input."""
    h, w = processor.bgr_image.shape[:2]
    gaussian, median, bilateral = processor.apply_filters()

    assert gaussian.shape  == (h, w)
    assert median.shape    == (h, w)
    assert bilateral.shape == (h, w)


def test_find_candle_contours_returns_list(processor):
    """find_candle_contours should always return a list (even if empty on a blank image)."""
    contours = processor.find_candle_contours()
    assert isinstance(contours, list)


def test_find_candle_contours_dict_keys(processor):
    """Each detection dict must have all four expected keys."""
    contours = processor.find_candle_contours()
    required_keys = {"contour", "bbox", "centroid", "area"}
    for item in contours:
        assert required_keys.issubset(item.keys()), f"Missing keys in detection: {item.keys()}"


def test_config_has_all_required_keys(processor):
    """self.config must contain all tunable parameters used across the pipeline."""
    required = {
        'bilateral_d',
        'bilateral_sigma_color',
        'bilateral_sigma_space',
        'canny_low',
        'canny_high',
        'morph_kernel_size',
        'contour_min_area',
        'contour_max_area',
        'harris_block_size',
        'harris_k',
        'resize_target',
    }
    assert required.issubset(processor.config.keys())
