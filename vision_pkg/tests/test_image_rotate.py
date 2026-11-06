"""image_rotate_node.rotate_pixels：順時針旋轉、step 有 padding、不支援的編碼。"""
import numpy as np
import pytest

from vision_pkg.image_rotate_node import rotate_pixels


def _img(h=3, w=4, ch=3):
    return (np.arange(h * w * ch, dtype=np.uint8).reshape(h, w, ch))


@pytest.mark.parametrize('deg,k', [(0, 0), (90, -1), (180, 2), (270, 1)])
def test_rotation_matches_numpy(deg, k):
    img = _img()
    data, h, w, step = rotate_pixels(img.tobytes(), 3, 4, 12, 'rgb8', deg)
    out = np.frombuffer(data, np.uint8).reshape(h, step)[:, :w * 3].reshape(h, w, 3)
    assert np.array_equal(out, np.rot90(img, k))


def test_180_maps_top_left_to_bottom_right():
    img = _img()
    data, h, w, _ = rotate_pixels(img.tobytes(), 3, 4, 12, 'rgb8', 180)
    out = np.frombuffer(data, np.uint8).reshape(h, w, 3)
    assert np.array_equal(out[-1, -1], img[0, 0])


def test_row_padding_is_dropped():
    img = _img(2, 2, 1)
    padded = np.zeros((2, 4), np.uint8)
    padded[:, :2] = img[:, :, 0]
    data, h, w, step = rotate_pixels(padded.tobytes(), 2, 2, 4, 'mono8', 180)
    assert (h, w, step) == (2, 2, 2)
    assert np.array_equal(np.frombuffer(data, np.uint8).reshape(2, 2), np.rot90(img[:, :, 0], 2))


def test_unsupported_encoding_and_angle():
    with pytest.raises(ValueError):
        rotate_pixels(b'\0' * 16, 2, 4, 8, 'yuv422_yuy2', 180)
    with pytest.raises(ValueError):
        rotate_pixels(_img().tobytes(), 3, 4, 12, 'rgb8', 45)
