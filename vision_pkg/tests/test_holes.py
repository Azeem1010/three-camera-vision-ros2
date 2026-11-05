"""C 區方孔偵測、固定編號、占用比例與放置順序檢查。"""
import numpy as np
import pytest
import synthetic as S

from vision_pkg.vision_core.color_zone import hsv_mask
from vision_pkg.vision_core.holes import check_fill_order, locate_holes, order_slots, slot_ratios


def _expected_centers_img():
    return S.cv2.perspectiveTransform(np.float32(S.SLOT_CENTERS_MM).reshape(-1, 1, 2),
                                      S.arm2_field_to_img()).reshape(-1, 2)


def _layout(img, arm2_cfg, hsv, k=0):
    return locate_holes(hsv_mask(img, hsv['pink'], arm2_cfg['field_mp']), arm2_cfg['hole'], k)


def test_holes_are_found_and_numbered_bottom_up_left_to_right(arm2_cfg, hsv):
    lay = _layout(S.make_arm2_scene([1, 2, 3]), arm2_cfg, hsv)
    assert lay.ok, lay.message
    got = np.array([[h.cx, h.cy] for h in lay.holes])
    assert np.abs(got - _expected_centers_img()).max() < 3.0
    # 使用者視角：1 號在左下；2 號在 1 號右邊；3 號在 1 號上面
    assert got[1][0] > got[0][0] and abs(got[1][1] - got[0][1]) < 10
    assert got[2][1] < got[0][1] and abs(got[2][0] - got[0][0]) < 10


@pytest.mark.parametrize('k', [1, 2, 3])
def test_rotated_camera_with_matching_k_gives_same_numbering(k, arm2_cfg, hsv):
    filled = [1, 4, 6]
    img = S.make_arm2_scene(filled, rotation_k=k)
    lay = _layout(img, arm2_cfg, hsv, k)
    assert lay.ok, lay.message
    obj = hsv_mask(img, hsv['blue'], arm2_cfg['obj_mp'])
    ratios = slot_ratios(obj, [h.corners for h in lay.holes], arm2_cfg['hole'].margin_ratio)
    assert [r >= arm2_cfg['hole'].occupied_threshold for r in ratios] == [s in filled for s in range(1, 9)]


def test_wrong_rotation_is_reported_not_guessed(arm2_cfg, hsv):
    lay = _layout(S.make_arm2_scene(), arm2_cfg, hsv, 1)    # 影像沒轉，卻設 k=1：長軸變橫的
    assert not lay.ok and 'view_rotation_k' in lay.message


def test_covered_hole_fails_with_count(arm2_cfg, hsv):
    lay = _layout(S.make_arm2_scene(cover_slot=5), arm2_cfg, hsv)
    assert not lay.ok and '預期 8' in lay.message


def test_occupancy_ratios(arm2_cfg, hsv):
    img = S.make_arm2_scene([2, 5, 8])
    lay = _layout(img, arm2_cfg, hsv)
    r = slot_ratios(hsv_mask(img, hsv['blue'], arm2_cfg['obj_mp']), [h.corners for h in lay.holes], 0.1)
    for i in range(8):
        if i + 1 in (2, 5, 8):
            assert r[i] > 0.4
        else:
            assert r[i] < 0.02


def test_order_slots_rejects_non_lattice():
    pts = [(0, 0), (10, 0), (0, 50), (10, 50), (0, 100), (10, 100), (0, 150), (10, 150)]
    assert order_slots(pts, 0)[0] is not None
    bad = [(0, 0), (10, 30), (0, 50), (10, 50), (0, 100), (10, 100), (0, 150), (10, 150)]
    order, why = order_slots(bad, 0)
    assert order is None and '格子' in why
    assert order_slots(pts[:7], 0)[0] is None


def test_check_fill_order():
    c = check_fill_order([True, True, False, False, False, False, False, False])
    assert c.order_ok and c.wrong_slots == [] and c.next_slot == 3
    c = check_fill_order([True, True, False, True, False, False, True, False])
    assert not c.order_ok and c.wrong_slots == [4, 7] and c.next_slot == 3 and '跳號' in c.message
    c = check_fill_order([True] * 8)
    assert c.order_ok and c.next_slot == 0
    c = check_fill_order([False] * 8)
    assert c.order_ok and c.next_slot == 1
