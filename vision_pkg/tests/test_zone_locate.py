"""以靜態圖片測試場地定位（粉紅場地 -> C|B|A 三區）。"""
import numpy as np
import synthetic as S

from vision_pkg.vision_core.color_zone import (FieldParams, HsvRange, MaskParams, find_field, fuse_corners,
                                               hsv_mask, locate_field_zones, max_area_quad)

LTR = ('c', 'b', 'a')


def _locate(frames, env_cfg, hsv, **fp_over):
    fp = FieldParams(**{**env_cfg['field'].__dict__, **fp_over})
    return locate_field_zones(frames, hsv['pink'], LTR, env_cfg['field_mp'], env_cfg['zone_mp'],
                              env_cfg['zone_find'], fp, env_cfg['field_size'], env_cfg['min_det'],
                              env_cfg['erode'], env_cfg['min_baseline'])


def test_field_is_split_into_c_b_a_from_left_to_right(env_scenes, env_cfg, hsv):
    res = _locate([env_scenes['empty']] * 5, env_cfg, hsv)
    for z in 'abc':
        assert res[z].ok, f'{z}: {res[z].message}'
        err = np.abs(res[z].corners - S.zone_corners_img(z)).max()
        assert err < 3.0, f'{z} 角點誤差 {err:.2f}px'
    assert res['c'].corners[:, 0].mean() < res['b'].corners[:, 0].mean() < res['a'].corners[:, 0].mean()
    assert res['a'].baseline > 0.9 and res['b'].baseline > 0.9
    assert 0.6 < res['c'].baseline < res['a'].baseline      # C 區的方孔不是粉紅色，基準比較低


def test_field_corners_ignore_the_arm_mount_tab(env_scenes, env_cfg, hsv):
    """場地下緣凸出的手臂底座泡棉板不能把角點拉歪。"""
    zr, _ = find_field(env_scenes['empty'], hsv['pink'], env_cfg['field_mp'], env_cfg['zone_find'])
    assert zr.found, zr.reason
    assert np.abs(zr.corners - S.FIELD_CORNERS).max() < 3.0


def test_field_still_found_with_caps_on_it(env_scenes, env_cfg, hsv):
    res = _locate([env_scenes['c3']] * 5, env_cfg, hsv)
    assert all(res[z].ok for z in 'abc')
    assert np.abs(res['a'].corners - S.zone_corners_img('a')).max() < 3.0


def test_reverse_puts_a_on_the_left(env_scenes, env_cfg, hsv):
    res = _locate([env_scenes['empty']] * 5, env_cfg, hsv, reverse=True)
    assert res['a'].corners[:, 0].mean() < res['c'].corners[:, 0].mean()


def test_field_fails_with_reason(env_scenes, env_cfg, hsv):
    blank = env_scenes['empty'].copy()
    blank[:] = S.DESK
    res = _locate([blank] * 5, env_cfg, hsv)
    assert all(not res[z].ok and '場地' in res[z].message and '0/5' in res[z].message for z in 'abc')
    tall = np.ascontiguousarray(np.rot90(env_scenes['empty']))   # 相機轉 90°：場地變直的
    res = _locate([tall] * 5, env_cfg, hsv)
    assert not res['a'].ok and '橫向' in res['a'].message


def test_low_baseline_is_flagged(env_scenes, env_cfg, hsv):
    fp = env_cfg['field']
    res = locate_field_zones([env_scenes['empty']] * 3, hsv['pink'], LTR, env_cfg['field_mp'], env_cfg['zone_mp'],
                             env_cfg['zone_find'], fp, env_cfg['field_size'], 1, env_cfg['erode'], 1.01)
    assert not res['a'].ok and '基準' in res['a'].message


def test_max_area_quad_picks_true_corners_of_rectangle_with_bump():
    rect = np.array([[0, 0], [600, 0], [600, 300], [230, 300], [230, 340], [320, 340], [320, 300], [0, 300]],
                    np.int32).reshape(-1, 1, 2)
    q = max_area_quad(rect)
    assert np.abs(q - np.float32([[0, 0], [600, 0], [600, 300], [0, 300]])).max() < 1e-3


def test_fuse_corners_uses_median_and_min_count():
    base = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    lst = [base, base + 0.2, base + 50, base - 0.2]  # 一個離群
    fused = fuse_corners(lst, 3)
    assert np.abs(fused - base).max() < 0.3
    assert fuse_corners(lst[:2], 3) is None


def test_hsv_mask_supports_hue_wraparound():
    img = np.zeros((20, 40, 3), np.uint8)
    img[:, :20] = (0, 0, 255)   # 紅 H≈0
    img[:, 20:] = (70, 60, 230)  # 偏紫紅 H≈178
    wrap = HsvRange((170, 100, 100), (10, 255, 255))  # 下限 > 上限：環繞
    m = hsv_mask(img, wrap, MaskParams(0, 0, 0))
    assert m[:, :20].all() and m[:, 20:].all()
    plain = hsv_mask(img, HsvRange((0, 100, 100), (10, 255, 255)), MaskParams(0, 0, 0))
    assert plain[:, :20].all() and not plain[:, 20:].any()


def test_default_pink_does_not_match_desk_or_hole_paper(env_scenes, hsv):
    m = hsv_mask(env_scenes['empty'], hsv['pink'], MaskParams(0, 0, 0))
    assert m[5:60, 5:200].mean() < 1                       # 左上角是桌面
    x, y = S.SLOT_CENTERS_MM[0]
    c = np.round(S.cv2.perspectiveTransform(np.float32([[[x, y]]]), S.env_field_to_img())[0, 0]).astype(int)
    assert m[c[1] - 3:c[1] + 3, c[0] - 3:c[0] + 3].mean() < 1   # 方孔底的白紙
