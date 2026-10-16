"""以靜態圖片測試 A/B/C 三區定位。"""
import numpy as np
import synthetic as S

from vision_pkg.vision_core.color_zone import (FieldParams, HsvRange, MaskParams, ZoneFindParams, find_field,
                                               find_zone, fuse_corners, hsv_mask, locate_field_zones,
                                               locate_zones, split_field)
from vision_pkg.vision_core.params_util import dataclass_from_flat


def _ranges(hsv):
    return {'a': hsv['blue'], 'b': hsv['green'], 'c': hsv['c_zone']}


def test_locate_three_zones_on_empty_desk(env_scenes, env_cfg, hsv):
    frames = [env_scenes['empty']] * 5
    res = locate_zones(frames, _ranges(hsv), env_cfg['zone_mp'], env_cfg['zone_find'],
                       env_cfg['min_det'], env_cfg['erode'], env_cfg['min_baseline'])
    for name in 'abc':
        r = res[name]
        assert r.ok, f'{name}: {r.message}'
        err = np.abs(r.corners - S.ENV_ZONES[name]).max()
        assert err < 3.0, f'{name} 角點誤差 {err:.2f}px'
        assert r.baseline > 0.8, f'{name} 基準可見比例 {r.baseline:.2f}'


def test_find_zone_returns_ordered_corners_and_mask(env_scenes, env_cfg, hsv):
    zr, mask = find_zone(env_scenes['empty'], hsv['blue'], env_cfg['zone_mp'], env_cfg['zone_find'])
    assert zr.found and zr.method in ('approx', 'minrect')
    assert mask.shape == env_scenes['empty'].shape[:2]
    tl, tr, br, bl = zr.corners
    assert tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]


def test_locate_still_works_when_objects_sit_on_station(env_scenes, env_cfg, hsv):
    # A 區有物件時（不在邊緣），外輪廓不變，仍可定位（只是基準比例會變低）
    res = locate_zones([env_scenes['a8']] * 5, {'a': hsv['blue']}, env_cfg['zone_mp'], env_cfg['zone_find'],
                       env_cfg['min_det'], env_cfg['erode'], 0.3)
    assert res['a'].ok
    assert np.abs(res['a'].corners - S.ENV_ZONES['a']).max() < 3.0


def test_locate_fails_with_reason_when_color_absent(env_cfg, hsv):
    blank = S.make_env_scene()
    blank[:] = S.WOOD  # 整張只剩桌面
    res = locate_zones([blank] * 5, _ranges(hsv), env_cfg['zone_mp'], env_cfg['zone_find'],
                       env_cfg['min_det'], env_cfg['erode'], env_cfg['min_baseline'])
    for name in 'abc':
        assert not res[name].ok
        assert res[name].corners is None
        assert '0/5' in res[name].message


def test_locate_flags_low_baseline(env_scenes, env_cfg, hsv):
    res = locate_zones([env_scenes['empty']] * 3, {'a': hsv['blue']}, env_cfg['zone_mp'], env_cfg['zone_find'],
                       1, env_cfg['erode'], 1.01)  # 可見比例最大就是 1，設 >1 必定觸發
    assert not res['a'].ok and '基準' in res['a'].message


# --------------------------------------------------------------------------- #
# field 模式：黑色外框場地均分成 A|B|C
# --------------------------------------------------------------------------- #
def _field_args(flat, hsv, **fp_over):
    fp = dataclass_from_flat(FieldParams, flat, 'field')
    fp = FieldParams(**{**fp.__dict__, **fp_over})
    ranges = {z: hsv[flat[f'zone_color.{z}']] for z in 'abc'}
    return (hsv['black'], ranges, dataclass_from_flat(MaskParams, flat, 'field_mask'),
            dataclass_from_flat(MaskParams, flat, 'zone_mask'), dataclass_from_flat(ZoneFindParams, flat, 'zone_find'),
            fp, flat['field.size_mm'], int(flat['boot_min_detections']), int(flat['roi_erode_px']),
            float(flat['min_baseline_ratio']))


def _expected_zones():
    return split_field(S.FIELD_CORNERS, S.FIELD_SIZE_MM, 'abc')


def test_field_mode_splits_outer_frame_into_three_zones(field_flat, hsv):
    res = locate_field_zones([S.make_field_scene()] * 5, *_field_args(field_flat, hsv))
    exp = _expected_zones()
    for z in 'abc':
        assert res[z].ok, f'{z}: {res[z].message}'
        err = np.abs(res[z].corners - exp[z]).max()
        assert err < 3.0, f'{z} 角點誤差 {err:.2f}px'
        assert res[z].baseline > 0.8, f'{z} 基準（分隔線/外框應被 ROI 內縮排除）{res[z].baseline:.2f}'
    assert res['a'].corners[:, 0].mean() < res['b'].corners[:, 0].mean() < res['c'].corners[:, 0].mean()


def test_field_mode_inner_edge_and_reverse(field_flat, hsv):
    res = locate_field_zones([S.make_field_scene()] * 5, *_field_args(field_flat, hsv, edge='inner'))
    assert all(res[z].ok for z in 'abc')
    b = S.FIELD_BORDER_MM * 566 / 600          # 外框線寬約幾像素：內緣應比外緣往內縮
    assert res['a'].corners[0][0] > S.FIELD_CORNERS[0][0] + 0.5 * b
    rev = locate_field_zones([S.make_field_scene()] * 5, *_field_args(field_flat, hsv, reverse=True))
    assert rev['a'].corners[:, 0].mean() > rev['c'].corners[:, 0].mean()   # C|B|A


def test_field_mode_fails_with_reason(field_flat, hsv):
    blank = S.make_field_scene()
    blank[:] = S.WOOD
    res = locate_field_zones([blank] * 5, *_field_args(field_flat, hsv))
    assert all(not res[z].ok and '外框' in res[z].message for z in 'abc')
    tall = np.ascontiguousarray(np.rot90(S.make_field_scene()))   # 相機轉 90°：外框變直的
    res = locate_field_zones([tall] * 5, *_field_args(field_flat, hsv))
    assert not res['a'].ok and '橫向' in res['a'].message


def test_find_field_outer_corners(field_flat, hsv):
    zr, _ = find_field(S.make_field_scene(), hsv['black'], dataclass_from_flat(MaskParams, field_flat, 'field_mask'),
                       dataclass_from_flat(ZoneFindParams, field_flat, 'zone_find'), 'outer')
    assert zr.found and np.abs(zr.corners - S.FIELD_CORNERS).max() < 3.0


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
