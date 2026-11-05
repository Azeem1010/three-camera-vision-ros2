"""各區是否有瓶蓋 / B 區手臂 的判斷（藍色為主、粉紅被遮擋為輔、黑色視為手臂）。"""
import cv2
import synthetic as S

from vision_pkg.vision_core.color_zone import (count_blobs, judge_arm_present, judge_object_present,
                                               locate_field_zones, make_roi, measure_zones)


def _setup(env_scenes, env_cfg, hsv):
    loc = locate_field_zones([env_scenes['empty']] * 5, hsv['pink'], ('c', 'b', 'a'), env_cfg['field_mp'],
                             env_cfg['zone_mp'], env_cfg['zone_find'], env_cfg['field'], env_cfg['field_size'],
                             env_cfg['min_det'], env_cfg['erode'], env_cfg['min_baseline'])
    shape = env_scenes['empty'].shape
    rois = {n: make_roi(shape, loc[n].corners, env_cfg['erode']) for n in 'abc'}
    base = {n: loc[n].baseline for n in 'abc'}
    return rois, base


def _states(img, env_cfg, hsv, rois, base):
    metrics, obj_mask = measure_zones(img, rois, hsv['pink'], hsv['blue'], hsv['black'],
                                      env_cfg['zone_mp'], env_cfg['obj_mp'], env_cfg['black_mp'])
    out = {}
    for n in 'abc':
        has_obj, _ = judge_object_present(metrics[n], base[n], env_cfg['occupancy'])
        arm, _ = judge_arm_present(metrics[n], base[n], env_cfg['occupancy'])
        out[n] = (has_obj, arm)
    return out, obj_mask, metrics


def _count(obj_mask, roi, env_cfg):
    mm2_per_px = (200.0 * 300.0) / cv2.contourArea(roi.corners)
    return count_blobs(cv2.bitwise_and(obj_mask, roi.mask), env_cfg['count_min_mm2'] / mm2_per_px)


def test_empty_field_reports_nothing(env_scenes, env_cfg, hsv):
    rois, base = _setup(env_scenes, env_cfg, hsv)
    st, _, _ = _states(env_scenes['empty'], env_cfg, hsv, rois, base)
    assert st == {'a': (False, False), 'b': (False, False), 'c': (False, False)}


def test_a_with_caps_and_a_single_cap(env_scenes, env_cfg, hsv):
    rois, base = _setup(env_scenes, env_cfg, hsv)
    st, obj_mask, _ = _states(env_scenes['a8'], env_cfg, hsv, rois, base)
    assert st['a'][0] and not st['b'][0] and not st['c'][0]
    assert _count(obj_mask, rois['a'], env_cfg) == 8
    one = S.make_env_scene(a_caps=[(500, 150)])
    st, _, m = _states(one, env_cfg, hsv, rois, base)
    assert st['a'][0], f'單一瓶蓋未被偵測 {m["a"]}'


def test_arm_over_b_is_arm_not_object(env_scenes, env_cfg, hsv):
    rois, base = _setup(env_scenes, env_cfg, hsv)
    st, _, m = _states(env_scenes['b_arm'], env_cfg, hsv, rois, base)
    assert st['b'] == (False, True), f'B 區應為「有手臂、無物件」: {m["b"]}'


def test_cap_in_b_is_object_not_arm(env_scenes, env_cfg, hsv):
    rois, base = _setup(env_scenes, env_cfg, hsv)
    st, _, _ = _states(S.make_env_scene(b_caps=[(300, 150)]), env_cfg, hsv, rois, base)
    assert st['b'] == (True, False)


def test_c_zone_counts_caps_in_holes(env_scenes, env_cfg, hsv):
    rois, base = _setup(env_scenes, env_cfg, hsv)
    st, obj_mask, _ = _states(env_scenes['c3'], env_cfg, hsv, rois, base)
    assert _count(obj_mask, rois['c'], env_cfg) == 3 and st['c'][0]
    assert _count(obj_mask, rois['a'], env_cfg) == 5
