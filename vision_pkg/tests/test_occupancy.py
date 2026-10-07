"""各區是否有物件 / B 區手臂 的判斷（站台色被遮擋為主、橘色為輔、黑色視為手臂）。"""
import synthetic as S

from vision_pkg.vision_core.color_zone import (count_blobs, judge_arm_present, judge_object_present,
                                               locate_zones, make_roi, measure_zones)


def _setup(env_scenes, env_cfg, hsv):
    ranges = {'a': hsv['blue'], 'b': hsv['green'], 'c': hsv['c_zone']}
    loc = locate_zones([env_scenes['empty']] * 5, ranges, env_cfg['zone_mp'], env_cfg['zone_find'],
                       env_cfg['min_det'], env_cfg['erode'], env_cfg['min_baseline'])
    shape = env_scenes['empty'].shape
    rois = {n: make_roi(shape, loc[n].corners, env_cfg['erode']) for n in 'abc'}
    base = {n: loc[n].baseline for n in 'abc'}
    return ranges, rois, base


def _states(img, env_cfg, hsv, ranges, rois, base):
    metrics, orange_mask = measure_zones(img, rois, ranges, hsv['orange'], hsv['black'],
                                         env_cfg['zone_mp'], env_cfg['orange_mp'], env_cfg['black_mp'])
    out = {}
    for n in 'abc':
        has_obj, _ = judge_object_present(metrics[n], base[n], env_cfg['occupancy'])
        arm, _ = judge_arm_present(metrics[n], base[n], env_cfg['occupancy'])
        out[n] = (has_obj, arm)
    return out, orange_mask, metrics


def test_empty_desk_reports_nothing(env_scenes, env_cfg, hsv):
    ranges, rois, base = _setup(env_scenes, env_cfg, hsv)
    st, _, _ = _states(env_scenes['empty'], env_cfg, hsv, ranges, rois, base)
    assert st == {'a': (False, False), 'b': (False, False), 'c': (False, False)}


def test_objects_in_a_are_detected_even_the_smallest(env_scenes, env_cfg, hsv):
    ranges, rois, base = _setup(env_scenes, env_cfg, hsv)
    st, _, _ = _states(env_scenes['a8'], env_cfg, hsv, ranges, rois, base)
    assert st['a'][0] and not st['b'][0] and not st['c'][0]
    one_small = S.make_env_scene(a_objs=[(100, 100, 20, 0)])
    st, _, m = _states(one_small, env_cfg, hsv, ranges, rois, base)
    assert st['a'][0], f'單一 20mm 物件未被偵測 {m["a"]}'


def test_arm_over_b_is_arm_not_object(env_scenes, env_cfg, hsv):
    ranges, rois, base = _setup(env_scenes, env_cfg, hsv)
    st, _, m = _states(env_scenes['b_arm'], env_cfg, hsv, ranges, rois, base)
    assert st['b'] == (False, True), f'B 區應為「有手臂、無物件」: {m["b"]}'


def test_object_in_b_is_object_not_arm(env_cfg, env_scenes, hsv):
    ranges, rois, base = _setup(env_scenes, env_cfg, hsv)
    img = S.make_env_scene(b_objs=[(100, 100, 40, 10)])
    st, _, _ = _states(img, env_cfg, hsv, ranges, rois, base)
    assert st['b'] == (True, False)


def test_c_zone_blob_count(env_scenes, env_cfg, hsv):
    ranges, rois, base = _setup(env_scenes, env_cfg, hsv)
    img = S.make_env_scene(c_objs=[(40, 40, 30, 0), (150, 60, 30, 20), (100, 150, 30, 0)])
    st, orange_mask, _ = _states(img, env_cfg, hsv, ranges, rois, base)
    import cv2
    in_c = cv2.bitwise_and(orange_mask, rois['c'].mask)
    mm2_per_px = (S.ZONE_SIZE_MM[0] * S.ZONE_SIZE_MM[1]) / rois['c'].area
    assert count_blobs(in_c, env_cfg['count_min_mm2'] / mm2_per_px) == 3
    assert st['c'][0]
