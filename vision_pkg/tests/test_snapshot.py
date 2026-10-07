"""A 區快照（偵測 + 相對排序）與座標配對。"""
import numpy as np
import synthetic as S

from vision_pkg.vision_core.size_ranker import AbsoluteTableRanker, RelativeRanker
from vision_pkg.vision_core.snapshot import ObjectInfo, capture_snapshot, match_object


def _capture(img, env_cfg, hsv, ranker=None, expected=8):
    ranker = ranker or RelativeRanker(env_cfg['margin_px'] / env_cfg['ppm'])
    return capture_snapshot(img, S.ENV_ZONES['a'], S.ZONE_SIZE_MM, env_cfg['ppm'], hsv['orange'],
                            env_cfg['orange_mp'], env_cfg['square'], ranker, expected)


def test_capture_ranks_by_size_and_keeps_plane_positions(env_scenes, env_cfg, hsv):
    res = _capture(env_scenes['a8'], env_cfg, hsv)
    assert res.success, res.message
    assert res.count == 8 and not res.low_confidence
    assert [o.rank for o in res.objects] == list(range(1, 9))
    for i, truth_size in enumerate(S.A_SIZES):
        tx, ty = S.A_POSITIONS[S.A_PERM[i]]
        o = res.objects[i]  # objects 已依 rank 排序，rank i+1 即第 i 小的物件
        assert np.hypot(o.x_mm - tx, o.y_mm - ty) < 3.0, f'rank {o.rank} 位置誤差過大'
        assert abs(o.side_mm - truth_size) < 2.5


def test_capture_fails_without_guessing_when_count_is_not_eight(env_scenes, env_cfg, hsv):
    res = _capture(env_scenes['a7'], env_cfg, hsv)
    assert not res.success and res.count == 7
    assert '7' in res.message and '8' in res.message
    assert res.objects == []


def test_capture_explains_rejected_blobs(env_cfg, hsv):
    img = S.make_env_scene(a_objs=S.a_objects(skip=[3]) + [(100, 100, 40, 0), (141, 100, 40, 0)])  # 兩個黏住
    res = _capture(img, env_cfg, hsv)
    assert not res.success
    assert '被排除' in res.message


def test_low_confidence_when_two_sizes_are_too_close(env_cfg, hsv):
    sizes = [20, 24, 28, 32, 36, 40, 41, 48]  # 40 與 41 只差 1mm(=2px) < 門檻 3px
    img = S.make_env_scene(a_objs=S.a_objects(sizes))
    res = _capture(img, env_cfg, hsv)
    assert res.success and res.low_confidence
    assert '低信心' in res.message


def test_absolute_table_ranker_is_a_drop_in_replacement(env_scenes, env_cfg, hsv):
    rel = _capture(env_scenes['a8'], env_cfg, hsv)
    absr = _capture(env_scenes['a8'], env_cfg, hsv, ranker=AbsoluteTableRanker(S.A_SIZES, 2.5))
    assert absr.success and not absr.low_confidence
    assert [(o.rank, round(o.x_mm), round(o.y_mm)) for o in rel.objects] == \
           [(o.rank, round(o.x_mm), round(o.y_mm)) for o in absr.objects]


def _objs():
    return [ObjectInfo(rank=i + 1, x_mm=30.0 * (i + 1), y_mm=50.0, side_mm=20.0 + 4 * i, px=0.0, py=0.0)
            for i in range(4)]


def test_match_picks_nearest_within_threshold():
    m = match_object(_objs(), [], 62.0, 55.0, 20.0)
    assert m.success and m.rank == 2 and m.index == 1
    assert abs(m.distance_mm - np.hypot(2, 5)) < 1e-6


def test_match_fails_beyond_threshold():
    m = match_object(_objs(), [], 62.0, 90.0, 20.0)
    assert not m.success and '門檻' in m.message and m.rank == 0


def test_match_skips_taken_object_and_reports_it():
    objs = _objs()
    m = match_object(objs, [1], 60.0, 50.0, 20.0)  # rank 2 已被取走，位置就在它上面
    assert not m.success and '已被標記為取走' in m.message
    m2 = match_object(objs, [1], 70.0, 50.0, 20.0)  # 離 rank 2 較近但已取走；rank 3(x=90) 在 20mm 內
    assert m2.success and m2.rank == 3


def test_match_without_snapshot():
    m = match_object([], [], 0.0, 0.0, 20.0)
    assert not m.success and 'capture_ranking' in m.message
