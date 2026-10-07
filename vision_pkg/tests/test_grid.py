"""C 區透視校正、2x4 格位切割、占用判斷與排序檢查（含 warp_rotation_k）。"""
import numpy as np
import pytest
import synthetic as S

from vision_pkg.vision_core.color_zone import (MaskParams, ZoneFindParams, find_zone, hsv_mask,
                                               warp_to_plane)
from vision_pkg.vision_core.geometry import rotate_corners
from vision_pkg.vision_core.grid import (GridParams, analyze_grid, cell_centers, cell_rects, occupancy)
from vision_pkg.vision_core.params_util import dataclass_from_flat
from vision_pkg.vision_core.size_ranker import RelativeRanker
from vision_pkg.vision_core.square_detect import SquareParams

ASC = [28, 30, 32, 34, 36, 38, 40, 42]


@pytest.fixture(scope='module')
def cfg(arm2_flat):
    f = arm2_flat
    return {
        'zone_mp': dataclass_from_flat(MaskParams, f, 'zone_mask'),
        'zone_find': dataclass_from_flat(ZoneFindParams, f, 'zone_find'),
        'orange_mp': dataclass_from_flat(MaskParams, f, 'orange_mask'),
        'square': dataclass_from_flat(SquareParams, f, 'square'),
        'grid': dataclass_from_flat(GridParams, f, 'grid'),
        'ppm': float(f['warp_px_per_mm']),
        'margin_px': float(f['ranker.confidence_margin_px']),
    }


def _analyze(img, cfg, hsv, k=0):
    objects = hsv_mask(img, hsv['orange'], cfg['orange_mp'])  # 物件併入站台遮罩，避免貼邊物件咬掉角落
    zr, _ = find_zone(img, hsv['c_zone'], cfg['zone_mp'], cfg['zone_find'], extra_mask=objects)
    assert zr.found, zr.reason
    warped, _ = warp_to_plane(img, rotate_corners(zr.corners, k), *S.C_SIZE_MM, cfg['ppm'])
    return analyze_grid(warped, hsv['orange'], cfg['orange_mp'], cfg['square'], cfg['grid'])


def test_cell_rects_are_row_major_with_top_row_first():
    rects = cell_rects(400, 200, 2, 4)
    assert len(rects) == 8
    assert rects[0] == (0, 0, 100, 100) and rects[3] == (300, 0, 400, 100)
    assert rects[4] == (0, 100, 100, 200) and rects[7] == (300, 100, 400, 200)
    assert cell_centers(rects)[0] == (50.0, 50.0)


def test_occupancy_threshold():
    assert occupancy([0.0, 0.29, 0.3, 0.9], 0.3) == [False, False, True, True]


def test_full_ascending_grid(data_dir, cfg, hsv):
    ga = _analyze(S.imread(data_dir / 'arm2_c_ascending.png'), cfg, hsv)
    assert ga.occupied == [True] * 8
    assert not ga.unmeasurable_slots
    sides_mm = [s / cfg['ppm'] for s in ga.sides_px]
    for got, truth in zip(sides_mm, ASC):
        assert abs(got - truth) < 2.5, f'量到 {got:.1f}mm，真值 {truth}mm'
    chk = RelativeRanker(cfg['margin_px'] / cfg['ppm']).check_order(sides_mm)
    assert chk.order_ok and chk.wrong_slots == []


def test_empty_slots_are_not_occupied(cfg, hsv):
    sides = [28, None, 32, 34, None, 38, 40, 42]
    ga = _analyze(S.make_arm2_scene(sides), cfg, hsv)
    assert ga.occupied == [s is not None for s in sides]
    assert ga.sides_px[1] is None and ga.sides_px[4] is None
    assert all(r < 0.05 for i, r in enumerate(ga.ratios) if sides[i] is None)


def test_swapped_objects_are_reported_as_wrong_slots(data_dir, cfg, hsv):
    ga = _analyze(S.imread(data_dir / 'arm2_c_wrong_order.png'), cfg, hsv)
    sides_mm = [s / cfg['ppm'] if s is not None else None for s in ga.sides_px]
    chk = RelativeRanker(cfg['margin_px'] / cfg['ppm']).check_order(sides_mm)
    assert not chk.order_ok and chk.wrong_slots == [3, 6]


def test_partial_fill_in_order_is_ok(cfg, hsv):
    sides = [28, 30, None, None, None, 38, None, 42]
    ga = _analyze(S.make_arm2_scene(sides), cfg, hsv)
    sides_mm = [s / cfg['ppm'] if s is not None else None for s in ga.sides_px]
    chk = RelativeRanker(cfg['margin_px'] / cfg['ppm']).check_order(sides_mm)
    assert chk.order_ok


def test_rotation_k_aligns_slot_numbering(cfg, hsv):
    """相機相對站台轉 180° 時，用 warp_rotation_k=2 應得到與 k=0 場景相同的格位結果。"""
    sides = [28, None, 32, 34, 36, None, 40, 42]
    base = _analyze(S.make_arm2_scene(sides, rotation_k=0), cfg, hsv, k=0)
    rotated = _analyze(S.make_arm2_scene(sides, rotation_k=2), cfg, hsv, k=2)
    wrong = _analyze(S.make_arm2_scene(sides, rotation_k=2), cfg, hsv, k=0)  # 沒轉正：格位會對不上
    assert rotated.occupied == base.occupied == [s is not None for s in sides]
    assert wrong.occupied != base.occupied
    for a, b in zip(base.sides_px, rotated.sides_px):
        assert (a is None) == (b is None)
        if a is not None:
            assert abs(a - b) < 3.0


def test_touching_objects_make_cells_unmeasurable_instead_of_guessing(cfg, hsv):
    sides = [42, 42, None, None, None, None, None, None]
    img = S.make_arm2_scene(sides, extra_objs=[(50, 25, 20, 0)])  # 在兩格之間再補一塊，讓兩個物件黏成一塊
    ga = _analyze(img, cfg, hsv)
    assert ga.occupied[0] and ga.occupied[1]
    assert set(ga.unmeasurable_slots) >= {1, 2}
    assert ga.sides_px[0] is None and ga.sides_px[1] is None
