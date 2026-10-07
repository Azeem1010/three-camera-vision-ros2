"""橘色正方形偵測與邊長量測。"""
import cv2
import numpy as np
import synthetic as S

from vision_pkg.vision_core.color_zone import hsv_mask, warp_to_plane
from vision_pkg.vision_core.square_detect import analyze_blobs, detect_squares


def _warped_mask(img, env_cfg, hsv, zone='a'):
    warped, _ = warp_to_plane(img, S.ENV_ZONES[zone], *S.ZONE_SIZE_MM, env_cfg['ppm'])
    return hsv_mask(warped, hsv['orange'], env_cfg['orange_mp'])


def test_detects_all_eight_squares_and_measures_side(env_scenes, env_cfg, hsv):
    mask = _warped_mask(env_scenes['a8'], env_cfg, hsv)
    squares, rejected = detect_squares(mask, env_cfg['square'])
    assert len(squares) == 8 and not rejected
    got = sorted(s.side / env_cfg['ppm'] for s in squares)
    for g, truth in zip(got, S.A_SIZES):
        # 校正後 2px/mm、原圖約 1px/mm：允許 ~2mm 量測誤差；並且不能偏大過多
        assert abs(g - truth) < 2.5, f'量到 {g:.1f}mm，真值 {truth}mm'


def test_wood_desk_is_not_mistaken_for_orange(env_scenes, env_cfg, hsv):
    full = hsv_mask(env_scenes['empty'], hsv['orange'], env_cfg['orange_mp'])
    assert cv2.countNonZero(full) == 0


def test_rotated_square_still_passes(env_cfg, hsv):
    img = S.make_env_scene(a_objs=[(100, 100, 40, 33)])
    squares, rejected = detect_squares(_warped_mask(img, env_cfg, hsv), env_cfg['square'])
    assert len(squares) == 1 and not rejected
    assert abs(squares[0].side / env_cfg['ppm'] - 40) < 2.5
    assert squares[0].aspect <= env_cfg['square'].max_aspect


def test_rectangle_is_rejected_for_aspect(env_cfg, hsv):
    H = S.plane_to_img_H(S.ENV_ZONES['a'], S.ZONE_SIZE_MM)

    def draw(img):
        pts = np.float32([[60, 80], [140, 80], [140, 110], [60, 110]]).reshape(-1, 1, 2)  # 80x30 長方形
        cv2.fillConvexPoly(img, np.round(cv2.perspectiveTransform(pts, H).reshape(-1, 2)).astype(np.int32), S.ORANGE)

    img = S.make_env_scene(extra_draw=draw)
    squares, rejected = detect_squares(_warped_mask(img, env_cfg, hsv), env_cfg['square'])
    assert not squares and len(rejected) == 1
    assert '長寬比' in rejected[0].reject_reason


def test_circle_is_rejected(env_cfg, hsv):
    H = S.plane_to_img_H(S.ENV_ZONES['a'], S.ZONE_SIZE_MM)

    def draw(img):
        c = cv2.perspectiveTransform(np.float32([[[100, 100]]]), H)[0, 0]
        cv2.circle(img, (int(c[0]), int(c[1])), 20, S.ORANGE, -1, cv2.LINE_AA)

    squares, rejected = detect_squares(_warped_mask(S.make_env_scene(extra_draw=draw), env_cfg, hsv),
                                       env_cfg['square'])
    assert not squares and len(rejected) == 1


def test_touching_squares_are_merged_and_rejected(env_cfg, hsv):
    img = S.make_env_scene(a_objs=[(80, 100, 40, 0), (121, 100, 40, 0)])  # 邊對邊黏在一起
    squares, rejected = detect_squares(_warped_mask(img, env_cfg, hsv), env_cfg['square'])
    assert not squares and len(rejected) == 1
    assert rejected[0].aspect > env_cfg['square'].max_aspect or rejected[0].too_large


def test_border_touching_object_is_rejected(env_cfg, hsv):
    img = S.make_env_scene(a_objs=[(8, 100, 40, 0)])  # 一半在站台外 -> 被站台邊界截斷
    squares, rejected = detect_squares(_warped_mask(img, env_cfg, hsv), env_cfg['square'])
    assert not squares
    assert rejected and rejected[0].touches_border


def test_tiny_noise_is_ignored(env_cfg):
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 6, 255, -1)  # 面積遠小於 min_area_px
    assert analyze_blobs(mask, env_cfg['square']) == []
