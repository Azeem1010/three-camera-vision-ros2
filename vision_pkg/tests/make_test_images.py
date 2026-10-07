"""把合成場景輸出成 PNG（tests/data/），可拿去餵 hsv_tuner --image 練習調色。

    python tests/make_test_images.py
"""
from pathlib import Path

import synthetic as S

DATA = Path(__file__).resolve().parent / 'data'

SCENES = {
    'env_empty.png': lambda: S.make_env_scene(),
    'env_a_8objects.png': lambda: S.make_env_scene(a_objs=S.a_objects()),
    'env_a_7objects.png': lambda: S.make_env_scene(a_objs=S.a_objects(skip=[3])),
    'env_b_arm.png': lambda: S.make_env_scene(arm_in_b=True),
    'arm2_c_ascending.png': lambda: S.make_arm2_scene([28, 30, 32, 34, 36, 38, 40, 42]),
    'arm2_c_wrong_order.png': lambda: S.make_arm2_scene([28, 30, 38, 34, 36, 32, 40, 42]),
}


def ensure_images() -> Path:
    """缺少的 PNG 就補生（測試啟動時呼叫），已存在的不覆蓋。"""
    for name, fn in SCENES.items():
        p = DATA / name
        if not p.exists():
            S.imwrite(p, fn())
    return DATA


if __name__ == '__main__':
    for name, fn in SCENES.items():
        S.imwrite(DATA / name, fn())
        print('wrote', DATA / name)
