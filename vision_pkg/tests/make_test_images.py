"""把合成場景輸出成 PNG（tests/data/），可拿去餵 hsv_tuner --image 練習調色。

    python tests/make_test_images.py
"""
from pathlib import Path

import synthetic as S

DATA = Path(__file__).resolve().parent / 'data'

SCENES = {
    'env_empty.png': lambda: S.make_env_scene(),
    'env_a_8caps.png': lambda: S.make_env_scene(a_caps=S.A_CAPS_MM),
    'env_c_3caps.png': lambda: S.make_env_scene(a_caps=S.A_CAPS_MM[3:], filled_slots=[1, 2, 3]),
    'env_b_arm.png': lambda: S.make_env_scene(arm_in_b=True),
    'arm2_c_empty.png': lambda: S.make_arm2_scene(),
    'arm2_c_slots_1_2_3.png': lambda: S.make_arm2_scene([1, 2, 3]),
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
