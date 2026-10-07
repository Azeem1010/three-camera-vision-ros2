"""參數工具：把扁平的 {'a.b.c': value} 字典轉成 dataclass（不含 ROS）。

節點端：ROS 參數 -> 扁平字典 -> dataclass_from_flat()
測試端：config/*.yaml -> load_ros_params_yaml() -> 同一套 dataclass_from_flat()
因此「參數名稱 = 前綴 + dataclass 欄位名」，YAML 與程式只有一份真相。
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Type, TypeVar

T = TypeVar('T')


def flatten(d: Mapping, prefix: str = '') -> Dict[str, Any]:
    """巢狀字典攤平成點分隔鍵（與 ROS 2 參數展開規則一致）。"""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def load_ros_params_yaml(path) -> Dict[str, Any]:
    """讀取 ROS 2 參數檔，回傳扁平字典（取第一個含 ros__parameters 的區段）。"""
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    for body in data.values():
        if isinstance(body, Mapping) and 'ros__parameters' in body:
            return flatten(body['ros__parameters'])
    raise ValueError(f'{path} 內找不到 ros__parameters 區段')


def _to_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(v)


_COERCE = {'float': float, 'int': int, 'bool': _to_bool, 'str': str}


def dataclass_from_flat(cls: Type[T], flat: Mapping[str, Any], prefix: str = '') -> T:
    """依 dataclass 欄位名稱，從 flat['prefix.field'] 取值建立物件；缺少的欄位一次列出。"""
    kwargs: Dict[str, Any] = {}
    missing = []
    for f in dataclasses.fields(cls):
        key = f'{prefix}.{f.name}' if prefix else f.name
        if key not in flat or flat[key] is None:
            missing.append(key)
            continue
        tname = f.type if isinstance(f.type, str) else getattr(f.type, '__name__', str(f.type))
        kwargs[f.name] = _COERCE[tname](flat[key])
    if missing:
        raise KeyError('缺少參數（請檢查 config/*.yaml）: ' + ', '.join(missing))
    return cls(**kwargs)
