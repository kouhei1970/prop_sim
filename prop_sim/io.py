"""結果の入出力 (CSV / JSON)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

__all__ = ["write_csv", "read_csv", "write_json"]


def write_csv(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    """列辞書を CSV に書き出す (すべて同じ長さであること)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    arrays = [np.asarray(columns[k]).ravel() for k in names]
    n = len(arrays[0]) if arrays else 0
    for name, a in zip(names, arrays):
        if len(a) != n:
            raise ValueError(f"列 '{name}' の長さが他と一致しません")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow([f"{a[i]:.9g}" for a in arrays])


def read_csv(path: str | Path) -> dict[str, np.ndarray]:
    """``write_csv`` で書いた CSV を読み戻す."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(data[name]) for name in (data.dtype.names or ())}


def write_json(path: str | Path, obj) -> None:
    """設定オブジェクト (dataclass / dict) を JSON に保存する."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if is_dataclass(o):
            return asdict(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)
