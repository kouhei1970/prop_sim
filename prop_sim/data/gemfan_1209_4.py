"""Gemfan 1209-4 のメーカ公表試験データ (3.7 V).

出典
----
Gemfan 公式 "Test Datas" シート (Gemfan 1209-4, Voltage: 3.7 V).
スロットル 10-100 % に対する 電流 / 推力 / T.E.1 (推力効率 g/W).

**回転数は公表されていない.** 使用モータの型番も不明.

既知の内部矛盾
--------------
表の推力列と ``T.E.1 x 電気入力`` が一致しない.

===========  ======  ======  ======  ======  ======  ======
スロットル%    10      30      50      60      80      100
===========  ======  ======  ======  ======  ======  ======
推力列 [g]      1.00    4.00   12.00   16.00   22.00   29.00
T.E.1 x P      0.62    3.93   10.59   14.56   20.84   28.35
ずれ           -38%     -2%    -12%     -9%     -5%     -2%
===========  ======  ======  ======  ======  ======  ======

推力列は整数に丸められている (1, 2, 4, 7, 12, ...) ので, T.E.1 の方が
元の分解能を保っている可能性が高い. どちらを使うかで結論が変わりうる
ため, :func:`gemfan_1209_4_points` は ``source`` で選べるようにしてある.
"""

from __future__ import annotations

import numpy as np

from ..calibration import PowerCurvePoint

__all__ = ["GEMFAN_1209_4", "gemfan_1209_4_points"]

#: 公表値そのまま
GEMFAN_1209_4 = {
    "source": "Gemfan official test data sheet, Gemfan 1209-4, 3.7 V",
    "voltage_v": 3.7,
    "throttle_pct": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], float),
    "current_a": np.array(
        [0.24, 0.49, 0.81, 1.17, 1.59, 1.92, 2.24, 2.56, 2.93, 3.58]),
    "thrust_g": np.array([1, 2, 4, 7, 12, 16, 19, 22, 25, 29], float),
    "thrust_efficiency_g_per_w": np.array(
        [0.70, 1.04, 1.31, 1.60, 1.80, 2.05, 2.18, 2.20, 2.17, 2.14]),
    "rpm": None,          # 公表されていない
    "motor": None,        # 不明
    "note": "推力列と T.E.1 x P が最大 38 % 食い違う (モジュール docstring 参照)",
}


def gemfan_1209_4_points(source: str = "thrust_column") -> list[PowerCurvePoint]:
    """試験点のリストを返す.

    Parameters
    ----------
    source:
        ``"thrust_column"``  : 表の推力列をそのまま使う (既定)
        ``"efficiency"``     : ``T.E.1 x 電気入力`` から推力を復元する
    """
    d = GEMFAN_1209_4
    p_el = d["voltage_v"] * d["current_a"]
    if source == "thrust_column":
        thrust = d["thrust_g"]
    elif source == "efficiency":
        thrust = d["thrust_efficiency_g_per_w"] * p_el
    else:
        raise ValueError(f"未知の source: {source}")
    return [
        PowerCurvePoint(thrust_gf=float(t), power_w=float(p),
                        voltage=d["voltage_v"], current=float(c),
                        throttle_pct=float(th))
        for th, c, t, p in zip(d["throttle_pct"], d["current_a"], thrust, p_el)
    ]
