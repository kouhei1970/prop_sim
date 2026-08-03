"""プロペラ空力モデル群.

忠実度の段階
------------
================  ==========================  ==========  ====================
レベル             実装 / クラス                計算コスト   6 分力の再現
================  ==========================  ==========  ====================
0 代理モデル       :class:`QuadraticModel`      ~ us        Fz, Mz (+ 同定時のみ面内)
1 翼素+一様流入    :class:`BET`                 ~ ms        粗い
2 翼素運動量理論   :class:`BEMT`  (既定)        ~ ms        6 分力すべて
2.5 動的インフロー :class:`BEMT` + 時間積分     ~ ms        過渡応答込み
3 揚力線+渦後流    :class:`LiftingLine`         ~ 0.5 s     軸流のみ (翼端渦・旋回)
4 外部データ結合   :class:`SurrogateModel`      ~ us        与えたデータ次第
S 代理モデル       :class:`SurrogateModel`      ~ us        学習範囲内
================  ==========================  ==========  ====================

``get_model("bemt")`` で名前から生成できる.
"""

from __future__ import annotations

from .base import AeroModel, RotorSolution
from .bemt import BEMT, BET
from .quadratic import QuadraticModel
from .surrogate import SurrogateModel, SurrogateSamples
from .vortex import LiftingLine, biot_savart

__all__ = [
    "AeroModel",
    "RotorSolution",
    "BEMT",
    "BET",
    "QuadraticModel",
    "LiftingLine",
    "SurrogateModel",
    "SurrogateSamples",
    "biot_savart",
    "get_model",
    "register_model",
    "MODEL_REGISTRY",
]

MODEL_REGISTRY: dict[str, type[AeroModel]] = {
    "bemt": BEMT,
    "bet": BET,
    "quadratic": QuadraticModel,
    "lifting_line": LiftingLine,
    "surrogate": SurrogateModel,
}


def get_model(name: str, **kwargs) -> AeroModel:
    """名前から空力モデルを生成する."""
    key = name.lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"未知のモデル '{name}'. 利用可能: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def register_model(name: str, cls: type[AeroModel]) -> None:
    """独自モデルを登録する (プラグイン用)."""
    MODEL_REGISTRY[name.lower().replace("-", "_")] = cls
