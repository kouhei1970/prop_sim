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
3 渦法 (未実装)    ``AeroModel`` を継承         ~ s         翼端渦まで
4 CFD  (未実装)    外部ソルバのラッパ           ~ h         最も高い
================  ==========================  ==========  ====================

``get_model("bemt")`` で名前から生成できる.
"""

from __future__ import annotations

from .base import AeroModel, RotorSolution
from .bemt import BEMT, BET
from .quadratic import QuadraticModel

__all__ = [
    "AeroModel",
    "RotorSolution",
    "BEMT",
    "BET",
    "QuadraticModel",
    "get_model",
    "MODEL_REGISTRY",
]

MODEL_REGISTRY: dict[str, type[AeroModel]] = {
    "bemt": BEMT,
    "bet": BET,
    "quadratic": QuadraticModel,
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
