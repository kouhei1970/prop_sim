"""座標系とレンチ(力+モーメント)の定義.

座標系の定義
------------
ハブ座標系 (hub frame) を基準とする.

    z 軸 : シャフト軸.推力が正となる向き.
    x 軸 : 方位角 psi = 0 の基準方向 (ディスク面内).
    y 軸 : z x x で決まる右手系.

方位角 psi は +x から +y へ向かう向きを正とする.
ブレードの回転方向は ``spin`` (+1: +z まわり右ねじ / -1: 逆) で表し,
時刻 t におけるブレード方位角は psi_b = spin * Omega * t + psi_b0 となる.

6 分力の符号
------------
本シミュレータが出力する 6 分力は
"プロペラアセンブリが試験スタンド(=ロードセル)に及ぼすレンチ" と定義する.
これは実際の 6 分力計が読む値と一致し, 定常状態ではプロペラに働く
空力レンチと等しくなる.

    Fz > 0 : 推力
    Mz < 0 : spin = +1 のとき.空力トルクの反作用 (回転を妨げる向き)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

E_X = np.array([1.0, 0.0, 0.0])
E_Y = np.array([0.0, 1.0, 0.0])
E_Z = np.array([0.0, 0.0, 1.0])

COMPONENT_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass
class Wrench:
    """力とモーメントの組.

    Parameters
    ----------
    force, moment:
        形状 ``(..., 3)`` の配列. 先頭次元は時系列などに使える.
    """

    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.force = np.asarray(self.force, dtype=float)
        self.moment = np.asarray(self.moment, dtype=float)
        if self.force.shape != self.moment.shape:
            raise ValueError(
                f"force と moment の形状が一致しません: "
                f"{self.force.shape} != {self.moment.shape}"
            )
        if self.force.shape[-1] != 3:
            raise ValueError("force / moment は最終次元が 3 である必要があります")

    # ------------------------------------------------------------------ 演算
    def __add__(self, other: "Wrench") -> "Wrench":
        return Wrench(self.force + other.force, self.moment + other.moment)

    def __sub__(self, other: "Wrench") -> "Wrench":
        return Wrench(self.force - other.force, self.moment - other.moment)

    def __mul__(self, k: float) -> "Wrench":
        return Wrench(self.force * k, self.moment * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Wrench":
        return Wrench(self.force / k, self.moment / k)

    def __neg__(self) -> "Wrench":
        return Wrench(-self.force, -self.moment)

    # ------------------------------------------------------------ 座標変換
    def translate(self, offset: np.ndarray) -> "Wrench":
        """モーメントの基準点を ``offset`` だけ移動したレンチを返す.

        ``offset`` は現在の基準点から新しい基準点へのベクトル.
        M_new = M_old - offset x F
        """
        offset = np.asarray(offset, dtype=float)
        return Wrench(self.force, self.moment - np.cross(offset, self.force))

    def rotate(self, dcm: np.ndarray) -> "Wrench":
        """回転行列 ``dcm`` (新座標系 <- 現座標系) で成分を変換する."""
        dcm = np.asarray(dcm, dtype=float)
        return Wrench(self.force @ dcm.T, self.moment @ dcm.T)

    # ------------------------------------------------------------ 変換/表示
    def as_array(self) -> np.ndarray:
        """``(..., 6)`` の配列 [Fx, Fy, Fz, Mx, My, Mz] として返す."""
        return np.concatenate([self.force, self.moment], axis=-1)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Wrench":
        arr = np.asarray(arr, dtype=float)
        return cls(arr[..., :3], arr[..., 3:])

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = ()) -> "Wrench":
        return cls(np.zeros(shape + (3,)), np.zeros(shape + (3,)))

    def as_dict(self) -> dict[str, np.ndarray]:
        arr = self.as_array()
        return {name: arr[..., i] for i, name in enumerate(COMPONENT_NAMES)}

    def __repr__(self) -> str:  # pragma: no cover - 表示のみ
        if self.force.ndim == 1:
            f, m = self.force, self.moment
            return (
                f"Wrench(F=[{f[0]:+.4g}, {f[1]:+.4g}, {f[2]:+.4g}] N, "
                f"M=[{m[0]:+.4g}, {m[1]:+.4g}, {m[2]:+.4g}] N.m)"
            )
        return f"Wrench(shape={self.force.shape[:-1]})"


def skew(v: np.ndarray) -> np.ndarray:
    """ベクトルの歪対称行列 (v x a = skew(v) @ a)."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)
