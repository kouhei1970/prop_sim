"""プロペラ形状の定義.

実機のブレード形状 (コード長・ねじり角分布) を与えるのが基本だが,
"10x4.7" のような直径-ピッチ表記から代表的な分布を生成することもできる.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .airfoil import Airfoil, CLARK_Y, FLAT_PLATE

INCH = 0.0254

__all__ = ["PropellerGeometry", "BladeStation", "from_diameter_pitch"]


@dataclass
class BladeStation:
    """ブレード断面の定義 (単一半径位置)."""

    r_R: float          # 無次元半径 r/R
    chord_R: float      # コード長 c/R
    twist_deg: float    # 幾何ねじり角 (回転面基準)
    airfoil_index: int = 0


@dataclass
class PropellerGeometry:
    """プロペラのブレード形状.

    Parameters
    ----------
    radius:
        半径 R [m].
    n_blades:
        ブレード枚数 B.
    r_R, chord_R, twist_deg:
        半径方向分布 (同じ長さの配列). ``r_R`` は昇順.
    airfoils:
        翼型のリスト.
    airfoil_index:
        各ステーションが参照する翼型の添字. 省略時はすべて 0.
    hub_radius_ratio:
        スピナ/ハブ半径 r_hub/R. 翼素積分の内側境界になる.
    mass:
        ブレード + ハブ質量 [kg] (慣性・アンバランス計算に使用).
    polar_inertia:
        回転軸まわりの慣性モーメント Jz [kg m^2].
        None の場合 0.30 * m * R^2 で推定する.
    name:
        識別名.
    """

    radius: float
    n_blades: int
    r_R: np.ndarray
    chord_R: np.ndarray
    twist_deg: np.ndarray
    airfoils: list[Airfoil] = field(default_factory=lambda: [CLARK_Y])
    airfoil_index: np.ndarray | None = None
    hub_radius_ratio: float = 0.15
    mass: float = 0.012
    polar_inertia: float | None = None
    name: str = "propeller"

    def __post_init__(self) -> None:
        self.r_R = np.asarray(self.r_R, dtype=float)
        self.chord_R = np.asarray(self.chord_R, dtype=float)
        self.twist_deg = np.asarray(self.twist_deg, dtype=float)
        if not (self.r_R.shape == self.chord_R.shape == self.twist_deg.shape):
            raise ValueError("r_R / chord_R / twist_deg は同じ長さが必要です")
        if np.any(np.diff(self.r_R) <= 0):
            raise ValueError("r_R は狭義単調増加である必要があります")
        if self.airfoil_index is None:
            self.airfoil_index = np.zeros(self.r_R.size, dtype=int)
        else:
            self.airfoil_index = np.asarray(self.airfoil_index, dtype=int)
        if self.polar_inertia is None:
            # 細長いブレードの近似 (根元寄りに質量が偏る分を考慮)
            self.polar_inertia = 0.30 * self.mass * self.radius**2

    # ------------------------------------------------------------ 基本諸元
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def hub_radius(self) -> float:
        return self.hub_radius_ratio * self.radius

    @property
    def disk_area(self) -> float:
        return np.pi * self.radius**2

    def chord(self, r_R: np.ndarray) -> np.ndarray:
        """コード長 [m] (半径 r_R における)."""
        return np.interp(r_R, self.r_R, self.chord_R) * self.radius

    def twist(self, r_R: np.ndarray) -> np.ndarray:
        """幾何ねじり角 [rad]."""
        return np.deg2rad(np.interp(r_R, self.r_R, self.twist_deg))

    def airfoil_at(self, r_R: np.ndarray) -> np.ndarray:
        """各半径位置の翼型添字."""
        idx = np.interp(r_R, self.r_R, self.airfoil_index.astype(float))
        return np.rint(idx).astype(int)

    @property
    def solidity(self) -> float:
        """レートソリディティ sigma = B*c_75 / (pi*R)."""
        c75 = float(np.interp(0.75, self.r_R, self.chord_R))
        return self.n_blades * c75 / np.pi

    def thrust_weighted_solidity(self) -> float:
        """推力加重ソリディティ (面積比の等価値)."""
        x = np.linspace(self.hub_radius_ratio, 1.0, 200)
        c = np.interp(x, self.r_R, self.chord_R)
        return self.n_blades * float(np.trapezoid(c * x**2, x)) * 3.0 / np.pi

    def pitch_at(self, r_R: float = 0.75) -> float:
        """幾何ピッチ [m] (指定半径のねじり角から換算)."""
        beta = self.twist(np.asarray(r_R))
        return float(2.0 * np.pi * r_R * self.radius * np.tan(beta))

    def blade_area(self) -> float:
        """全ブレードの平面積 [m^2]."""
        x = np.linspace(self.hub_radius_ratio, 1.0, 200)
        c = np.interp(x, self.r_R, self.chord_R) * self.radius
        return self.n_blades * float(np.trapezoid(c, x * self.radius))

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return (
            f"{self.name}: D={self.diameter/INCH:.1f} in ({self.diameter*1e3:.0f} mm), "
            f"B={self.n_blades}, P0.75={self.pitch_at(0.75)/INCH:.2f} in, "
            f"sigma={self.solidity:.3f}, Jz={self.polar_inertia:.3e} kg.m^2"
        )


def _default_chord_shape(x: np.ndarray, x_root: float, x_peak: float = 0.42) -> np.ndarray:
    """代表的な小型プロペラのコード分布形状 (最大値 1.0 に正規化)."""
    x = np.asarray(x, dtype=float)
    s = np.empty_like(x)
    inner = x <= x_peak
    t_in = np.clip((x[inner] - x_root) / max(x_peak - x_root, 1e-6), 0.0, 1.0)
    s[inner] = 0.55 + 0.45 * np.sin(0.5 * np.pi * t_in)
    outer = ~inner
    t_out = np.clip((x[outer] - x_peak) / (1.0 - x_peak), 0.0, 1.0)
    s[outer] = np.sqrt(np.clip(1.0 - 0.94 * t_out**2, 1e-4, None))
    return s


def from_diameter_pitch(
    diameter_in: float,
    pitch_in: float,
    n_blades: int = 2,
    *,
    chord_ratio_75: float = 0.155,
    hub_radius_ratio: float = 0.15,
    pitch_distribution: str = "geometric",
    root_airfoil: Airfoil = CLARK_Y,
    tip_airfoil: Airfoil | None = FLAT_PLATE,
    tip_blend_r_R: float = 0.85,
    mass: float | None = None,
    n_stations: int = 25,
    name: str | None = None,
) -> PropellerGeometry:
    """"D x P" 表記から代表的なプロペラ形状を生成する.

    実機データがない段階での検討用. 実測のコード/ねじり分布があれば
    ``PropellerGeometry`` を直接組み立てること.

    Parameters
    ----------
    diameter_in, pitch_in:
        直径・ピッチ [inch].
    chord_ratio_75:
        r/R = 0.75 における c/R.
    pitch_distribution:
        ``"geometric"``  : 幾何ピッチ一定 beta = atan(P / (2 pi r))
        ``"constant"``   : ねじり角一定 (0.75R の値)
        ``"washout"``    : 幾何ピッチ一定から翼端で 2 deg 抜く
    """
    radius = 0.5 * diameter_in * INCH
    pitch = pitch_in * INCH
    x = np.linspace(hub_radius_ratio, 1.0, n_stations)

    shape = _default_chord_shape(x, hub_radius_ratio)
    shape75 = float(np.interp(0.75, x, shape))
    chord_R = shape / shape75 * chord_ratio_75

    beta_geo = np.arctan2(pitch, 2.0 * np.pi * x * radius)
    if pitch_distribution == "geometric":
        beta = beta_geo
    elif pitch_distribution == "constant":
        beta = np.full_like(x, float(np.interp(0.75, x, beta_geo)))
    elif pitch_distribution == "washout":
        beta = beta_geo - np.deg2rad(2.0) * np.clip((x - 0.5) / 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"未知の pitch_distribution: {pitch_distribution}")
    # 根元の極端なねじり角を抑える (実機のブレードルートは 45 deg 程度が上限)
    beta = np.minimum(beta, np.deg2rad(48.0))

    airfoils: list[Airfoil] = [root_airfoil]
    idx = np.zeros(x.size, dtype=int)
    if tip_airfoil is not None:
        airfoils.append(tip_airfoil)
        idx[x >= tip_blend_r_R] = 1

    if mass is None:
        # 樹脂製小型プロペラの経験式 (直径の 2.6 乗におおよそ比例)
        mass = 0.0115 * (diameter_in / 10.0) ** 2.6 * (n_blades / 2.0)

    return PropellerGeometry(
        radius=radius,
        n_blades=n_blades,
        r_R=x,
        chord_R=chord_R,
        twist_deg=np.rad2deg(beta),
        airfoils=airfoils,
        airfoil_index=idx,
        hub_radius_ratio=hub_radius_ratio,
        mass=mass,
        name=name or f"{diameter_in:g}x{pitch_in:g}",
    )
