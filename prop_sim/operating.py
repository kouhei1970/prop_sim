"""運転条件 (作動点) の定義."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atmosphere import Atmosphere, SEA_LEVEL

__all__ = ["OperatingPoint"]


@dataclass
class OperatingPoint:
    """プロペラの作動点.

    Parameters
    ----------
    rpm:
        回転数 [rev/min] (常に正. 回転方向は ``Rotor.spin`` で指定).
    v_inf:
        一様流の速さ [m/s]. 0 なら静止 (ホバー) 状態.
    inflow_angle_deg:
        シャフト軸 (+z) と機体進行方向のなす角 [deg].
        0   : 軸流 (前進飛行 / 通常の風洞軸流試験)
        90  : 完全な横流れ (エッジワイズ)
        180 : 後退流
    inflow_azimuth_deg:
        横流れ成分の方位 [deg]. +x を 0 とする.
    body_rate:
        ハブ (試験台) の角速度ベクトル [rad/s], ハブ座標系.
        ジャイロモーメントと非定常流入の発生源になる.
    atmosphere:
        大気条件.

    Notes
    -----
    ``v_hub`` は "空気に対するハブの速度" である. 風洞では気流速度 U に対し
    ハブは静止しているので, ハブから見た相対速度は -U となる. 本クラスでは
    「機体が空気に対して進む速度」で定義しているため, 推力方向に前進する
    軸流状態 (プロペラにとっての前進率 J > 0) が ``inflow_angle_deg = 0`` に
    対応する.
    """

    rpm: float = 5000.0
    v_inf: float = 0.0
    inflow_angle_deg: float = 0.0
    inflow_azimuth_deg: float = 0.0
    body_rate: np.ndarray = field(default_factory=lambda: np.zeros(3))
    atmosphere: Atmosphere = SEA_LEVEL

    def __post_init__(self) -> None:
        self.body_rate = np.asarray(self.body_rate, dtype=float).reshape(3)
        a = np.deg2rad(self.inflow_angle_deg)
        b = np.deg2rad(self.inflow_azimuth_deg)
        # 生成時に固定する (フィールドを直接書き換えず replace() を使うこと)
        self._v_hub = self.v_inf * np.array(
            [np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
        )
        self._omega = self.rpm * 2.0 * np.pi / 60.0
        self._has_body_rate = bool(np.any(self.body_rate))

    # ---------------------------------------------------------------- 速度
    @property
    def omega(self) -> float:
        """角速度の大きさ [rad/s]."""
        return self._omega

    @property
    def n_rps(self) -> float:
        """毎秒回転数 [rev/s]."""
        return self.rpm / 60.0

    @property
    def v_hub(self) -> np.ndarray:
        """空気に対するハブ速度ベクトル [m/s] (ハブ座標系)."""
        return self._v_hub

    @property
    def has_body_rate(self) -> bool:
        """試験台の角速度が与えられているか."""
        return self._has_body_rate

    @property
    def v_axial(self) -> float:
        """軸方向速度成分 [m/s] (正 = 推力方向へ前進)."""
        return float(self.v_hub[2])

    @property
    def v_edge(self) -> float:
        """面内速度成分の大きさ [m/s]."""
        return float(np.hypot(self.v_hub[0], self.v_hub[1]))

    @property
    def edge_direction(self) -> np.ndarray:
        """面内速度の単位ベクトル (面内速度 0 なら +x)."""
        ve = self.v_hub[:2]
        n = np.linalg.norm(ve)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return np.array([ve[0] / n, ve[1] / n, 0.0])

    # ---------------------------------------------------------- 無次元パラメータ
    def advance_ratio(self, diameter: float) -> float:
        """前進率 J = V / (n D) (軸方向成分基準)."""
        n = self.n_rps
        return float(self.v_axial / (n * diameter)) if n > 1e-9 else np.inf

    def mu(self, radius: float) -> float:
        """前進比 mu = V_edge / (Omega R)."""
        om = self.omega
        return float(self.v_edge / (om * radius)) if om > 1e-9 else np.inf

    def lambda_c(self, radius: float) -> float:
        """軸流入比 lambda_c = V_axial / (Omega R)."""
        om = self.omega
        return float(self.v_axial / (om * radius)) if om > 1e-9 else np.inf

    def tip_mach(self, radius: float) -> float:
        """翼端 Mach 数 (回転 + 一様流の合成)."""
        u_tip = np.hypot(self.omega * radius + self.v_edge, self.v_axial)
        return float(u_tip / self.atmosphere.speed_of_sound)

    def replace(self, **kwargs) -> "OperatingPoint":
        """一部のパラメータだけ変更したコピーを返す."""
        data = dict(
            rpm=self.rpm,
            v_inf=self.v_inf,
            inflow_angle_deg=self.inflow_angle_deg,
            inflow_azimuth_deg=self.inflow_azimuth_deg,
            body_rate=self.body_rate.copy(),
            atmosphere=self.atmosphere,
        )
        data.update(kwargs)
        return OperatingPoint(**data)

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return (
            f"rpm={self.rpm:.0f}, V={self.v_inf:.2f} m/s, "
            f"alpha_in={self.inflow_angle_deg:.1f} deg, "
            f"psi_in={self.inflow_azimuth_deg:.1f} deg"
        )
