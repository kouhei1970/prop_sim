"""大気モデル (ISA 標準大気, 対流圏).

風洞・室内試験の条件設定に使う. 温度オフセットや実測の気圧・湿度を
与えて実験室条件を再現できるようにしてある.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_AIR = 287.05287  # 乾燥空気の気体定数 [J/(kg K)]
R_VAPOR = 461.495  # 水蒸気の気体定数 [J/(kg K)]
GAMMA = 1.4
G0 = 9.80665
T0 = 288.15
P0 = 101325.0
LAPSE = -0.0065  # [K/m]


def _saturation_pressure(temp_k: float) -> float:
    """飽和水蒸気圧 [Pa] (Tetens の式)."""
    t_c = temp_k - 273.15
    return 610.78 * np.exp(17.27 * t_c / (t_c + 237.3))


@dataclass(frozen=True)
class Atmosphere:
    """空気の状態量.

    Parameters
    ----------
    altitude_m:
        ISA 高度 [m].
    delta_isa_k:
        ISA からの温度偏差 [K].
    pressure_pa:
        実測気圧 [Pa]. 指定した場合は高度より優先される.
    temperature_k:
        実測気温 [K]. 指定した場合は高度・温度偏差より優先される.
    humidity:
        相対湿度 [0-1]. 密度をわずかに下げる.
    """

    altitude_m: float = 0.0
    delta_isa_k: float = 0.0
    pressure_pa: float | None = None
    temperature_k: float | None = None
    humidity: float = 0.0

    @property
    def temperature(self) -> float:
        """気温 [K]."""
        if self.temperature_k is not None:
            return float(self.temperature_k)
        return T0 + LAPSE * self.altitude_m + self.delta_isa_k

    @property
    def pressure(self) -> float:
        """気圧 [Pa]."""
        if self.pressure_pa is not None:
            return float(self.pressure_pa)
        t_isa = T0 + LAPSE * self.altitude_m
        return P0 * (t_isa / T0) ** (-G0 / (LAPSE * R_AIR))

    @property
    def density(self) -> float:
        """空気密度 [kg/m^3] (湿り空気補正込み)."""
        t = self.temperature
        p = self.pressure
        p_v = self.humidity * _saturation_pressure(t)
        p_d = p - p_v
        return p_d / (R_AIR * t) + p_v / (R_VAPOR * t)

    @property
    def dynamic_viscosity(self) -> float:
        """粘性係数 [Pa s] (Sutherland の式)."""
        t = self.temperature
        return 1.458e-6 * t**1.5 / (t + 110.4)

    @property
    def kinematic_viscosity(self) -> float:
        """動粘性係数 [m^2/s]."""
        return self.dynamic_viscosity / self.density

    @property
    def speed_of_sound(self) -> float:
        """音速 [m/s]."""
        return float(np.sqrt(GAMMA * R_AIR * self.temperature))

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return (
            f"T={self.temperature:.2f} K, p={self.pressure/100:.2f} hPa, "
            f"rho={self.density:.4f} kg/m^3, a={self.speed_of_sound:.1f} m/s"
        )


SEA_LEVEL = Atmosphere()
