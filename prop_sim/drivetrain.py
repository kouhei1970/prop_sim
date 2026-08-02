"""モータ / ESC / 回転慣性 (動的試験のスロットル応答用).

ブラシレスモータは定常特性の等価回路 (Kv, 巻線抵抗, 無負荷電流) で表す.
実測の Q-omega 特性がある場合は :class:`Drivetrain.from_table` を使う.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Motor", "ESC", "Drivetrain"]


@dataclass
class Motor:
    """アウトランナ BLDC モータの等価回路モデル.

    Parameters
    ----------
    kv:
        速度定数 [rpm/V] (無負荷).
    resistance:
        相間等価抵抗 [ohm].
    no_load_current:
        無負荷電流 [A].
    inertia:
        モータ回転子の慣性モーメント [kg m^2].
    max_current:
        電流制限 [A].
    """

    kv: float = 920.0
    resistance: float = 0.11
    no_load_current: float = 0.5
    inertia: float = 8.0e-6
    max_current: float = 30.0
    name: str = "motor"

    @property
    def kt(self) -> float:
        """トルク定数 [N m / A]."""
        return 60.0 / (2.0 * np.pi * self.kv)

    def electrical(self, voltage: float, omega: float) -> tuple[float, float, float]:
        """(軸トルク [N m], 電流 [A], 電気入力 [W]) を返す."""
        back_emf = self.kt * omega
        current = (voltage - back_emf) / max(self.resistance, 1e-6)
        current = float(np.clip(current, -self.max_current, self.max_current))
        torque = self.kt * (current - np.sign(max(omega, 1e-9)) * self.no_load_current)
        return float(torque), current, float(voltage * current)


@dataclass
class ESC:
    """ESC (スロットル -> 実効電圧) の 1 次遅れモデル."""

    battery_voltage: float = 14.8
    time_constant: float = 0.02
    voltage_sag: float = 0.02   # [V/A] 電池内部抵抗による電圧降下
    min_throttle: float = 0.0

    def voltage_command(self, throttle: float, current: float = 0.0) -> float:
        thr = float(np.clip(throttle, self.min_throttle, 1.0))
        return thr * self.battery_voltage - self.voltage_sag * current


@dataclass
class Drivetrain:
    """モータ + ESC + プロペラ慣性.

    回転数の運動方程式:

        J dOmega/dt = Q_motor - Q_aero - Q_friction

    ``Q_aero`` は空力トルク (常に回転を妨げる正の値), ``Q_friction``
    はベアリング損失.
    """

    motor: Motor = field(default_factory=Motor)
    esc: ESC = field(default_factory=ESC)
    prop_inertia: float = 5.0e-5
    friction_coulomb: float = 1.0e-4     # [N m]
    friction_viscous: float = 2.0e-7     # [N m /(rad/s)]

    @property
    def inertia(self) -> float:
        """回転部全体の慣性モーメント [kg m^2]."""
        return self.prop_inertia + self.motor.inertia

    def friction_torque(self, omega: float) -> float:
        return self.friction_coulomb * np.tanh(omega / 5.0) + (
            self.friction_viscous * omega
        )

    def omega_dot(
        self, omega: float, voltage: float, q_aero: float
    ) -> tuple[float, dict[str, float]]:
        """角加速度と補助量を返す."""
        q_m, current, p_elec = self.motor.electrical(voltage, omega)
        q_f = self.friction_torque(omega)
        acc = (q_m - q_aero - q_f) / self.inertia
        return float(acc), {
            "motor_torque": q_m,
            "current": current,
            "electrical_power": p_elec,
            "friction_torque": q_f,
        }

    # ------------------------------------------------------------------ 静定
    def steady_throttle(
        self, target_rpm: float, q_aero: float
    ) -> float:
        """目標回転数を保つために必要なスロットル (定常)."""
        omega = target_rpm * 2.0 * np.pi / 60.0
        q_need = q_aero + self.friction_torque(omega)
        current = q_need / self.motor.kt + self.motor.no_load_current
        voltage = self.motor.kt * omega + current * self.motor.resistance
        volt_avail = self.esc.battery_voltage - self.esc.voltage_sag * current
        return float(np.clip(voltage / max(volt_avail, 1e-6), 0.0, 1.5))

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return (
            f"Kv={self.motor.kv:.0f} rpm/V, R={self.motor.resistance*1e3:.0f} mOhm, "
            f"J={self.inertia:.2e} kg.m^2, Vbat={self.esc.battery_voltage:.1f} V"
        )
