"""空力モデルの共通インターフェイス.

新しい忠実度のモデル (渦法, CFD 代理モデル, 実測テーブル等) を追加する
ときは ``AeroModel`` を継承して ``solve`` を実装すればよい.
実験スクリプト側のコードは一切変更せずにモデルを差し替えられる.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..frames import Wrench
from ..inflow import InflowField

if TYPE_CHECKING:  # pragma: no cover
    from ..operating import OperatingPoint
    from ..rotor import Rotor, SectionState

__all__ = ["AeroModel", "RotorSolution"]


@dataclass
class RotorSolution:
    """定常解 (1 回転平均).

    Attributes
    ----------
    wrench:
        ハブ中心まわりの空力 6 分力 (1 回転平均).
    inflow:
        収束した誘導速度場.
    sections:
        翼素の状態量 (診断用, モデルによっては None).
    """

    wrench: Wrench
    inflow: InflowField
    sections: "SectionState | None" = None
    converged: bool = True
    iterations: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------ 主要諸元
    @property
    def thrust(self) -> float:
        """推力 [N] (シャフト軸方向成分)."""
        return float(self.wrench.force[2])

    def torque(self, spin: int = 1) -> float:
        """軸トルク [N m] (正 = 駆動に必要なトルク)."""
        return float(-spin * self.wrench.moment[2])

    def power(self, op: "OperatingPoint", spin: int = 1) -> float:
        """軸動力 [W]."""
        return self.torque(spin) * op.omega

    def coefficients(self, rotor: "Rotor", op: "OperatingPoint") -> dict[str, float]:
        """無次元係数一式を返す.

        プロペラ流儀 (n, D 基準) とヘリコプタ流儀 (Omega R, A 基準) の
        両方を含む.
        """
        geo = rotor.geometry
        rho = op.atmosphere.density
        n = op.n_rps
        d = geo.diameter
        r = geo.radius
        a_disk = geo.disk_area
        omega = op.omega
        vtip = omega * r

        t = self.thrust
        q = self.torque(rotor.spin)
        p = q * omega

        out: dict[str, float] = {}
        if n > 1e-6:
            out["J"] = op.advance_ratio(d)
            out["Ct"] = t / (rho * n**2 * d**4)
            out["Cq"] = q / (rho * n**2 * d**5)
            out["Cp"] = p / (rho * n**3 * d**5)
            out["eta"] = (
                out["J"] * out["Ct"] / out["Cp"] if abs(out["Cp"]) > 1e-12 else 0.0
            )
        if vtip > 1e-6:
            out["CT"] = t / (rho * a_disk * vtip**2)
            out["CQ"] = q / (rho * a_disk * vtip**2 * r)
            out["CP"] = p / (rho * a_disk * vtip**3)
            ct = out["CT"]
            if ct > 1e-9 and abs(out["CP"]) > 1e-12:
                out["FM"] = ct**1.5 / np.sqrt(2.0) / out["CP"]
        if abs(p) > 1e-9:
            out["thrust_per_power"] = t / p          # [N/W]
            out["disk_loading"] = t / a_disk          # [N/m^2]
        return out


class AeroModel(ABC):
    """プロペラ空力モデルの基底クラス."""

    name: str = "aero-model"

    @abstractmethod
    def solve(self, rotor: "Rotor", op: "OperatingPoint") -> RotorSolution:
        """作動点 ``op`` における定常解を返す."""

    # -- 動的シミュレーション用 (対応するモデルのみ実装) --------------------
    supports_dynamic_inflow: bool = False

    def initial_inflow(self, rotor: "Rotor", op: "OperatingPoint") -> InflowField:
        """時間積分の初期インフロー状態."""
        return self.solve(rotor, op).inflow

    def inflow_derivative(
        self, rotor: "Rotor", op: "OperatingPoint", inflow: InflowField
    ) -> np.ndarray:
        """動的インフローの状態微分 d(v0)/dt [m/s^2]."""
        raise NotImplementedError(
            f"{self.name} は動的インフローに対応していません"
        )

    def instantaneous_wrench(
        self,
        rotor: "Rotor",
        op: "OperatingPoint",
        inflow: InflowField,
        psi1: float,
    ) -> Wrench:
        """与えられたインフロー状態での瞬時ハブレンチ."""
        return rotor.aero_wrench(psi1, inflow, op)

    def step(
        self,
        rotor: "Rotor",
        op: "OperatingPoint",
        inflow: InflowField,
        psi1: float,
    ) -> tuple[Wrench, np.ndarray]:
        """時間積分 1 ステップ分の (瞬時ハブレンチ, d v0/dt).

        既定は準定常 (インフローは変化しない) 扱い.
        """
        return self.instantaneous_wrench(rotor, op, inflow, psi1), np.zeros_like(
            inflow.v0
        )
