"""低次元代理モデル (Ct/Cq 多項式 + 斜め流入の 1 次項).

制御系シミュレーションや HILS のように 10^6 回オーダで評価する用途向け.
係数は BEMT などの高忠実度モデル (あるいは実測データ) から
``QuadraticModel.fit`` で同定できる.

    T      = rho n^2 D^4 (ct0 + ct1 J + ct2 J^2)
    Q      = rho n^2 D^5 (cq0 + cq1 J + cq2 J^2)
    F_edge = rho n^2 D^4 mu_e (cf0 + cf1 J)      # 面内力 (法線力)
    M_edge = rho n^2 D^5 mu_e (cm0 + cm1 J)      # ハブモーメント

``mu_e = V_edge / (n D)`` は面内前進率.
面内力は流入の面内方向 (機体が進む向き) と逆向き, ハブモーメントは
その 90 deg 位相に現れる.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frames import Wrench
from ..inflow import InflowField
from ..operating import OperatingPoint
from ..rotor import Rotor
from .base import AeroModel, RotorSolution

__all__ = ["QuadraticModel"]


@dataclass
class QuadraticModel(AeroModel):
    """前進率の多項式で表した代理モデル."""

    name: str = "quadratic"
    ct: tuple[float, float, float] = (0.105, -0.030, -0.080)
    cq: tuple[float, float, float] = (0.0075, 0.0030, -0.0090)
    cf: tuple[float, float] = (0.0, 0.0)
    cm: tuple[float, float] = (0.0, 0.0)
    j_limits: tuple[float, float] = (-0.2, 1.2)
    source: str = "default (generic 2-blade UAV propeller)"
    _fit_info: dict = field(default_factory=dict, repr=False)

    def solve(self, rotor: Rotor, op: OperatingPoint) -> RotorSolution:
        geo = rotor.geometry
        rho = op.atmosphere.density
        n = op.n_rps
        d = geo.diameter
        r_R = rotor.radial_grid(8)

        if abs(n) < 1e-9:
            return RotorSolution(Wrench.zeros(), InflowField.uniform(r_R, 0.0))

        j = float(np.clip(op.v_axial / (n * d), *self.j_limits))
        mu_e = op.v_edge / (n * d)

        thrust = rho * n**2 * d**4 * np.polyval(self.ct[::-1], j)
        torque = rho * n**2 * d**5 * np.polyval(self.cq[::-1], j)
        f_edge = rho * n**2 * d**4 * mu_e * (self.cf[0] + self.cf[1] * j)
        m_edge = rho * n**2 * d**5 * mu_e * (self.cm[0] + self.cm[1] * j)

        e_e = op.edge_direction                      # 面内速度の向き
        e_p = np.cross(np.array([0.0, 0.0, 1.0]), e_e)  # その 90 deg 位相

        force = thrust * np.array([0.0, 0.0, 1.0]) + f_edge * e_e
        moment = -rotor.spin * torque * np.array([0.0, 0.0, 1.0]) + m_edge * e_p

        # 参考用に運動量理論の一様誘導速度を付けておく
        v_i = _momentum_inflow(thrust, rho, geo.disk_area, op.v_axial, op.v_edge)
        return RotorSolution(
            Wrench(force, moment), InflowField.uniform(r_R, v_i), None, True, 0
        )

    def instantaneous_wrench(self, rotor, op, inflow, psi1):
        """代理モデルは方位角依存性を持たないため定常値をそのまま返す."""
        return self.solve(rotor, op).wrench

    def step(self, rotor, op, inflow, psi1):
        return self.solve(rotor, op).wrench, np.zeros_like(inflow.v0)

    # ------------------------------------------------------------------ 同定
    @classmethod
    def fit(
        cls,
        rotor: Rotor,
        reference: AeroModel,
        *,
        rpm: float = 5000.0,
        advance_ratios: np.ndarray | None = None,
        inflow_angles_deg: np.ndarray | None = None,
        atmosphere=None,
    ) -> "QuadraticModel":
        """高忠実度モデルの計算結果から係数を最小二乗同定する."""
        from ..atmosphere import SEA_LEVEL

        atmosphere = atmosphere or SEA_LEVEL
        geo = rotor.geometry
        if advance_ratios is None:
            advance_ratios = np.linspace(0.0, 0.8, 9)
        if inflow_angles_deg is None:
            inflow_angles_deg = np.array([0.0, 20.0, 40.0])

        n = rpm / 60.0
        d = geo.diameter
        rho = atmosphere.density

        # --- 軸流 (J のみ) で Ct, Cq を同定
        rows, ct_v, cq_v = [], [], []
        for j in advance_ratios:
            op = OperatingPoint(
                rpm=rpm, v_inf=j * n * d, inflow_angle_deg=0.0, atmosphere=atmosphere
            )
            sol = reference.solve(rotor, op)
            rows.append([1.0, j, j**2])
            ct_v.append(sol.thrust / (rho * n**2 * d**4))
            cq_v.append(sol.torque(rotor.spin) / (rho * n**2 * d**5))
        a = np.asarray(rows)
        ct, *_ = np.linalg.lstsq(a, np.asarray(ct_v), rcond=None)
        cq, *_ = np.linalg.lstsq(a, np.asarray(cq_v), rcond=None)

        # --- 斜め流入で面内力・ハブモーメントを同定
        rows_e, f_v, m_v = [], [], []
        for j in advance_ratios:
            for ang in inflow_angles_deg:
                if ang == 0.0 or j == 0.0:
                    continue
                v = j * n * d / max(np.cos(np.deg2rad(ang)), 1e-3)
                op = OperatingPoint(
                    rpm=rpm, v_inf=v, inflow_angle_deg=float(ang),
                    inflow_azimuth_deg=0.0, atmosphere=atmosphere,
                )
                sol = reference.solve(rotor, op)
                mu_e = op.v_edge / (n * d)
                if mu_e < 1e-6:
                    continue
                j_ax = op.v_axial / (n * d)
                rows_e.append([mu_e, mu_e * j_ax])
                f_v.append(sol.wrench.force[0] / (rho * n**2 * d**4))
                m_v.append(sol.wrench.moment[1] / (rho * n**2 * d**5))
        if rows_e:
            ae = np.asarray(rows_e)
            cf, *_ = np.linalg.lstsq(ae, np.asarray(f_v), rcond=None)
            # e_p = z x e_e = +y なので My は -e_p 成分に相当する
            cm_fit, *_ = np.linalg.lstsq(ae, np.asarray(m_v), rcond=None)
        else:  # pragma: no cover
            cf = np.zeros(2)
            cm_fit = np.zeros(2)

        return cls(
            ct=tuple(ct),
            cq=tuple(cq),
            cf=tuple(cf),
            cm=tuple(cm_fit),
            j_limits=(float(min(advance_ratios)) - 0.1,
                      float(max(advance_ratios)) + 0.1),
            source=f"fitted to {reference.name} @ {rpm:.0f} rpm",
            _fit_info={
                "rpm": rpm,
                "advance_ratios": np.asarray(advance_ratios).tolist(),
                "inflow_angles_deg": np.asarray(inflow_angles_deg).tolist(),
            },
        )


def _momentum_inflow(
    thrust: float, rho: float, area: float, v_axial: float, v_edge: float
) -> float:
    """運動量理論の一様誘導速度 (参考値)."""
    v = np.sqrt(abs(thrust) / (2.0 * rho * area)) * np.sign(thrust)
    for _ in range(30):
        s = np.hypot(v_edge, v_axial + v)
        s = max(s, 1e-6)
        f = 2.0 * rho * area * v * s - thrust
        df = 2.0 * rho * area * (s + v * (v_axial + v) / s)
        v -= f / max(abs(df), 1e-9) * np.sign(df)
    return float(v)
