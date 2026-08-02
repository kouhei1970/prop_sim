"""翼素運動量理論 (BEMT) と翼素理論 + 一様インフロー (BET).

BEMT
----
各環状要素 (アニュラス) について

    翼素理論の推力 dT/dr = sum_b <dFz/dr>_psi
    運動量理論の推力 dT/dr = 4 pi r rho F v0 * sqrt(V_edge^2 + (V_ax + v0)^2)

が釣り合うように誘導速度 v0(r) を求める. ``F`` は Prandtl の翼端/ハブ
損失係数. 斜め流入 (V_edge != 0) では Glauert の一般化運動量式を用い,
方位角方向には線形インフロー分布 (Drees / Pitt / Coleman) を重ねる.
これにより 1/rev の翼素荷重変動が生じ, ハブモーメント Mx, My と
面内力 Fx, Fy が現れる — これが 6 分力計測の主対象である.

数値解法は各アニュラス独立の二分法 (ベクトル化) で, 発散しない.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..inflow import InflowField, skew_coefficients
from ..operating import OperatingPoint
from ..rotor import Rotor
from .base import AeroModel, RotorSolution

__all__ = ["BEMT", "BET"]


@dataclass
class BEMT(AeroModel):
    """翼素運動量理論.

    Parameters
    ----------
    n_radial:
        半径方向の翼素分割数.
    n_azimuth:
        方位角方向の分割数. 軸流のみの場合は自動的に 1 に縮退する.
    tip_loss, hub_loss:
        Prandtl の翼端/ハブ損失を使うか.
    loss_floor:
        損失係数の下限 (根元・翼端での特異点回避).
    skew_model:
        斜め流入時の方位角インフロー分布 (``drees``/``pitt``/``coleman``/``none``).
    n_outer:
        スキュー係数更新の外側反復回数.
    n_bisect:
        誘導速度を求める二分法の反復回数.
    inflow_time_constant:
        動的インフローの見かけ質量係数 (Carpenter-Fridovich の 0.85).
    """

    name: str = "bemt"
    n_radial: int = 24
    n_azimuth: int = 24
    tip_loss: bool = True
    hub_loss: bool = True
    loss_floor: float = 0.05
    skew_model: str = "drees"
    n_outer: int = 4
    n_bisect: int = 26
    inflow_time_constant: float = 0.85
    supports_dynamic_inflow: bool = True

    # ------------------------------------------------------------- 補助関数
    def radial_grid(self, rotor: Rotor) -> np.ndarray:
        return rotor.radial_grid(self.n_radial)

    def azimuth_grid(self, op: OperatingPoint) -> np.ndarray:
        n = 1 if self._is_axial(op) else self.n_azimuth
        return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    @staticmethod
    def _is_axial(op: OperatingPoint) -> bool:
        return op.v_edge < 1e-9 and not bool(np.any(op.body_rate[:2]))

    def loss_factor(
        self, rotor: Rotor, op: OperatingPoint, r_R: np.ndarray, v0: np.ndarray
    ) -> np.ndarray:
        """Prandtl の翼端/ハブ損失係数 F(r)."""
        geo = rotor.geometry
        r = r_R * geo.radius
        u_p = op.v_axial + v0
        u_t = op.omega * r
        s_phi = np.abs(u_p) / np.maximum(np.hypot(u_p, u_t), 1e-9)
        s_phi = np.maximum(s_phi, 1e-3)

        f = np.ones_like(r_R)
        b2 = 0.5 * geo.n_blades
        if self.tip_loss:
            ft = b2 * (geo.radius - r) / np.maximum(r * s_phi, 1e-12)
            f = f * (2.0 / np.pi) * np.arccos(np.exp(-np.clip(ft, 0.0, 50.0)))
        if self.hub_loss and geo.hub_radius > 0.0:
            fh = b2 * (r - geo.hub_radius) / np.maximum(geo.hub_radius * s_phi, 1e-12)
            f = f * (2.0 / np.pi) * np.arccos(np.exp(-np.clip(fh, 0.0, 50.0)))
        return np.clip(f, self.loss_floor, 1.0)

    def momentum_thrust_gradient(
        self,
        rotor: Rotor,
        op: OperatingPoint,
        r_R: np.ndarray,
        v0: np.ndarray,
        loss: np.ndarray,
    ) -> np.ndarray:
        """運動量理論による dT/dr [N/m]."""
        rho = op.atmosphere.density
        r = r_R * rotor.geometry.radius
        u_disk = np.hypot(op.v_edge, op.v_axial + v0)
        return 4.0 * np.pi * r * rho * loss * v0 * u_disk

    def blade_thrust_gradient(
        self,
        rotor: Rotor,
        op: OperatingPoint,
        r_R: np.ndarray,
        psi: np.ndarray,
        inflow: InflowField,
    ) -> np.ndarray:
        """翼素理論による方位平均 dT/dr [N/m] (全ブレード合計)."""
        total = np.zeros_like(r_R)
        for b in range(rotor.n_blades):
            st = rotor.section_state(
                r_R, psi, inflow, op,
                pitch_offset_deg=float(rotor.blade_pitch_offsets_deg[b]),
            )
            total += st.dfz_dr.mean(axis=0)
        return total

    # ------------------------------------------------------------------ 求解
    def solve(self, rotor: Rotor, op: OperatingPoint) -> RotorSolution:
        geo = rotor.geometry
        r_R = self.radial_grid(rotor)
        psi = self.azimuth_grid(op)
        warns: list[str] = []

        if op.omega < 1e-6:
            inflow = InflowField.uniform(r_R, 0.0)
            w = rotor.mean_aero_wrench(inflow, op, r_R, max(psi.size, 1))
            return RotorSolution(w, inflow, converged=True)

        v_max = 0.5 * op.omega * geo.radius + abs(op.v_axial) + op.v_edge
        psi_edge = float(np.arctan2(op.v_hub[1], op.v_hub[0]))
        kx = ky = 0.0
        v0 = np.zeros_like(r_R)
        lo = np.full_like(r_R, -v_max)
        hi = np.full_like(r_R, +v_max)

        n_outer = 1 if self._is_axial(op) else max(1, self.n_outer)
        iterations = 0
        for outer in range(n_outer):

            def residual(v: np.ndarray) -> np.ndarray:
                field = InflowField(r_R, v, kx, ky, psi_edge)
                loss = self.loss_factor(rotor, op, r_R, v)
                dt_be = self.blade_thrust_gradient(rotor, op, r_R, psi, field)
                dt_mom = self.momentum_thrust_gradient(rotor, op, r_R, v, loss)
                return dt_be - dt_mom

            r_lo, r_hi = residual(lo), residual(hi)
            bad = ~((r_lo > 0.0) & (r_hi < 0.0))
            if np.any(bad) and outer == 0:
                warns.append(
                    f"{int(np.count_nonzero(bad))} 個のアニュラスで解が"
                    "ブラケットの外にあります (誘導速度をクリップしました)"
                )
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo + hi)
                pos = residual(mid) > 0.0
                lo = np.where(pos, mid, lo)
                hi = np.where(pos, hi, mid)
                iterations += 1
            v0 = 0.5 * (lo + hi)

            if self._is_axial(op):
                break
            lam = (op.v_axial + float(np.mean(v0))) / (op.omega * geo.radius)
            kx_new, ky_new = skew_coefficients(
                op.mu(geo.radius), lam, self.skew_model
            )
            # 横方向の流入勾配は前進側/後退側の入れ替わりで符号が変わる
            ky_new *= rotor.spin
            if abs(kx_new - kx) < 1e-4 and abs(ky_new - ky) < 1e-4:
                kx, ky = kx_new, ky_new
                break
            kx, ky = kx_new, ky_new
            # 次の外側反復は前回解の近傍だけを探索する
            span = 0.25 * v_max
            lo = v0 - span
            hi = v0 + span

        inflow = InflowField(
            r_R, v0, kx, ky, psi_edge, converged=True, iterations=iterations
        )
        if op.v_axial < 0.0:
            ratio = -op.v_axial / max(float(np.mean(np.abs(v0))), 1e-6)
            if 0.2 < ratio < 1.8:
                warns.append(
                    "ボルテックスリング状態の可能性があります "
                    "(運動量理論は有効ではありません)"
                )
        inflow.warnings = tuple(warns)

        wrench = rotor.mean_aero_wrench(inflow, op, r_R, max(psi.size, 8))
        sections = rotor.section_state(
            r_R, np.linspace(0.0, 2.0 * np.pi, max(psi.size, 24), endpoint=False),
            inflow, op,
        )
        return RotorSolution(
            wrench, inflow, sections, converged=True,
            iterations=iterations, warnings=tuple(warns),
        )

    # ------------------------------------------------------- 動的インフロー
    def refresh_skew(self, rotor: Rotor, op: OperatingPoint, inflow: InflowField) -> None:
        """現在の状態からスキュー係数を更新する (その場書き換え)."""
        if self._is_axial(op) or op.omega < 1e-6:
            inflow.kx = inflow.ky = 0.0
            return
        r = rotor.geometry.radius
        lam = (op.v_axial + float(np.mean(inflow.v0))) / (op.omega * r)
        inflow.kx, inflow.ky = skew_coefficients(op.mu(r), lam, self.skew_model)
        inflow.ky *= rotor.spin
        inflow.psi_edge = float(np.arctan2(op.v_hub[1], op.v_hub[0]))

    def quasi_steady_inflow(
        self, rotor: Rotor, op: OperatingPoint, inflow: InflowField
    ) -> np.ndarray:
        """現在の翼素荷重に対応する準定常誘導速度 (運動量式の逆解き)."""
        r_R = inflow.r_R
        psi = self.azimuth_grid(op)
        dt_be = self.blade_thrust_gradient(rotor, op, r_R, psi, inflow)
        loss = self.loss_factor(rotor, op, r_R, inflow.v0)
        return _invert_momentum(
            dt_be, r_R * rotor.geometry.radius, op.atmosphere.density,
            loss, op.v_axial, op.v_edge, inflow.v0,
        )

    def inflow_time_constants(
        self, rotor: Rotor, op: OperatingPoint, inflow: InflowField
    ) -> float:
        """一様インフロー成分の時定数 [s].

        Pitt-Peters / Carpenter-Fridovich の見かけ質量から
        tau = 0.85 / (4 nu Omega), nu = v_i/(Omega R).
        """
        omega = max(op.omega, 1e-3)
        vtip = omega * rotor.geometry.radius
        nu = max(abs(inflow.mean_v0()) / max(vtip, 1e-6), 0.01)
        return float(np.clip(self.inflow_time_constant / (4.0 * nu * omega), 1e-4, 0.5))

    def inflow_derivative(
        self, rotor: Rotor, op: OperatingPoint, inflow: InflowField
    ) -> np.ndarray:
        v_qs = self.quasi_steady_inflow(rotor, op, inflow)
        tau = self.inflow_time_constants(rotor, op, inflow)
        return (v_qs - inflow.v0) / tau

    def step(
        self, rotor: Rotor, op: OperatingPoint, inflow: InflowField, psi1: float
    ):
        """瞬時レンチと動的インフロー微分を翼素評価 1 回分で求める.

        ブレードが現在いる方位角での翼素荷重をそのまま運動量収支にも使う.
        時間積分の各ステップで呼ばれるため, ``solve`` のような反復は行わない.
        """
        r_R = inflow.r_R
        psi_b = rotor.blade_azimuths(psi1)
        offsets = rotor.blade_pitch_offsets_deg
        if np.all(offsets == offsets[0]):
            # 全ブレード同一ピッチ -> 1 回の評価でまとめて処理する
            st = rotor.section_state(
                r_R, psi_b, inflow, op, pitch_offset_deg=float(offsets[0])
            )
            wrench = rotor._blade_wrench_from_state(st, reduce="sum")
            dt_dr = st.dfz_dr.sum(axis=0)
        else:
            wrench = None
            dt_dr = np.zeros_like(r_R)
            for b, pb in enumerate(psi_b):
                st = rotor.section_state(
                    r_R, np.asarray([pb]), inflow, op,
                    pitch_offset_deg=float(offsets[b]),
                )
                w = rotor._blade_wrench_from_state(st)
                wrench = w if wrench is None else wrench + w
                dt_dr += st.dfz_dr[0]

        loss = self.loss_factor(rotor, op, r_R, inflow.v0)
        v_qs = _invert_momentum(
            dt_dr, r_R * rotor.geometry.radius, op.atmosphere.density,
            loss, op.v_axial, op.v_edge, inflow.v0,
        )
        tau = self.inflow_time_constants(rotor, op, inflow)
        return wrench, (v_qs - inflow.v0) / tau

    def initial_inflow(self, rotor: Rotor, op: OperatingPoint) -> InflowField:
        return self.solve(rotor, op).inflow


@dataclass
class BET(BEMT):
    """翼素理論 + 一様インフロー (ディスク全体で 1 つの運動量収支).

    半径方向のインフロー分布を解かないぶん物理的には粗いが, 翼端損失
    モデルやアニュラス独立の仮定が結果に与える影響を切り分けたいとき
    の参照解として有用. 計算コストは BEMT とほぼ同等.
    """

    name: str = "bet-uniform"
    tip_loss: bool = False
    hub_loss: bool = False

    def solve(self, rotor: Rotor, op: OperatingPoint) -> RotorSolution:
        geo = rotor.geometry
        r_R = self.radial_grid(rotor)
        psi = self.azimuth_grid(op)
        rho = op.atmosphere.density

        if op.omega < 1e-6:
            inflow = InflowField.uniform(r_R, 0.0)
            return RotorSolution(
                rotor.mean_aero_wrench(inflow, op, r_R, max(psi.size, 1)), inflow
            )

        v_max = 0.5 * op.omega * geo.radius + abs(op.v_axial) + op.v_edge
        psi_edge = float(np.arctan2(op.v_hub[1], op.v_hub[0]))
        kx = ky = 0.0
        lo, hi = -v_max, v_max
        n_outer = 1 if self._is_axial(op) else max(1, self.n_outer)
        iterations = 0
        v = 0.0
        for outer in range(n_outer):

            def residual(vv: float) -> float:
                field = InflowField.uniform(r_R, vv, kx=kx, ky=ky, psi_edge=psi_edge)
                dt_be = self.blade_thrust_gradient(rotor, op, r_R, psi, field)
                thrust = float(np.trapezoid(dt_be, r_R * geo.radius))
                u_disk = np.hypot(op.v_edge, op.v_axial + vv)
                return thrust - 2.0 * rho * geo.disk_area * vv * u_disk

            for _ in range(self.n_bisect + 8):
                mid = 0.5 * (lo + hi)
                if residual(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
                iterations += 1
            v = 0.5 * (lo + hi)
            if self._is_axial(op):
                break
            lam = (op.v_axial + v) / (op.omega * geo.radius)
            kx, ky = skew_coefficients(op.mu(geo.radius), lam, self.skew_model)
            ky *= rotor.spin
            lo, hi = v - 0.25 * v_max, v + 0.25 * v_max

        inflow = InflowField.uniform(
            r_R, v, kx=kx, ky=ky, psi_edge=psi_edge, iterations=iterations
        )
        wrench = rotor.mean_aero_wrench(inflow, op, r_R, max(psi.size, 8))
        sections = rotor.section_state(
            r_R, np.linspace(0.0, 2.0 * np.pi, max(psi.size, 24), endpoint=False),
            inflow, op,
        )
        return RotorSolution(wrench, inflow, sections, iterations=iterations)

    def quasi_steady_inflow(
        self, rotor: Rotor, op: OperatingPoint, inflow: InflowField
    ) -> np.ndarray:
        geo = rotor.geometry
        r_R = inflow.r_R
        psi = self.azimuth_grid(op)
        dt_be = self.blade_thrust_gradient(rotor, op, r_R, psi, inflow)
        thrust = float(np.trapezoid(dt_be, r_R * geo.radius))
        # 2 rho A v sqrt(Ve^2 + (Va+v)^2) = T を v について解く
        # (4 pi r_eff rho = 2 rho A となる等価半径 r_eff = R^2/2 を使う)
        v = _invert_momentum(
            np.asarray([thrust]),
            np.asarray([0.5 * geo.radius**2]),
            op.atmosphere.density,
            np.asarray([1.0]),
            op.v_axial,
            op.v_edge,
            np.asarray([float(np.mean(inflow.v0))]),
        )
        return np.full_like(r_R, float(v[0]))


def _invert_momentum(
    dt_target: np.ndarray,
    r: np.ndarray,
    rho: float,
    loss: np.ndarray,
    v_axial: float,
    v_edge: float,
    v_guess: np.ndarray,
    n_iter: int = 6,
) -> np.ndarray:
    """4 pi r rho F v sqrt(Ve^2 + (Va+v)^2) = dt_target を v について解く.

    減衰 Newton 法 (ベクトル化). 前ステップの解を初期値にするため
    数回の反復で十分収束する.
    """
    k = 4.0 * np.pi * r * rho * loss
    k = np.where(np.abs(k) < 1e-12, 1e-12, k)
    v = np.array(v_guess, dtype=float, copy=True)
    v_lim = 10.0 * (np.max(np.abs(v_guess)) + abs(v_axial) + v_edge + 1.0)
    lim_step = 0.25 * v_lim
    for _ in range(n_iter):
        va = v_axial + v
        s = np.maximum(np.hypot(v_edge, va), 1e-6)
        f = k * v * s - dt_target
        df = k * (s + v * va / s)
        df = np.where(np.abs(df) < 1e-9, 1e-9, df)
        step = np.minimum(np.maximum(f / df, -lim_step), lim_step)
        v = np.minimum(np.maximum(v - step, -v_lim), v_lim)
    return v
