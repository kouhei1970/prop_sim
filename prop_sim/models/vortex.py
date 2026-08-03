"""Lv3: 揚力線 + 渦後流 (規定後流 / 自由後流).

考え方
------
ブレードを半径方向に分割し, 各要素に束縛渦 (循環 Gamma_i) を置く.
循環の半径方向変化 dGamma は後縁から後流へ流出し, らせん状の渦フィラメント
になる. 各制御点での誘導速度を Biot-Savart 則で全フィラメントから直接
足し上げ, 翼素の有効迎角を更新して循環を再計算する — これを収束するまで
繰り返す.

BEMT との違い
-------------
============================  ==========================  ======================
                              BEMT                        揚力線 + 渦後流
============================  ==========================  ======================
誘導速度の決め方              アニュラスごとの運動量収支  後流渦の Biot-Savart
アニュラス間の干渉            なし (独立と仮定)           あり
翼端の荷重低下                Prandtl 損失係数で経験的に  翼端渦から自動的に
旋回 (swirl)                  無視                        束縛渦から直接
計算コスト                    ~30 ms                      ~0.5 s
============================  ==========================  ======================

つまり本モデルは **BEMT の 2 つの近似 (アニュラス独立 + 経験的翼端損失) を
外した参照解**であり, BEMT の妥当性を確かめるのに使う.

後流は**規定後流** (剛体らせん) で, 軸方向のピッチは運動量理論の完全発達
後流の速度 2*v_i から決める (Goldstein 以来の標準的な仮定).

制限
----
1. **軸流専用**. 斜め流入では循環が方位角とともに変化し, 放出渦
   (shed vorticity) を含む非定常問題になるため ``NotImplementedError``
   を出す — 斜め流入には BEMT を使うこと.
2. **自由後流 (後流形状の自己整合的な追跡) は未実装**. 素朴な緩和法では
   後流が数回転で絡まって発散する. 実用的な自由後流には予測子-修正子法
   (Bagai-Leishman PC2B など) と渦核成長モデルが要る. 近似としては
   ``wake_contraction`` で収縮を与えられるが, 0.9 を下回ると翼端渦が
   ブレードに近づいて数値的に硬くなる (収束しない場合は
   ``RotorSolution.converged`` が False になる).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..inflow import InflowField
from ..operating import OperatingPoint
from ..rotor import Rotor
from .base import AeroModel, RotorSolution

__all__ = ["LiftingLine", "biot_savart"]

#: ハブレンチを組み立てるときの方位角分割数. 軸流なので解自体は方位角に
#: よらないが, 面内成分を厳密に打ち消すために 1 回転ぶん平均する.
_N_AZIMUTH = 24


def biot_savart(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    gamma: np.ndarray,
    core: float,
    chunk: int = 4096,
) -> np.ndarray:
    """直線渦要素群が作る誘導速度 (Biot-Savart 則).

    Parameters
    ----------
    points:
        評価点 ``(P, 3)``.
    seg_a, seg_b:
        渦要素の始点・終点 ``(S, 3)``.
    gamma:
        各要素の循環 ``(S,)`` [m^2/s].
    core:
        渦核半径 [m] (特異点の正則化. Rankine 型).
    chunk:
        メモリを抑えるための評価点の分割数.

    Returns
    -------
    ``(P, 3)`` の誘導速度 [m/s].
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    out = np.zeros_like(points)
    if seg_a.size == 0:
        return out

    lvec = seg_b - seg_a                      # (S, 3)
    core2 = (core**2) * np.ones(seg_a.shape[0])
    for i0 in range(0, points.shape[0], chunk):
        p = points[i0:i0 + chunk]
        r1 = p[:, None, :] - seg_a[None, :, :]
        r2 = p[:, None, :] - seg_b[None, :, :]
        cr = np.cross(r1, r2)
        cr2 = np.einsum("psi,psi->ps", cr, cr)
        n1 = np.sqrt(np.einsum("psi,psi->ps", r1, r1))
        n2 = np.sqrt(np.einsum("psi,psi->ps", r2, r2))
        n1 = np.maximum(n1, 1e-12)
        n2 = np.maximum(n2, 1e-12)
        dot = np.einsum("si,psi->ps", lvec, r1 / n1[..., None] - r2 / n2[..., None])
        denom = cr2 + core2[None, :] * np.einsum("si,si->s", lvec, lvec)[None, :]
        k = gamma[None, :] / (4.0 * np.pi) * dot / np.maximum(denom, 1e-30)
        out[i0:i0 + chunk] = np.einsum("ps,psi->pi", k, cr)
    return out


@dataclass
class WakeGeometry:
    """後流フィラメントの節点座標.

    ``nodes`` の形状は ``(n_filament, n_node, 3)``. フィラメント f の
    循環は ``gamma[f]`` で, 全長にわたって一定 (定常・軸流の仮定).
    """

    nodes: np.ndarray
    gamma: np.ndarray

    def segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = self.nodes[:, :-1, :].reshape(-1, 3)
        b = self.nodes[:, 1:, :].reshape(-1, 3)
        g = np.repeat(self.gamma, self.nodes.shape[1] - 1)
        return a, b, g


@dataclass
class LiftingLine(AeroModel):
    """揚力線 + 渦後流モデル (軸流専用).

    Parameters
    ----------
    n_radial:
        ブレードの半径方向分割数 (制御点数).
    n_turns:
        後流を何回転分たどるか.
    n_per_turn:
        後流 1 回転あたりの節点数.
    wake_contraction:
        規定後流の収縮率. 0 なら収縮なし (Goldstein の剛体らせん後流),
        0.78 なら遠方で半径が 78 % になる (ホバーの実測に近い).
    wake_convection_factor:
        後流の軸方向移流速度を誘導速度の何倍にするか. 1.0 はディスク上の
        値, 既定の 2.0 は運動量理論の完全発達後流の値 (剛体らせん後流は
        遠方後流のピッチで作るのが Goldstein 以来の標準).
    core_radius_ratio:
        渦核半径 / コード長. 特異点の正則化に使う.
    n_iter, relax, tol:
        循環の反復解法のパラメータ.
    include_swirl:
        旋回誘導速度を翼素に反映するか (BEMT との差を見たいときは False).
    """

    name: str = "lifting-line"
    n_radial: int = 16
    n_turns: int = 6
    n_per_turn: int = 24
    wake_contraction: float = 0.0
    wake_convection_factor: float = 2.0
    core_radius_ratio: float = 0.25
    n_iter: int = 120
    relax: float = 0.30
    tol: float = 1.0e-4
    include_swirl: bool = True
    _diag: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ 格子
    def _panels(self, rotor: Rotor) -> tuple[np.ndarray, np.ndarray]:
        """パネル境界 (n+1) と制御点 (n) の無次元半径."""
        x0 = rotor.geometry.hub_radius_ratio
        s = np.linspace(0.0, 1.0, self.n_radial + 1)
        edges = x0 + (1.0 - x0) * np.sin(0.5 * np.pi * s)   # 翼端側を密に
        return edges, 0.5 * (edges[:-1] + edges[1:])

    # ------------------------------------------------------------ 後流形状
    def _prescribed_wake(
        self, rotor: Rotor, op: OperatingPoint, edges: np.ndarray, w_edge: np.ndarray
    ) -> np.ndarray:
        """らせん後流の節点座標 ``(n_blade * (n_r+1), n_node, 3)``."""
        geo = rotor.geometry
        n_node = self.n_turns * self.n_per_turn + 1
        zeta = np.linspace(0.0, 2.0 * np.pi * self.n_turns, n_node)   # 後流年齢
        omega = max(op.omega, 1e-6)

        # 収縮 (ホバーの実測に合わせた指数則)
        if self.wake_contraction > 0.0:
            lam = self.wake_contraction + (1.0 - self.wake_contraction) * np.exp(
                -0.6 * zeta
            )
        else:
            lam = np.ones_like(zeta)

        r_e = edges * geo.radius                       # (n_e,)
        nb = rotor.n_blades
        psi_b = rotor.blade_azimuths(0.0)              # (nb,)

        # 放出時の方位角 -> 現在の後流年齢 zeta だけ前に放出された
        ang = psi_b[:, None, None] - rotor.spin * zeta[None, None, :]   # (nb,1,n_node)
        rad = r_e[None, :, None] * lam[None, None, :]                   # (1,n_e,n_node)
        x = rad * np.cos(ang)
        y = rad * np.sin(ang)

        # 軸方向の移流: 一様流 + 誘導速度 (どちらも -z 方向へ後流を流す)
        w = self.wake_convection_factor * w_edge[None, :, None]
        z = -(op.v_axial + w) * (zeta[None, None, :] / omega)
        z = np.broadcast_to(z, x.shape)
        return np.stack([x, y, z], axis=-1).reshape(nb * edges.size, n_node, 3)

    # ------------------------------------------------------------ 求解
    def _circulation_loop(
        self,
        rotor: Rotor,
        op: OperatingPoint,
        edges: np.ndarray,
        r_c: np.ndarray,
        gamma: np.ndarray,
        w_ctrl: np.ndarray,
        frozen_nodes: np.ndarray | None,
    ):
        """循環 Gamma を収束させる内側ループ.

        ``frozen_nodes`` を与えると後流形状を固定して循環だけ解く
        (自由後流の 2 段目). None なら毎回らせん後流を作り直す.
        """
        geo = rotor.geometry
        nb = rotor.n_blades
        psi_b = rotor.blade_azimuths(0.0)
        r_ctrl = r_c * geo.radius
        chord = geo.chord(r_c)
        pts = np.stack([r_ctrl * np.cos(psi_b[0]), r_ctrl * np.sin(psi_b[0]),
                        np.zeros_like(r_ctrl)], axis=-1)
        e_t = rotor.spin * np.stack(
            [-np.sin(psi_b[0]) * np.ones_like(r_ctrl),
             np.cos(psi_b[0]) * np.ones_like(r_ctrl),
             np.zeros_like(r_ctrl)], axis=-1)
        bound_a, bound_b, own = self._bound_vortices(rotor, edges, psi_b)

        relax = self.relax
        prev = np.inf
        swirl = np.zeros_like(r_c)
        converged = False
        it = 0
        for it in range(1, self.n_iter + 1):
            g_pad = np.concatenate([[0.0], gamma, [0.0]])
            g_trail = -np.diff(g_pad) * rotor.spin
            if frozen_nodes is None:
                nodes = self._prescribed_wake(
                    rotor, op, edges, np.interp(edges, r_c, w_ctrl))
            else:
                nodes = frozen_nodes
            wa, wb, wg = WakeGeometry(nodes, np.tile(g_trail, nb)).segments()
            v = biot_savart(pts, wa, wb, wg, self._core)
            v = v + biot_savart(pts, bound_a[~own], bound_b[~own],
                                np.tile(gamma, nb)[~own] * rotor.spin, self._core)

            w_new = -v[:, 2]
            swirl = np.sum(v * e_t, axis=-1)
            u_t = op.omega * r_ctrl - (swirl if self.include_swirl else 0.0)
            u_p = op.v_axial + w_new
            u = np.hypot(u_t, u_p)
            alpha = (geo.twist(r_c) + np.deg2rad(rotor.collective_deg)
                     - np.arctan2(u_p, u_t))
            gamma_new = 0.5 * chord * u * _section_cl(rotor, r_c, alpha, u, op)

            res = float(np.max(np.abs(gamma_new - gamma))
                        / max(np.max(np.abs(gamma_new)), 1e-9))
            if res > prev:                       # 振動したら緩和を弱める
                relax = max(0.4 * relax, 0.02)
            prev = res
            gamma = gamma + relax * (gamma_new - gamma)
            w_ctrl = w_ctrl + relax * (w_new - w_ctrl)
            if res < self.tol:
                converged = True
                break
        return gamma, w_ctrl, swirl, converged, it, res

    @staticmethod
    def _bound_vortices(rotor: Rotor, edges: np.ndarray, psi_b: np.ndarray):
        """束縛渦の端点と「自ブレードかどうか」のマスク."""
        r = edges * rotor.geometry.radius
        n = edges.size - 1
        a = np.concatenate([
            np.stack([r[:-1] * np.cos(p), r[:-1] * np.sin(p), np.zeros(n)], -1)
            for p in psi_b])
        b = np.concatenate([
            np.stack([r[1:] * np.cos(p), r[1:] * np.sin(p), np.zeros(n)], -1)
            for p in psi_b])
        own = np.zeros(a.shape[0], dtype=bool)
        own[:n] = True                    # 揚力線の仮定: 自分の束縛渦は寄与しない
        return a, b, own

    def solve(self, rotor: Rotor, op: OperatingPoint) -> RotorSolution:
        geo = rotor.geometry
        if op.v_edge > 1e-9 or bool(np.any(op.body_rate)):
            raise NotImplementedError(
                "LiftingLine は軸流専用です (斜め流入では循環が方位角依存に "
                "なり非定常問題になります). 斜め流入には BEMT を使ってください."
            )
        edges, r_c = self._panels(rotor)
        self._core = float(self.core_radius_ratio * np.mean(geo.chord(r_c)))
        self._spin = rotor.spin
        if op.omega < 1e-6:
            inflow = InflowField.uniform(r_c, 0.0)
            return RotorSolution(
                rotor.mean_aero_wrench(inflow, op, r_c, _N_AZIMUTH), inflow)

        psi_b = rotor.blade_azimuths(0.0)
        gamma = np.zeros(r_c.size)
        w_ctrl = np.full(r_c.size, 0.02 * op.omega * geo.radius)
        warns: list[str] = []

        gamma, w_ctrl, swirl, conv, it, res = self._circulation_loop(
            rotor, op, edges, r_c, gamma, w_ctrl, None)
        total_it = it

        if not conv:
            warns.append(f"循環が収束しませんでした (残差 {res:.2e})")

        inflow = InflowField(
            r_c, w_ctrl, swirl=swirl if self.include_swirl else None,
            converged=conv, iterations=total_it)
        inflow.warnings = tuple(warns)
        # 軸流でも面内成分を厳密に 0 にするため方位角平均をとる
        wrench = rotor.mean_aero_wrench(inflow, op, r_c, _N_AZIMUTH)
        sections = rotor.section_state(r_c, np.asarray([psi_b[0]]), inflow, op)
        self._diag.update({"gamma": gamma.copy(), "r_R": r_c.copy(),
                           "swirl": swirl.copy()})
        return RotorSolution(wrench, inflow, sections, converged=conv,
                             iterations=total_it, warnings=tuple(warns))

    @property
    def circulation(self) -> dict:
        """最後に解いた循環分布 (診断用)."""
        return self._diag


def _section_cl(
    rotor: Rotor, r_R: np.ndarray, alpha: np.ndarray, u: np.ndarray,
    op: OperatingPoint,
) -> np.ndarray:
    """翼型の揚力係数 (Re / Mach 補正込み)."""
    geo = rotor.geometry
    atm = op.atmosphere
    chord = geo.chord(r_R)
    re = atm.density * u * chord / atm.dynamic_viscosity
    ma = u / atm.speed_of_sound
    idx = geo.airfoil_at(r_R)
    cl = np.empty_like(alpha)
    for k, foil in enumerate(geo.airfoils):
        m = idx == k
        if np.any(m):
            cl[m], _, _ = foil.coefficients(alpha[m], re[m], ma[m])
    return cl
