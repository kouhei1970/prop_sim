"""ロータ (プロペラ + 取り付け情報) と翼素荷重の評価.

このモジュールは「与えられた誘導速度場のもとで翼素にどんな力が働くか」
だけを担当する. 誘導速度をどう決めるか (運動量理論, 渦法, ...) は
``prop_sim.models`` 側の責務であり, モデルを差し替えても本モジュールは
変わらない.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import E_Z, Wrench
from .geometry import PropellerGeometry
from .inflow import InflowField
from .operating import OperatingPoint

__all__ = ["Rotor", "Unbalance", "SectionState"]


@dataclass
class Unbalance:
    """回転アンバランス.

    Parameters
    ----------
    static:
        静アンバランス U = m*e [kg m]. 1/rev の面内力 U*Omega^2 を生む.
    static_phase_deg:
        アンバランス質量の方位 (ブレード 1 の方位角基準).
    couple:
        偶力アンバランス [kg m^2]. 1/rev の面内モーメント C*Omega^2 を生む.
    couple_phase_deg:
        偶力アンバランスの方位.
    """

    static: float = 0.0
    static_phase_deg: float = 0.0
    couple: float = 0.0
    couple_phase_deg: float = 0.0

    def wrench(self, psi1: float, omega: float) -> Wrench:
        """ブレード 1 が方位角 ``psi1`` にあるときの遠心アンバランス荷重."""
        w = Wrench.zeros()
        if self.static == 0.0 and self.couple == 0.0:
            return w
        om2 = omega**2
        if self.static != 0.0:
            a = psi1 + np.deg2rad(self.static_phase_deg)
            w.force[:] = self.static * om2 * np.array([np.cos(a), np.sin(a), 0.0])
        if self.couple != 0.0:
            b = psi1 + np.deg2rad(self.couple_phase_deg)
            w.moment[:] = self.couple * om2 * np.array([np.cos(b), np.sin(b), 0.0])
        return w


@dataclass
class SectionState:
    """翼素の状態量 (診断・可視化用). 形状は (n_psi, n_r)."""

    r_R: np.ndarray
    psi: np.ndarray
    u_t: np.ndarray
    u_p: np.ndarray
    phi: np.ndarray
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    reynolds: np.ndarray
    mach: np.ndarray
    dfz_dr: np.ndarray
    dft_dr: np.ndarray
    dm_span_dr: np.ndarray
    v_induced: np.ndarray


@dataclass
class Rotor:
    """プロペラ + 回転方向 + 取り付け誤差.

    Parameters
    ----------
    geometry:
        ブレード形状.
    spin:
        +1 : +z 軸まわり右ねじ方向に回転 / -1 : 逆回転.
    collective_deg:
        全ブレード共通のピッチオフセット [deg] (可変ピッチ機構や取付角誤差).
    blade_pitch_offsets_deg:
        ブレードごとのピッチずれ [deg]. 空力アンバランス (1/rev 荷重) の原因.
    blade_azimuth_offsets_deg:
        ブレードごとの取付方位ずれ [deg].
    unbalance:
        質量アンバランス.
    include_pitching_moment:
        翼型の空力モーメント (cm) をハブモーメントに含めるか.
    """

    geometry: PropellerGeometry
    spin: int = 1
    collective_deg: float = 0.0
    blade_pitch_offsets_deg: np.ndarray | None = None
    blade_azimuth_offsets_deg: np.ndarray | None = None
    unbalance: Unbalance = field(default_factory=Unbalance)
    include_pitching_moment: bool = True

    def __post_init__(self) -> None:
        if self.spin not in (1, -1):
            raise ValueError("spin は +1 または -1")
        b = self.geometry.n_blades
        self.blade_pitch_offsets_deg = _as_blade_array(
            self.blade_pitch_offsets_deg, b, "blade_pitch_offsets_deg"
        )
        self.blade_azimuth_offsets_deg = _as_blade_array(
            self.blade_azimuth_offsets_deg, b, "blade_azimuth_offsets_deg"
        )

    # --------------------------------------------------------------- 便利属性
    @property
    def n_blades(self) -> int:
        return self.geometry.n_blades

    @property
    def radius(self) -> float:
        return self.geometry.radius

    def radial_grid(self, n: int = 24) -> np.ndarray:
        """翼素積分に使う半径格子 (翼端側を密にする)."""
        s = np.linspace(0.0, 1.0, n)
        x0 = self.geometry.hub_radius_ratio
        return x0 + (1.0 - x0) * np.sin(0.5 * np.pi * s)

    def blade_azimuths(self, psi1: float) -> np.ndarray:
        """ブレード 1 が ``psi1`` のときの全ブレード方位角 [rad]."""
        b = self.n_blades
        base = psi1 + self.spin * 2.0 * np.pi * np.arange(b) / b
        return base + np.deg2rad(self.blade_azimuth_offsets_deg)

    # --------------------------------------------------- 内部キャッシュ
    def _radial_cache(self, r_R: np.ndarray):
        """半径格子に依存する量 (コード長・ねじり角・翼型マスク) をキャッシュ."""
        key = r_R.tobytes()
        cache = getattr(self, "_r_cache", None)
        if cache is None:
            cache = self._r_cache = {}
        item = cache.get(key)
        if item is None:
            geo = self.geometry
            idx = geo.airfoil_at(r_R)
            masks = [
                (k, idx == k) for k in range(len(geo.airfoils)) if np.any(idx == k)
            ]
            r = r_R * geo.radius
            # 台形則の重み (積分を行列積で行うため)
            w = np.zeros_like(r)
            if r.size > 1:
                d = np.diff(r)
                w[:-1] += 0.5 * d
                w[1:] += 0.5 * d
            item = (
                geo.chord(r_R)[None, :],
                geo.twist(r_R)[None, :],
                masks,
                r,
                w,
            )
            if len(cache) > 32:
                cache.clear()
            cache[key] = item
        return item

    def _azimuth_cache(self, psi: np.ndarray):
        """方位角格子に依存する単位ベクトルをキャッシュ."""
        key = (psi.tobytes(), self.spin)
        cache = getattr(self, "_psi_cache", None)
        if cache is None:
            cache = self._psi_cache = {}
        item = cache.get(key)
        if item is None:
            cp, sp = np.cos(psi)[:, None], np.sin(psi)[:, None]
            z = np.zeros_like(cp)
            e_r = np.stack([cp, sp, z], axis=-1)          # (n_psi, 1, 3)
            e_t = self.spin * np.stack([-sp, cp, z], axis=-1)
            if len(cache) > 32:
                cache.clear()
            item = cache[key] = (e_r, e_t)
        return item

    # ------------------------------------------------------------- 翼素の評価
    def section_state(
        self,
        r_R: np.ndarray,
        psi: np.ndarray,
        inflow: InflowField,
        op: OperatingPoint,
        pitch_offset_deg: float = 0.0,
    ) -> SectionState:
        """格子 (psi, r_R) 上の翼素状態を計算する.

        Parameters
        ----------
        r_R:
            形状 ``(n_r,)`` の無次元半径.
        psi:
            形状 ``(n_psi,)`` の方位角 [rad].
        """
        geo = self.geometry
        atm = op.atmosphere
        rho, mu_air, a_snd = atm.density, atm.dynamic_viscosity, atm.speed_of_sound

        r_R = np.atleast_1d(np.asarray(r_R, dtype=float))
        psi = np.atleast_1d(np.asarray(psi, dtype=float))
        chord, twist, masks, r, _ = self._radial_cache(r_R)
        e_r, e_t = self._azimuth_cache(psi)
        rr = r[None, :]                            # (1, n_r)

        v_hub = op.v_hub
        if op.has_body_rate:
            r_vec = e_r * rr[..., None]            # (n_psi, n_r, 3)
            v_rel = v_hub + np.cross(op.body_rate, r_vec)
            v_t = np.sum(v_rel * e_t, axis=-1)
            v_z = v_rel[..., 2]
        else:
            v_t = np.sum(v_hub * e_t, axis=-1)     # (n_psi, 1)
            v_z = v_hub[2]

        vi = inflow.induced(r_R[None, :], psi[:, None])

        u_t = op.omega * rr + v_t
        if inflow.has_swirl:
            # 後流の旋回はブレードの進行方向に空気を連れ回すので相対速度を減らす
            u_t = u_t - inflow.induced_swirl(r_R[None, :], psi[:, None])
        u_p = v_z + vi
        u = np.hypot(u_t, u_p)

        phi = np.arctan2(u_p, u_t)
        theta = twist + np.deg2rad(self.collective_deg + pitch_offset_deg)
        alpha = theta - phi

        reynolds = rho * u * chord / mu_air
        mach = u / a_snd

        alpha = np.broadcast_to(alpha, u.shape)
        cl = np.empty_like(u)
        cd = np.empty_like(u)
        cm = np.empty_like(u)
        for k, m in masks:
            foil = geo.airfoils[k]
            cl[:, m], cd[:, m], cm[:, m] = foil.coefficients(
                alpha[:, m], reynolds[:, m], mach[:, m]
            )

        q = 0.5 * rho * u**2 * chord
        dl = q * cl
        dd = q * cd
        cph, sph = np.cos(phi), np.sin(phi)
        dfz_dr = dl * cph - dd * sph
        dft_dr = dl * sph + dd * cph
        dm_span_dr = q * chord * cm  # 翼素の空力モーメント (span 軸まわり)

        return SectionState(
            r_R=r_R, psi=psi, u_t=u_t, u_p=u_p, phi=phi, alpha=alpha,
            cl=cl, cd=cd, reynolds=reynolds, mach=mach,
            dfz_dr=dfz_dr, dft_dr=dft_dr, dm_span_dr=dm_span_dr, v_induced=vi,
        )

    def _blade_wrench_from_state(
        self, st: SectionState, reduce: str = "none"
    ) -> Wrench:
        """翼素分布からハブレンチを求める.

        ``reduce`` = ``"mean"`` で方位角平均 (1 枚のブレードの 1 回転平均),
        ``"sum"`` で方位角方向の総和 (複数ブレードを同時に評価した場合),
        ``"none"`` で方位角ごとの値をそのまま返す.
        """
        _, _, _, r, weights = self._radial_cache(st.r_R)
        e_r, e_t = self._azimuth_cache(st.psi)
        e_z = np.broadcast_to(E_Z, e_r.shape)

        df = st.dfz_dr[..., None] * e_z - st.dft_dr[..., None] * e_t
        r_vec = e_r * r[None, :, None]
        dm = np.cross(r_vec, df)
        if self.include_pitching_moment:
            dm = dm + st.dm_span_dr[..., None] * self.spin * e_r

        # 台形則の重みとの縮約 (np.trapezoid より高速)
        force = np.tensordot(df, weights, axes=([1], [0]))    # (n_psi, 3)
        moment = np.tensordot(dm, weights, axes=([1], [0]))
        if reduce == "mean":
            force, moment = force.mean(axis=0), moment.mean(axis=0)
        elif reduce == "sum":
            force, moment = force.sum(axis=0), moment.sum(axis=0)
        elif force.shape[0] == 1:
            force, moment = force[0], moment[0]
        return Wrench(force, moment)

    # ------------------------------------------------------------ ハブレンチ
    def aero_wrench(
        self,
        psi1: float,
        inflow: InflowField,
        op: OperatingPoint,
        r_R: np.ndarray | None = None,
    ) -> Wrench:
        """瞬時の空力ハブレンチ (ブレード 1 が方位角 ``psi1``)."""
        r_R = self.radial_grid() if r_R is None else np.asarray(r_R)
        total = Wrench.zeros()
        for b, psi_b in enumerate(self.blade_azimuths(psi1)):
            st = self.section_state(
                r_R, np.asarray([psi_b]), inflow, op,
                pitch_offset_deg=float(self.blade_pitch_offsets_deg[b]),
            )
            total = total + self._blade_wrench_from_state(st)
        return total

    def mean_aero_wrench(
        self,
        inflow: InflowField,
        op: OperatingPoint,
        r_R: np.ndarray | None = None,
        n_azimuth: int = 24,
    ) -> Wrench:
        """1 回転平均の空力ハブレンチ."""
        r_R = self.radial_grid() if r_R is None else np.asarray(r_R)
        psi = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
        total = Wrench.zeros()
        for b in range(self.n_blades):
            st = self.section_state(
                r_R, psi, inflow, op,
                pitch_offset_deg=float(self.blade_pitch_offsets_deg[b]),
            )
            total = total + self._blade_wrench_from_state(st, reduce="mean")
        return total

    # -------------------------------------------------------------- 慣性荷重
    def inertial_wrench(
        self,
        op: OperatingPoint,
        psi1: float = 0.0,
        omega_dot: float = 0.0,
        include_unbalance: bool = True,
    ) -> Wrench:
        """回転慣性・ジャイロ・アンバランスによる反力.

        いずれも「プロペラアセンブリが試験台に及ぼす」符号で返す.

        ``include_unbalance=False`` にするとアンバランスによる 1/rev 成分を
        除く. 時間平均をとる静的試験では回転アンバランスの寄与は平均 0 に
        なるため, 定常値の計算ではこちらを使う.
        """
        jz = self.geometry.polar_inertia
        w = Wrench.zeros()
        # 角加速度に対する反トルク
        w.moment[2] -= self.spin * jz * omega_dot
        # ジャイロモーメント: -omega_b x h
        if op.has_body_rate:
            h = jz * self.spin * op.omega * E_Z
            w.moment[:] -= np.cross(op.body_rate, h)
        # 回転アンバランス (1/rev, 時間平均は 0)
        if include_unbalance:
            w = w + self.unbalance.wrench(psi1, op.omega)
        return w

    # ------------------------------------------------------------------ 諸元
    def thrust_of(self, wrench: Wrench) -> float:
        return float(np.asarray(wrench.force)[..., 2])

    def torque_of(self, wrench: Wrench) -> float:
        """軸トルク [N m] (正 = 駆動に必要なトルク)."""
        return float(-self.spin * np.asarray(wrench.moment)[..., 2])

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        d = "CCW(+z)" if self.spin > 0 else "CW(-z)"
        return f"{self.geometry.summary()} | rotation={d}"


def _as_blade_array(value, n_blades: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(n_blades)
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(n_blades, float(arr[0]))
    if arr.size != n_blades:
        raise ValueError(f"{name} の長さはブレード枚数 {n_blades} と一致させてください")
    return arr
