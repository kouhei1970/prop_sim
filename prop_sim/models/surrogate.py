"""Lv4 / LvS: データ駆動モデル (外部データの取り込みと代理モデル).

同じ仕組みで 2 つの用途をまかなう.

* **Lv4 の受け皿** — CFD や風洞実測の結果を読み込んで補間する.
  本パッケージで CFD を回すことはできないが, 外部ソルバ/実験の
  6 分力データをこのモデルに食わせれば, 試験シナリオ・センサモデル・
  後処理をそのまま流用できる.
* **LvS の代理モデル** — BEMT や渦法の結果を応答曲面に落として
  1 点あたり数マイクロ秒で評価する. 制御シミュレータや最適化の
  内側ループ向け.

無次元化
--------
学習も予測も**係数空間**で行う. 回転数によるスケーリング
(rho n^2 D^4) は解析的に効かせ, 補間するのは前進率まわりの依存性だけ.

    ct   = Fz / (rho n^2 D^4)          cq   = Q  / (rho n^2 D^5)
    cfx  = Fx'/ (rho n^2 D^4)          cmx  = Mx'/ (rho n^2 D^5)
    cfy  = Fy'/ (rho n^2 D^4)          cmy  = My'/ (rho n^2 D^5)

``'`` は「面内速度の方位を +x に合わせた座標系」の意味. 予測時は
実際の流入方位へ回して戻すので, 流入方位について学習データを持つ
必要がない (軸対称性を利用).

説明変数は ``J = V_axial/(nD)``, ``mu_e = V_edge/(nD)``, および
学習データが複数の回転数を含む場合は ``log10(n)`` (Reynolds 数効果).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..frames import Wrench
from ..inflow import InflowField
from ..operating import OperatingPoint
from ..rotor import Rotor
from .base import AeroModel, RotorSolution

__all__ = ["SurrogateModel", "SurrogateSamples"]

_COEFF_NAMES = ("ct", "cfx", "cfy", "cq", "cmx", "cmy")


@dataclass
class SurrogateSamples:
    """学習/補間に使うサンプル点 (係数空間)."""

    j: np.ndarray            # 軸方向前進率 V_axial/(nD)
    mu_e: np.ndarray         # 面内前進率  V_edge/(nD)
    log_n: np.ndarray        # log10(回転数 [rev/s])
    coeffs: np.ndarray       # (n_sample, 6) = ct, cfx, cfy, cq, cmx, cmy

    def __post_init__(self) -> None:
        self.j = np.asarray(self.j, dtype=float).ravel()
        self.mu_e = np.asarray(self.mu_e, dtype=float).ravel()
        self.log_n = np.asarray(self.log_n, dtype=float).ravel()
        self.coeffs = np.asarray(self.coeffs, dtype=float).reshape(self.j.size, 6)

    @property
    def n_sample(self) -> int:
        return self.j.size

    def features(self, mask: tuple[bool, bool, bool]) -> np.ndarray:
        """``mask`` = (J, mu_e, log_n) のうち使う列だけを取り出す."""
        cols = [c for c, m in zip((self.j, self.mu_e, self.log_n), mask) if m]
        return np.stack(cols, axis=1)


def _nondimensionalise(
    wrench: Wrench, rotor: Rotor, op: OperatingPoint
) -> np.ndarray:
    """6 分力を「面内速度方位を +x に合わせた」係数に直す."""
    geo = rotor.geometry
    rho = op.atmosphere.density
    n, d = op.n_rps, geo.diameter
    if abs(n) < 1e-9:
        return np.zeros(6)
    qf = rho * n**2 * d**4
    qm = rho * n**2 * d**5
    psi = np.arctan2(op.v_hub[1], op.v_hub[0]) if op.v_edge > 1e-12 else 0.0
    c, s = np.cos(-psi), np.sin(-psi)
    f, m = wrench.force, wrench.moment
    fx = c * f[0] - s * f[1]
    fy = s * f[0] + c * f[1]
    mx = c * m[0] - s * m[1]
    my = s * m[0] + c * m[1]
    q = -rotor.spin * m[2]
    return np.array([f[2] / qf, fx / qf, fy / qf, q / qm, mx / qm, my / qm])


def _redimensionalise(
    coeffs: np.ndarray, rotor: Rotor, op: OperatingPoint
) -> Wrench:
    """係数から実際の流入方位における 6 分力へ戻す."""
    geo = rotor.geometry
    rho = op.atmosphere.density
    n, d = op.n_rps, geo.diameter
    qf = rho * n**2 * d**4
    qm = rho * n**2 * d**5
    ct, cfx, cfy, cq, cmx, cmy = coeffs
    psi = np.arctan2(op.v_hub[1], op.v_hub[0]) if op.v_edge > 1e-12 else 0.0
    c, s = np.cos(psi), np.sin(psi)
    fx, fy = c * cfx * qf - s * cfy * qf, s * cfx * qf + c * cfy * qf
    mx, my = c * cmx * qm - s * cmy * qm, s * cmx * qm + c * cmy * qm
    return Wrench(
        np.array([fx, fy, ct * qf]),
        np.array([mx, my, -rotor.spin * cq * qm]),
    )


def _poly_powers(n_var: int, degree: int) -> list[tuple[int, ...]]:
    """全次数 ``degree`` までの単項式の指数の組 (変数の数は任意)."""
    return [
        p for p in product(range(degree + 1), repeat=n_var) if sum(p) <= degree
    ]


def _poly_terms(x: np.ndarray, degree: int) -> np.ndarray:
    """全次数 ``degree`` までの多項式基底."""
    n, k = x.shape
    powers = _poly_powers(k, degree)
    out = np.ones((n, len(powers)))
    for c, p in enumerate(powers):
        for v, e in enumerate(p):
            if e:
                out[:, c] *= x[:, v] ** e
    return out


#: 面内成分 (cfx, cfy, cmx, cmy) は軸流 (mu_e = 0) では厳密に 0 になる.
#: 多項式回帰ではこの対称性が自動では満たされないので, これらのチャネルは
#: mu_e の指数が 1 以上の項だけで張った基底に制限する.
_INPLANE = (1, 2, 4, 5)


@dataclass
class SurrogateModel(AeroModel):
    """係数空間で補間/回帰するデータ駆動モデル.

    Parameters
    ----------
    samples:
        学習データ.
    method:
        ``"poly"``   : 多項式最小二乗 (既定, 外挿に強く滑らか)
        ``"rbf"``    : 動径基底関数補間 (学習点を厳密に通る)
        ``"linear"`` : 線形補間 (散布データ, 凸包の外は最近傍)
    degree:
        ``method="poly"`` の次数.
    clip_to_hull:
        学習範囲を外れた入力を範囲端にクリップするか.

    Notes
    -----
    軸流 (mu_e = 0) では面内成分が厳密に 0 になるという対称性を
    ``method="poly"`` では基底の制限として明示的に課している.
    学習データが軸流のみの場合, 面内成分は常に 0 を返す (外挿しない).
    source:
        由来の記録 (BEMT 由来 / CFD / 風洞実測 など).
    """

    samples: SurrogateSamples | None = None
    name: str = "surrogate"
    method: str = "poly"
    degree: int = 3
    clip_to_hull: bool = True
    source: str = "unspecified"
    _fit: object = field(default=None, repr=False)
    _use_rpm: bool = field(default=False, repr=False)
    _has_mu: bool = field(default=True, repr=False)
    _mask: tuple = field(default=(True, True, False), repr=False)
    _range: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.samples is not None:
            self._build()

    # --------------------------------------------------------------- 学習
    def _build(self) -> None:
        sp = self.samples
        # 変動のない説明変数は落とす (軸流だけの学習データなら mu_e を外す)
        self._mask = (
            True,
            bool(np.ptp(sp.mu_e) > 1e-9),
            bool(np.ptp(sp.log_n) > 1e-3),
        )
        self._use_rpm = self._mask[2]
        self._has_mu = self._mask[1]
        x = sp.features(self._mask)
        self._range = {
            "min": x.min(axis=0), "max": x.max(axis=0),
            "j": (float(sp.j.min()), float(sp.j.max())),
            "mu_e": (float(sp.mu_e.min()), float(sp.mu_e.max())),
        }
        if self.method == "poly":
            a = _poly_terms(x, self.degree)
            if a.shape[0] < a.shape[1]:
                raise ValueError(
                    f"サンプル数 {a.shape[0]} が多項式の項数 {a.shape[1]} より"
                    f"少ないため回帰できません (degree を {self.degree} より "
                    "下げるか, サンプルを増やしてください)"
                )
            powers = _poly_powers(x.shape[1], self.degree)
            if self._has_mu:
                self._mu_mask = np.array([p[1] >= 1 for p in powers])
            else:
                self._mu_mask = np.zeros(len(powers), dtype=bool)
            coef = np.zeros((a.shape[1], 6))
            for k in range(6):
                if k in _INPLANE:
                    if not self._has_mu:
                        continue          # 面内成分は恒等的に 0
                    use = self._mu_mask
                else:
                    use = np.ones(len(powers), dtype=bool)
                c, *_ = np.linalg.lstsq(a[:, use], sp.coeffs[:, k], rcond=None)
                coef[use, k] = c
            self._fit = coef
        elif self.method == "rbf":
            from scipy.interpolate import RBFInterpolator

            scale = np.maximum(self._range["max"] - self._range["min"], 1e-9)
            self._scale = scale
            self._fit = RBFInterpolator(
                x / scale, sp.coeffs, kernel="thin_plate_spline",
                smoothing=1e-8, degree=1,
            )
        elif self.method == "linear":
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

            scale = np.maximum(self._range["max"] - self._range["min"], 1e-9)
            self._scale = scale
            self._fit = (
                LinearNDInterpolator(x / scale, sp.coeffs),
                NearestNDInterpolator(x / scale, sp.coeffs),
            )
        else:
            raise ValueError(f"未知の method: {self.method}")

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self.clip_to_hull:
            x = np.clip(x, self._range["min"], self._range["max"])
        x = x[None, :]
        if self.method == "poly":
            return (_poly_terms(x, self.degree) @ self._fit)[0]
        if self.method == "rbf":
            return self._fit(x / self._scale)[0]
        lin, near = self._fit
        out = lin(x / self._scale)[0]
        if not np.all(np.isfinite(out)):
            out = near(x / self._scale)[0]
        return out

    # --------------------------------------------------------------- 評価
    def solve(self, rotor: Rotor, op: OperatingPoint) -> RotorSolution:
        if self.samples is None or self._fit is None:
            raise RuntimeError("学習データがありません (fit / from_data を使う)")
        r_R = rotor.radial_grid(8)
        if abs(op.n_rps) < 1e-9:
            return RotorSolution(Wrench.zeros(), InflowField.uniform(r_R, 0.0))
        d = rotor.geometry.diameter
        allf = (op.v_axial / (op.n_rps * d), op.v_edge / (op.n_rps * d),
                np.log10(max(op.n_rps, 1e-6)))
        feats = [v for v, m in zip(allf, self._mask) if m]
        if not self._has_mu and op.v_edge > 1e-9:
            pass          # 学習データが軸流のみ -> 面内成分は 0 のまま返る
        coeffs = self._predict(np.asarray(feats))
        wrench = _redimensionalise(coeffs, rotor, op)
        v_i = _momentum_inflow(
            float(wrench.force[2]), op.atmosphere.density,
            rotor.geometry.disk_area, op.v_axial, op.v_edge,
        )
        return RotorSolution(wrench, InflowField.uniform(r_R, v_i))

    def instantaneous_wrench(self, rotor, op, inflow, psi1):
        return self.solve(rotor, op).wrench

    def step(self, rotor, op, inflow, psi1):
        return self.solve(rotor, op).wrench, np.zeros_like(inflow.v0)

    # -------------------------------------------------------- 学習データ作成
    @classmethod
    def from_conditions(
        cls,
        rotor: Rotor,
        operating_points,
        wrenches,
        *,
        source: str = "external data",
        **kwargs,
    ) -> "SurrogateModel":
        """作動点と 6 分力の対から作る (CFD / 実測の取り込み口).

        Parameters
        ----------
        operating_points:
            :class:`OperatingPoint` のリスト.
        wrenches:
            対応する :class:`Wrench` のリスト, または ``(n, 6)`` 配列
            ``[Fx, Fy, Fz, Mx, My, Mz]``.
        """
        ops = list(operating_points)
        arr = []
        for op, w in zip(ops, wrenches):
            if not isinstance(w, Wrench):
                w = Wrench.from_array(np.asarray(w, dtype=float))
            arr.append(_nondimensionalise(w, rotor, op))
        d = rotor.geometry.diameter
        sp = SurrogateSamples(
            j=[op.v_axial / (op.n_rps * d) for op in ops],
            mu_e=[op.v_edge / (op.n_rps * d) for op in ops],
            log_n=[np.log10(max(op.n_rps, 1e-6)) for op in ops],
            coeffs=np.asarray(arr),
        )
        return cls(samples=sp, source=source, **kwargs)

    @classmethod
    def from_static_result(
        cls, result, rotor: Rotor, *, use_measured: bool = False, **kwargs
    ) -> "SurrogateModel":
        """:func:`~prop_sim.experiments.static_sweep` の結果から作る.

        ``use_measured=True`` にすると 6 分力計を通した「計測値」の列を
        使う — 実験データから代理モデルを作る手順そのものの検証になる.
        """
        from ..atmosphere import SEA_LEVEL

        cols = result.columns
        n = result.n_points
        prefix = "" if use_measured and "Fx" in cols else "_true"
        ops, ws = [], []
        for i in range(n):
            ops.append(OperatingPoint(
                rpm=float(cols["rpm"][i]), v_inf=float(cols["v_inf"][i]),
                inflow_angle_deg=float(cols["inflow_angle_deg"][i]),
                inflow_azimuth_deg=float(cols["inflow_azimuth_deg"][i]),
                atmosphere=SEA_LEVEL,
            ))
            ws.append(np.array([cols[c + prefix][i] for c in
                                ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]))
        kwargs.setdefault("source", "static_sweep " +
                          ("(measured)" if prefix == "" else "(truth)"))
        return cls.from_conditions(rotor, ops, ws, **kwargs)

    @classmethod
    def fit(
        cls,
        rotor: Rotor,
        reference: AeroModel,
        *,
        rpm=None,
        advance_ratios=None,
        inflow_angles_deg=None,
        atmosphere=None,
        progress: bool = False,
        **kwargs,
    ) -> "SurrogateModel":
        """高忠実度モデルを掃引して代理モデルを同定する."""
        from ..atmosphere import SEA_LEVEL

        atmosphere = atmosphere or SEA_LEVEL
        rpm = np.atleast_1d(np.asarray(
            [6000.0] if rpm is None else rpm, dtype=float))
        advance_ratios = (np.linspace(0.0, 0.7, 8) if advance_ratios is None
                          else np.asarray(advance_ratios, dtype=float))
        inflow_angles_deg = (np.array([0.0, 20.0, 40.0, 60.0])
                             if inflow_angles_deg is None
                             else np.asarray(inflow_angles_deg, dtype=float))
        d = rotor.geometry.diameter
        ops, ws = [], []
        for r in rpm:
            n = r / 60.0
            for j in advance_ratios:
                for ang in inflow_angles_deg:
                    ca = max(np.cos(np.deg2rad(ang)), 1e-6)
                    v = j * n * d / ca if j > 0 else 0.0
                    if j == 0.0 and ang != inflow_angles_deg[0]:
                        continue
                    op = OperatingPoint(rpm=float(r), v_inf=float(v),
                                        inflow_angle_deg=float(ang),
                                        atmosphere=atmosphere)
                    ops.append(op)
                    ws.append(reference.solve(rotor, op).wrench)
                    if progress:  # pragma: no cover
                        print(f"  fit {len(ops)}", flush=True)
        kwargs.setdefault("source", f"fitted to {reference.name}")
        return cls.from_conditions(rotor, ops, ws, **kwargs)

    # ------------------------------------------------------------ 診断
    def training_error(self, rotor: Rotor) -> dict[str, float]:
        """学習データ上での再現誤差 (RMS, 係数の絶対値)."""
        sp = self.samples
        x = sp.features(self._mask)
        pred = np.stack([self._predict(row) for row in x])
        err = pred - sp.coeffs
        return {
            name: float(np.sqrt(np.mean(err[:, i] ** 2)))
            for i, name in enumerate(_COEFF_NAMES)
        }

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        sp = self.samples
        return (
            f"SurrogateModel(method={self.method}, degree={self.degree}, "
            f"n={0 if sp is None else sp.n_sample}, "
            f"J={self._range.get('j')}, mu_e={self._range.get('mu_e')}, "
            f"source={self.source})"
        )


def _momentum_inflow(
    thrust: float, rho: float, area: float, v_axial: float, v_edge: float
) -> float:
    """参考用の一様誘導速度 (運動量理論)."""
    v = np.sqrt(abs(thrust) / (2.0 * rho * area)) * np.sign(thrust)
    for _ in range(30):
        s = max(float(np.hypot(v_edge, v_axial + v)), 1e-6)
        f = 2.0 * rho * area * v * s - thrust
        df = 2.0 * rho * area * (s + v * (v_axial + v) / s)
        v -= f / max(abs(df), 1e-9) * np.sign(df) if df != 0 else 0.0
    return float(v)
