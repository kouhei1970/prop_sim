"""6 分力計 (ロードセル) と試験スタンドのモデル.

実験シミュレータの肝は「真値」ではなく「計測値」を再現することにある.
本モジュールは 6 分力計測でよく問題になる要因を一通り模擬する.

    * 作用点のずれ (ハブ中心 <-> センサ原点) によるモーメント換算
    * 軸間干渉 (クロストーク) と校正行列の残差
    * 感度誤差 / オフセット / 温度ドリフト
    * 構造共振 (センサ + 治具の 2 次系) — 動的計測では支配的
    * 白色ノイズ + ランダムウォーク (1/f) ノイズ
    * 抗エイリアスフィルタ, サンプリング, A/D 量子化と飽和
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .frames import COMPONENT_NAMES, Wrench

__all__ = ["LoadCell", "TestRig", "SensorConfig", "MeasurementResult"]


@dataclass
class SensorConfig:
    """6 分力計の仕様.

    既定値は小型 6 軸ロードセル (ATI Mini45 クラス) を想定した代表値.

    Parameters
    ----------
    full_scale:
        各チャネルの定格 [Fx, Fy, Fz (N), Mx, My, Mz (N m)].
    crosstalk:
        軸間干渉行列 (6x6, 対角は 0). スカラーを与えると
        非対角成分にその値 (定格比) をランダムに配分する.
    calibration_residual:
        校正で除去しきれない干渉の割合 [0-1]. 0 なら完全に補正できる.
    gain_error:
        感度誤差 (各チャネル, 比率). 校正後に残るスケール誤差として扱う.
    bias:
        オフセット [各チャネル物理単位]. ゼロ点補正後の残差.
    noise_rms:
        白色ノイズ RMS [各チャネル物理単位]. None なら定格の 0.05%.
    random_walk:
        ランダムウォークノイズ強度 [単位/sqrt(s)]. ゼロ点ドリフトの原因.
    thermal_drift:
        温度ドリフト [単位/s] (試験中の単調ドリフト).
    natural_frequency:
        構造共振周波数 [Hz] (各チャネル).
    damping_ratio:
        減衰比.
    sample_rate:
        A/D サンプリング周波数 [Hz].
    adc_bits:
        A/D 分解能 [bit] (両極性, 定格の +-1 倍を範囲とする).
    anti_alias_hz:
        抗エイリアスフィルタのカットオフ [Hz]. None ならフィルタ無し
        (エイリアシングが再現される).
    anti_alias_order:
        フィルタ次数 (Butterworth).
    """

    full_scale: np.ndarray = field(
        default_factory=lambda: np.array([145.0, 145.0, 290.0, 5.0, 5.0, 5.0])
    )
    crosstalk: np.ndarray | float = 0.015
    calibration_residual: float = 0.15
    gain_error: np.ndarray | float = 0.002
    bias: np.ndarray | float = 0.0
    noise_rms: np.ndarray | float | None = None
    random_walk: np.ndarray | float = 0.0
    thermal_drift: np.ndarray | float = 0.0
    natural_frequency: np.ndarray | float = field(
        default_factory=lambda: np.array([900.0, 900.0, 1500.0, 1200.0, 1200.0, 1800.0])
    )
    damping_ratio: np.ndarray | float = 0.02
    sample_rate: float = 10000.0
    adc_bits: int = 16
    anti_alias_hz: float | None = 2000.0
    anti_alias_order: int = 4
    seed: int | None = 0

    def __post_init__(self) -> None:
        self.full_scale = np.asarray(self.full_scale, dtype=float).reshape(6)
        self.natural_frequency = _as6(self.natural_frequency)
        self.damping_ratio = _as6(self.damping_ratio)
        self.gain_error = _as6(self.gain_error)
        self.bias = _as6(self.bias)
        self.random_walk = _as6(self.random_walk)
        self.thermal_drift = _as6(self.thermal_drift)
        if self.noise_rms is None:
            self.noise_rms = 5.0e-4 * self.full_scale
        else:
            self.noise_rms = _as6(self.noise_rms)


def _as6(v) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(v, dtype=float))
    return np.full(6, float(arr[0])) if arr.size == 1 else arr.reshape(6)


@dataclass
class MeasurementResult:
    """計測結果 (真値と計測値の対)."""

    time: np.ndarray            # (n,)   サンプリング後の時刻 [s]
    measured: np.ndarray        # (n, 6) 計測値 (校正後, 物理単位)
    truth: np.ndarray           # (n, 6) センサ原点における真値
    time_raw: np.ndarray        # (n_sim,) シミュレーション時刻
    truth_raw: np.ndarray       # (n_sim, 6)

    def as_dict(self) -> dict[str, np.ndarray]:
        out = {"time": self.time}
        for i, n in enumerate(COMPONENT_NAMES):
            out[n] = self.measured[:, i]
            out[n + "_true"] = self.truth[:, i]
        return out

    def error(self) -> np.ndarray:
        return self.measured - self.truth


class LoadCell:
    """6 分力計のシミュレーション."""

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        self._rng = rng

        # --- 真の応答行列 K_true : 物理量 -> 電圧 (定格で正規化した単位系)
        if np.isscalar(cfg.crosstalk):
            level = float(cfg.crosstalk)
            ct = rng.normal(0.0, level, (6, 6))
            np.fill_diagonal(ct, 0.0)
        else:
            ct = np.asarray(cfg.crosstalk, dtype=float).reshape(6, 6).copy()
            np.fill_diagonal(ct, 0.0)
        fs = cfg.full_scale
        # 干渉は「定格比」で定義されるのでスケーリングして物理単位に直す
        k = np.eye(6) * (1.0 + cfg.gain_error) + ct * (fs[:, None] / fs[None, :])
        self.k_true = k
        # 校正行列: 干渉の一部だけを補正できたものとする.
        # 感度誤差 (gain_error) は「校正後に残るスケール誤差」なので
        # 校正行列には含めない -> そのまま計測値のスケール誤差になる.
        self.k_nominal = np.eye(6) + (
            ct * (1.0 - cfg.calibration_residual) * (fs[:, None] / fs[None, :])
        )
        self.k_decouple = np.linalg.inv(self.k_nominal)

    # ------------------------------------------------------------ 静的計測
    def measure_static(
        self, wrench: np.ndarray, n_average: int = 1
    ) -> np.ndarray:
        """定常値 1 点の計測 (構造動特性は無視, ノイズは平均化).

        Parameters
        ----------
        wrench:
            ``(..., 6)`` のセンサ原点における真の 6 分力.
        n_average:
            平均化サンプル数 (白色ノイズが 1/sqrt(n) に減る).
        """
        cfg = self.config
        w = np.asarray(wrench, dtype=float)
        raw = w @ self.k_true.T + cfg.bias
        noise = self._rng.normal(
            0.0, np.asarray(cfg.noise_rms) / np.sqrt(max(n_average, 1)), w.shape
        )
        raw = raw + noise
        raw = self._quantize(np.clip(raw, -cfg.full_scale, cfg.full_scale))
        return raw @ self.k_decouple.T

    # ------------------------------------------------------------ 動的計測
    def measure_series(
        self, time: np.ndarray, wrench: np.ndarray
    ) -> MeasurementResult:
        """時系列の計測 (構造共振 -> ノイズ -> AAF -> サンプリング -> A/D).

        Parameters
        ----------
        time:
            等間隔のシミュレーション時刻 [s].
        wrench:
            ``(n, 6)`` のセンサ原点における真の 6 分力.
        """
        cfg = self.config
        t = np.asarray(time, dtype=float)
        w = np.asarray(wrench, dtype=float)
        if w.shape[0] != t.size:
            raise ValueError("time と wrench の長さが一致しません")
        dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0 / cfg.sample_rate
        fs_sim = 1.0 / dt

        # 1) 構造応答 (2 次系)
        y = np.empty_like(w)
        for i in range(6):
            wn = 2.0 * np.pi * cfg.natural_frequency[i]
            zeta = cfg.damping_ratio[i]
            if not np.isfinite(wn) or wn <= 0 or wn > np.pi * fs_sim:
                y[:, i] = w[:, i]   # 共振が Nyquist を超える -> 剛体とみなす
                continue
            num = [wn**2]
            den = [1.0, 2.0 * zeta * wn, wn**2]
            b, a = signal.bilinear(num, den, fs=fs_sim)
            y[:, i] = signal.lfilter(b, a, w[:, i], zi=None)

        # 2) 干渉・感度誤差
        raw = y @ self.k_true.T + cfg.bias

        # 3) ノイズ (白色 + ランダムウォーク + 温度ドリフト)
        raw = raw + self._rng.normal(0.0, cfg.noise_rms, raw.shape)
        if np.any(cfg.random_walk > 0.0):
            steps = self._rng.normal(0.0, 1.0, raw.shape) * (
                cfg.random_walk * np.sqrt(dt)
            )
            raw = raw + np.cumsum(steps, axis=0)
        if np.any(cfg.thermal_drift != 0.0):
            raw = raw + cfg.thermal_drift * (t - t[0])[:, None]

        # 4) 抗エイリアスフィルタ
        if cfg.anti_alias_hz is not None:
            wn = min(cfg.anti_alias_hz / (0.5 * fs_sim), 0.99)
            if 0.0 < wn < 1.0:
                b, a = signal.butter(cfg.anti_alias_order, wn, btype="low")
                raw = signal.filtfilt(b, a, raw, axis=0)

        # 5) サンプリング + A/D
        step = max(int(round(fs_sim / cfg.sample_rate)), 1)
        idx = np.arange(0, t.size, step)
        sampled = self._quantize(
            np.clip(raw[idx], -cfg.full_scale, cfg.full_scale)
        )
        measured = sampled @ self.k_decouple.T
        return MeasurementResult(
            time=t[idx], measured=measured, truth=w[idx],
            time_raw=t, truth_raw=w,
        )

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.adc_bits is None or cfg.adc_bits <= 0:
            return x
        lsb = 2.0 * cfg.full_scale / (2**cfg.adc_bits)
        return np.round(x / lsb) * lsb


@dataclass
class TestRig:
    """試験スタンド (取り付け幾何 + 6 分力計).

    Parameters
    ----------
    sensor_offset:
        ハブ中心からセンサ原点へのベクトル [m] (ハブ座標系).
        例: プロペラ面の 120 mm 下にセンサ原点がある場合 ``(0, 0, -0.12)``.
    misalignment_deg:
        センサ取り付けの角度誤差 (x, y, z 軸まわり) [deg].
    load_cell:
        6 分力計.
    tare_wrench:
        ゼロ点補正で差し引くレンチ (静止時の自重など). None なら自動で
        自重 ``mass * g`` を +z 方向に設定できる.
    """

    __test__ = False       # pytest にテストクラスと誤認されないように

    sensor_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.12])
    )
    misalignment_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    load_cell: LoadCell = field(default_factory=LoadCell)
    tare_wrench: Wrench | None = None

    def __post_init__(self) -> None:
        self.sensor_offset = np.asarray(self.sensor_offset, dtype=float).reshape(3)
        self.misalignment_deg = np.asarray(
            self.misalignment_deg, dtype=float
        ).reshape(3)
        a, b, c = np.deg2rad(self.misalignment_deg)
        # 微小角の近似 (センサ座標系 <- ハブ座標系)
        self.dcm = np.array(
            [[1.0, c, -b], [-c, 1.0, a], [b, -a, 1.0]]
        )

    def to_sensor(self, hub_wrench: Wrench) -> Wrench:
        """ハブ中心のレンチをセンサ原点・センサ座標系に変換する."""
        w = hub_wrench.translate(self.sensor_offset).rotate(self.dcm)
        if self.tare_wrench is not None:
            w = w - self.tare_wrench
        return w

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        o = self.sensor_offset
        cfg = self.load_cell.config
        return (
            f"sensor offset=({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}) m, "
            f"fs={cfg.sample_rate:.0f} Hz, {cfg.adc_bits} bit, "
            f"fn(Fz)={cfg.natural_frequency[2]:.0f} Hz"
        )
