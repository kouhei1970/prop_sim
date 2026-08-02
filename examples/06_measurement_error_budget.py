"""例 6: 計測系の誤差要因を切り分ける.

同じ「真値」に対して 6 分力計の仕様だけを変え, どの要因が
どの成分にどれだけ効くかを定量化する. 試験計画 (センサ定格の選定,
サンプリング周波数, 治具の固有振動数) の検討に使える.

    python examples/06_measurement_error_budget.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照


import numpy as np

import prop_sim as ps
from prop_sim.experiments import dynamic_run
from prop_sim.sensor import LoadCell, SensorConfig, TestRig

OUT = Path(__file__).resolve().parent.parent / "results"
RPM = 7000.0
FS = np.array([20.0, 20.0, 50.0, 1.0, 1.0, 1.0])


def base_config(**kwargs) -> SensorConfig:
    cfg = dict(
        full_scale=FS,
        crosstalk=0.0,
        calibration_residual=0.0,
        gain_error=0.0,
        noise_rms=0.0,
        natural_frequency=np.full(6, 1e5),   # 十分に硬い = 剛体
        anti_alias_hz=None,
        sample_rate=5000.0,
        adc_bits=0,
        seed=11,
    )
    cfg.update(kwargs)
    return SensorConfig(**cfg)


def main() -> None:
    geo = ps.from_diameter_pitch(10.0, 4.7)
    rotor = ps.Rotor(geo, unbalance=ps.Unbalance(static=8e-6))
    model = ps.BEMT()

    # まず真値の時系列を 1 回だけ作る
    truth = dynamic_run(
        rotor, model, duration=0.12, rpm=RPM, v_inf=8.0, inflow_angle_deg=30.0,
        rig=TestRig(sensor_offset=np.array([0.0, 0.0, -0.12])),
        sim_rate=40000.0, measure=False,
    )
    t = truth.time
    w = truth.sensor_wrench
    bpf = rotor.n_blades * RPM / 60.0
    print(f"真値: Fz={w[:,2].mean():.3f} N, 1/rev={RPM/60:.0f} Hz, "
          f"BPF={bpf:.0f} Hz")

    cases = {
        "ideal": base_config(),
        "crosstalk 2% (30% residual)": base_config(
            crosstalk=0.02, calibration_residual=0.3
        ),
        "white noise 0.05%FS": base_config(noise_rms=5e-4 * FS),
        "gain error 0.5%": base_config(gain_error=0.005),
        "fixture resonance 350 Hz": base_config(
            natural_frequency=np.full(6, 350.0), damping_ratio=0.02
        ),
        "12 bit ADC": base_config(adc_bits=12),
        "fs=300 Hz, no AAF (aliasing)": base_config(
            sample_rate=300.0, anti_alias_hz=None
        ),
        "fs=300 Hz + AAF 120 Hz": base_config(
            sample_rate=300.0, anti_alias_hz=120.0
        ),
    }

    print(f"\n{'case':<34}" + "".join(f"{c:>10}" for c in ps.COMPONENT_NAMES))
    print(f"{'':<34}" + "".join(f"{'err RMS':>10}" for _ in range(6)))
    rows = {}
    for name, cfg in cases.items():
        res = LoadCell(cfg).measure_series(t, w)
        err = res.measured - res.truth
        skip = err.shape[0] // 10           # 過渡を除く
        rms = np.sqrt(np.mean(err[skip:] ** 2, axis=0))
        rows[name] = rms
        print(f"{name:<34}" + "".join(f"{v:10.2e}" for v in rms))

    # --- エイリアシングは「点ごとの誤差」には出ない. スペクトルで見る.
    from prop_sim.postproc import spectrum

    print(f"\nFz スペクトルの最大ピーク (真の変動成分は BPF = {bpf:.0f} Hz)")
    for name in ("ideal", "fs=300 Hz, no AAF (aliasing)", "fs=300 Hz + AAF 120 Hz"):
        res = LoadCell(cases[name]).measure_series(t, w)
        f, a = spectrum(res.time, res.measured[:, 2])
        k = int(np.argmax(a))
        print(f"  {name:<32} {f[k]:6.0f} Hz  振幅 {a[k]:.4f} N"
              f"   (fs/2 = {cases[name].sample_rate/2:.0f} Hz)")
    print(f"  -> fs=300 Hz では BPF {bpf:.0f} Hz が {300 - bpf:.0f} Hz に"
          "折り返す (存在しない低周波成分に見える)")

    print("\n読み方")
    print("  * 共振は 1/rev や BPF が共振点に近いと計測値を大きく歪める")
    print("  * エイリアシングは 1 点ごとの誤差には現れず, BPF が折り返して")
    print("    存在しない低周波のピークとしてスペクトルに現れる")
    print("  * 感度誤差はスケール誤差なので大きな成分 (Fz) で目立つ")

    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(rows)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    width = 0.13
    for i, c in enumerate(ps.COMPONENT_NAMES):
        ax.bar(x + (i - 2.5) * width, [max(rows[n][i], 1e-12) for n in names],
               width, label=c)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("error RMS [N or N m]")
    ax.legend(ncols=6, fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("measurement error budget")
    fig.savefig(OUT / "06_error_budget.png", dpi=130)
    print(f"\n-> {OUT / '06_error_budget.png'}")


if __name__ == "__main__":
    main()
