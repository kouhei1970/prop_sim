"""例 3: スロットルステップ応答 (動的データ).

見どころ

    * 回転数の 1 次遅れ的な立ち上がり (モータ + プロペラ慣性)
    * 誘導速度の遅れによる推力オーバーシュート (動的インフロー)
    * 加速中に -J dOmega/dt が計測トルクに乗る (慣性反トルク)
    * 6 分力計の構造共振がステップに励起される

    python examples/03_dynamic_throttle_step.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照


import numpy as np

import prop_sim as ps
from prop_sim.drivetrain import Drivetrain, Motor
from prop_sim.experiments import dynamic_run
from prop_sim.postproc import spectrum
from prop_sim.sensor import LoadCell, SensorConfig, TestRig

OUT = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    geo = ps.from_diameter_pitch(10.0, 4.7)
    rotor = ps.Rotor(geo)
    model = ps.BEMT()
    train = Drivetrain(
        motor=Motor(kv=920.0, resistance=0.11, no_load_current=0.5),
        prop_inertia=geo.polar_inertia,
    )
    rig = TestRig(
        sensor_offset=np.array([0.0, 0.0, -0.12]),
        load_cell=LoadCell(
            SensorConfig(
                full_scale=np.array([20.0, 20.0, 50.0, 1.0, 1.0, 1.0]),
                sample_rate=5000.0,
                natural_frequency=np.array([700, 700, 1100, 900, 900, 1400.0]),
                damping_ratio=0.015,
                anti_alias_hz=1500.0,
                seed=7,
            )
        ),
    )
    print(rotor.summary())
    print(train.summary())

    def throttle(t: float) -> float:
        return 0.35 if t < 0.05 else 0.75

    res = dynamic_run(
        rotor, model,
        duration=0.6,
        throttle=throttle,
        drivetrain=train,
        initial_rpm=4200.0,
        rig=rig,
        sim_rate=20000.0,
        progress=True,
    )

    t = res.time
    rpm = res.states["rpm"]
    thrust = res.states["thrust"]
    i_step = int(np.argmax(t >= 0.05))
    print(f"\n回転数 {rpm[0]:.0f} -> {rpm[-1]:.0f} rpm")
    print(f"推力   {thrust[:i_step].mean():.3f} -> {thrust[-200:].mean():.3f} N")

    # 各時刻の回転数における「定常解」と比べると, 誘導速度の遅れによる
    # 推力の行き過ぎがはっきり見える
    idx = np.arange(0, t.size, max(t.size // 240, 1))
    quasi = np.array([
        model.solve(rotor, ps.OperatingPoint(rpm=float(rpm[i]))).thrust for i in idx
    ])
    excess = thrust[idx] / np.maximum(quasi, 1e-9) - 1.0
    print(f"準定常解に対する推力の行き過ぎ: 最大 {100*excess.max():.1f} % "
          f"(t = {t[idx][int(np.argmax(excess))]*1e3:.0f} ms)")
    q_inertial = -res.hub_wrench[:, 5] - res.states["torque_aero"]
    print(f"慣性反トルクのピーク: {q_inertial.max()*1e3:.2f} mN m "
          f"(空力トルク {res.states['torque_aero'].max()*1e3:.1f} mN m)")

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "03_throttle_step.csv")
    print(f"-> {OUT / '03_throttle_step.csv'}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = res.measurement
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True,
                           constrained_layout=True)
    ax[0].plot(t, rpm)
    ax[0].set_ylabel("rpm")
    ax[1].plot(m.time, m.measured[:, 2], lw=0.6, color="tab:orange",
               label="measured")
    ax[1].plot(t, res.sensor_wrench[:, 2], color="tab:blue", label="true")
    ax[1].plot(t[idx], quasi, "--", color="k", lw=1.0, label="quasi-steady")
    ax[1].set_ylabel("Fz [N]")
    ax[1].legend(fontsize=8)
    ax[2].plot(m.time, m.measured[:, 5] * 1e3, lw=0.6, color="tab:orange")
    ax[2].plot(t, res.sensor_wrench[:, 5] * 1e3, color="tab:blue")
    ax[2].set_ylabel("Mz [N mm]")
    ax[3].plot(t, res.states["v_induced_mean"])
    ax[3].set_ylabel("induced velocity [m/s]")
    ax[3].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("throttle step response")
    fig.savefig(OUT / "03_throttle_step.png", dpi=130)

    f, a_ = spectrum(m.time, m.measured[:, 2])
    fig2, ax2 = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax2.semilogy(f, np.maximum(a_, 1e-8), lw=0.8)
    for k in (1, 2, 4):
        ax2.axvline(k * rpm[-1] / 60 * rotor.n_blades / 2, ls="--",
                    color="gray", lw=0.8)
    ax2.axvline(1100.0, ls=":", color="red", lw=1.0)
    ax2.text(1100.0, a_.max(), " load cell resonance", color="red", fontsize=8)
    ax2.set_xlabel("frequency [Hz]"); ax2.set_ylabel("Fz amplitude [N]")
    ax2.grid(alpha=0.3, which="both")
    fig2.savefig(OUT / "03_throttle_step_spectrum.png", dpi=130)
    print(f"-> {OUT / '03_throttle_step.png'}")


if __name__ == "__main__":
    main()
