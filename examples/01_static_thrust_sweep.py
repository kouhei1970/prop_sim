"""例 1: 静止推力試験 (回転数掃引).

もっとも基本的な試験. 回転数を階段状に変えて 6 分力・回転数・電流を
記録し, Ct / Cp / ホバー効率 (FM) を求める.

    python examples/01_static_thrust_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照


import numpy as np

import prop_sim as ps
from prop_sim.drivetrain import Drivetrain
from prop_sim.experiments import static_sweep
from prop_sim.sensor import LoadCell, SensorConfig, TestRig

OUT = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    # --- 供試体
    geo = ps.from_diameter_pitch(10.0, 4.7, n_blades=2)
    rotor = ps.Rotor(geo)
    model = ps.BEMT()
    print(rotor.summary())

    # --- 試験スタンド (ハブの 120 mm 下に 6 分力計)
    rig = TestRig(
        sensor_offset=np.array([0.0, 0.0, -0.12]),
        load_cell=LoadCell(
            SensorConfig(
                full_scale=np.array([20.0, 20.0, 50.0, 1.0, 1.0, 1.0]),
                sample_rate=2000.0,
                crosstalk=0.015,
                calibration_residual=0.2,
                seed=1,
            )
        ),
    )

    res = static_sweep(
        rotor, model,
        rpm=np.arange(2000.0, 10001.0, 500.0),
        rig=rig,
        drivetrain=Drivetrain(prop_inertia=geo.polar_inertia),
        n_average=2000,
    )

    print(f"\n{'rpm':>6} {'Fz[N]':>8} {'Mz[Nmm]':>9} {'Ct':>7} {'Cp':>7} "
          f"{'FM':>6} {'g/W':>7} {'I[A]':>6}")
    for i in range(res.n_points):
        print(
            f"{res['rpm'][i]:6.0f} {res['Fz'][i]:8.3f} "
            f"{res['Mz'][i]*1e3:9.2f} {res['Ct'][i]:7.4f} {res['Cp'][i]:7.4f} "
            f"{res['FM'][i]:6.3f} "
            f"{res['thrust'][i]/9.80665*1e3/res['power_electrical'][i]:7.2f} "
            f"{res['current'][i]:6.2f}"
        )

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "01_static_sweep.csv")
    print(f"\n-> {OUT / '01_static_sweep.csv'}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    ax[0].plot(res["rpm"], res["Fz"], "o-")
    ax[0].set_xlabel("rpm"); ax[0].set_ylabel("thrust Fz [N]")
    ax[1].plot(res["rpm"], -res["Mz"], "o-")
    ax[1].set_xlabel("rpm"); ax[1].set_ylabel("torque -Mz [N m]")
    ax[2].plot(res["rpm"], res["Ct"], "o-", label="Ct")
    ax[2].plot(res["rpm"], res["Cp"], "s-", label="Cp")
    ax[2].plot(res["rpm"], res["FM"], "^-", label="FM")
    ax[2].set_xlabel("rpm"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"static test — {geo.name}")
    fig.savefig(OUT / "01_static_sweep.png", dpi=130)
    print(f"-> {OUT / '01_static_sweep.png'}")


if __name__ == "__main__":
    main()
