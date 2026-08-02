"""例 5: 空力モデルの比較と代理モデルの同定.

BEMT (既定) / BET (一様インフロー) / 二次式代理モデルを同じ条件で
走らせて差と計算コストを比べる. 代理モデルの係数は BEMT の結果から
最小二乗で同定する.

    python examples/05_model_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import time

import numpy as np

import prop_sim as ps

OUT = Path(__file__).resolve().parent.parent / "results"
RPM = 7000.0


def main() -> None:
    geo = ps.from_diameter_pitch(10.0, 4.7)
    rotor = ps.Rotor(geo)
    n = RPM / 60.0
    d = geo.diameter
    j_grid = np.linspace(0.0, 0.7, 15)

    bemt = ps.BEMT()
    bet = ps.BET()

    print("代理モデルを BEMT から同定中 ...")
    t0 = time.perf_counter()
    surrogate = ps.QuadraticModel.fit(
        rotor, bemt, rpm=RPM, advance_ratios=np.linspace(0.0, 0.7, 8),
        inflow_angles_deg=np.array([0.0, 20.0, 40.0]),
    )
    print(f"  同定 {time.perf_counter() - t0:.2f} s")
    print(f"  Ct(J) = {surrogate.ct[0]:+.5f} {surrogate.ct[1]:+.5f} J "
          f"{surrogate.ct[2]:+.5f} J^2")
    print(f"  Cq(J) = {surrogate.cq[0]:+.6f} {surrogate.cq[1]:+.6f} J "
          f"{surrogate.cq[2]:+.6f} J^2")

    models = {"BEMT": bemt, "BET": bet, "quadratic": surrogate}
    thrust = {k: [] for k in models}
    torque = {k: [] for k in models}
    cost = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        for j in j_grid:
            op = ps.OperatingPoint(rpm=RPM, v_inf=j * n * d)
            sol = m.solve(rotor, op)
            thrust[name].append(sol.thrust)
            torque[name].append(sol.torque(rotor.spin))
        cost[name] = (time.perf_counter() - t0) / j_grid.size

    print(f"\n{'J':>6} " + " ".join(f"{k+' T[N]':>12}" for k in models))
    for i, j in enumerate(j_grid):
        print(f"{j:6.2f} " + " ".join(f"{thrust[k][i]:12.4f}" for k in models))
    print("\n1 点あたりの計算時間")
    for k, v in cost.items():
        print(f"  {k:10s} {v*1e3:8.3f} ms")

    # 斜め流入での比較 (代理モデルは面内 1 次項まで)
    print("\n斜め流入 (J=0.3, alpha=30 deg) の 6 分力")
    v = 0.3 * n * d / np.cos(np.deg2rad(30.0))
    op = ps.OperatingPoint(rpm=RPM, v_inf=v, inflow_angle_deg=30.0)
    print(f"{'model':>10} " + " ".join(f"{c:>9}" for c in ps.COMPONENT_NAMES))
    for name, m in models.items():
        w = m.solve(rotor, op).wrench.as_array()
        print(f"{name:>10} " + " ".join(f"{x:9.4f}" for x in w))

    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for k in models:
        ax[0].plot(j_grid, thrust[k], "o-", ms=3, label=k)
        ax[1].plot(j_grid, torque[k], "o-", ms=3, label=k)
    ax[0].set_xlabel("J"); ax[0].set_ylabel("thrust [N]")
    ax[1].set_xlabel("J"); ax[1].set_ylabel("torque [N m]")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"model comparison — {geo.name} @ {RPM:.0f} rpm")
    fig.savefig(OUT / "05_model_comparison.png", dpi=130)
    print(f"\n-> {OUT / '05_model_comparison.png'}")


if __name__ == "__main__":
    main()
