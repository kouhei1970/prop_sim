"""例 2: 風洞試験 (前進率 x 流入角のマップ).

6 分力計測がもっとも意味を持つのは斜め流入 (プロペラ軸が気流に対して
傾いた状態). 推力・トルクだけでなく面内力 (法線力) とハブモーメントが
現れ, これが機体のピッチ/ヨー特性に効いてくる.

    python examples/02_wind_tunnel_map.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照


import numpy as np

import prop_sim as ps
from prop_sim.experiments import static_sweep

OUT = Path(__file__).resolve().parent.parent / "results"
RPM = 7000.0


def main() -> None:
    geo = ps.from_diameter_pitch(10.0, 4.7)
    rotor = ps.Rotor(geo)
    model = ps.BEMT()

    n = RPM / 60.0
    j_grid = np.arange(0.0, 0.71, 0.05)
    angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0])

    # 流入角 alpha に対し, 軸方向成分が J になるよう速度を決める
    rows = []
    for ang in angles:
        for j in j_grid:
            v = j * n * geo.diameter / max(np.cos(np.deg2rad(ang)), 1e-6)
            if v > 45.0:
                continue
            rows.append((ang, j, v))

    cols: dict[str, list[float]] = {}
    for ang, j, v in rows:
        r = static_sweep(
            rotor, model, rpm=RPM, v_inf=v, inflow_angle_deg=ang,
        )
        cols.setdefault("inflow_angle_deg", []).append(ang)
        cols.setdefault("J_axial", []).append(j)
        cols.setdefault("v_inf", []).append(v)
        for k in ("Fx_true", "Fy_true", "Fz_true", "Mx_true", "My_true",
                  "Mz_true", "Ct", "Cp", "eta"):
            cols.setdefault(k, []).append(float(r[k][0]))

    data = {k: np.asarray(v) for k, v in cols.items()}
    OUT.mkdir(exist_ok=True)
    from prop_sim.io import write_csv

    write_csv(OUT / "02_wind_tunnel_map.csv", data)
    print(f"-> {OUT / '02_wind_tunnel_map.csv'}  ({len(rows)} 点)")

    print(f"\n{RPM:.0f} rpm, 流入角ごとの代表値 (J_axial = 0.30)")
    print(f"{'alpha':>6} {'Fz[N]':>8} {'Fx[N]':>8} {'My[Nmm]':>9} "
          f"{'Mx[Nmm]':>9} {'eta':>6}")
    for ang in angles:
        m = (data["inflow_angle_deg"] == ang) & (np.abs(data["J_axial"] - 0.30) < 1e-9)
        if not np.any(m):
            continue
        i = int(np.argmax(m))
        print(
            f"{ang:6.0f} {data['Fz_true'][i]:8.3f} {data['Fx_true'][i]:8.3f} "
            f"{data['My_true'][i]*1e3:9.2f} {data['Mx_true'][i]*1e3:9.2f} "
            f"{data['eta'][i]:6.3f}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ang in angles:
        m = data["inflow_angle_deg"] == ang
        lab = f"{ang:.0f} deg"
        ax[0, 0].plot(data["J_axial"][m], data["Fz_true"][m], "o-", label=lab)
        ax[0, 1].plot(data["J_axial"][m], data["Fx_true"][m], "o-", label=lab)
        ax[1, 0].plot(data["J_axial"][m], data["My_true"][m] * 1e3, "o-", label=lab)
        ax[1, 1].plot(data["J_axial"][m], data["Mx_true"][m] * 1e3, "o-", label=lab)
    for a, t in zip(ax.ravel(), ["Fz (thrust) [N]", "Fx (in-plane) [N]",
                                 "My [N mm]", "Mx [N mm]"]):
        a.set_xlabel("J (axial)")
        a.set_ylabel(t)
        a.grid(alpha=0.3)
    ax[0, 0].legend(title="inflow angle", fontsize=8)
    fig.suptitle(f"wind tunnel map — {geo.name} @ {RPM:.0f} rpm")
    fig.savefig(OUT / "02_wind_tunnel_map.png", dpi=130)
    print(f"-> {OUT / '02_wind_tunnel_map.png'}")


if __name__ == "__main__":
    main()
