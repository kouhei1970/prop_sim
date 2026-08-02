"""例 4: アンバランス診断 (次数分析).

6 分力計測の実務でよく使う「1/rev と B/rev を分離して原因を切り分ける」
手順を再現する.

    質量アンバランス   -> 面内力 Fx, Fy の 1/rev (振幅 = U * Omega^2)
    空力アンバランス   -> ハブモーメント Mx, My の 1/rev
    斜め流入           -> ハブモーメントの定常値 + B/rev
    正常なブレード     -> B/rev のみ

    python examples/04_imbalance_diagnostics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照


import numpy as np

import prop_sim as ps
from prop_sim.experiments import dynamic_run
from prop_sim.postproc import harmonic_amplitudes, order_spectrum

OUT = Path(__file__).resolve().parent.parent / "results"
RPM = 6000.0
ORDERS = [1, 2, 3, 4]


def analyse(name: str, rotor: ps.Rotor, **kwargs) -> dict:
    res = dynamic_run(
        rotor, ps.BEMT(), duration=0.08, rpm=RPM, sim_rate=20000.0, **kwargs
    )
    psi = np.deg2rad(res.states["psi_deg"])
    out = {"name": name, "result": res, "psi": psi}
    print(f"\n[{name}]")
    print(f"{'':>4} {'mean':>11} " + " ".join(f"{o}P".rjust(11) for o in ORDERS))
    for i, c in enumerate(ps.COMPONENT_NAMES):
        fit = harmonic_amplitudes(res.hub_wrench[:, i], psi, orders=ORDERS)
        out[c] = fit
        print(
            f"{c:>4} {fit.mean:11.5f} "
            + " ".join(f"{a:11.5f}" for a in fit.amplitude)
        )
    return out


def main() -> None:
    geo = ps.from_diameter_pitch(10.0, 4.7)
    omega = RPM * 2 * np.pi / 60.0
    unbalance = 1.5e-5  # [kg m] = 15 mg at 1 m ~ 0.15 g at 100 mm

    cases = [
        analyse("baseline (健全)", ps.Rotor(geo)),
        analyse(
            "mass unbalance (質量アンバランス)",
            ps.Rotor(geo, unbalance=ps.Unbalance(static=unbalance)),
        ),
        analyse(
            "aero unbalance (+1 deg on one blade)",
            ps.Rotor(geo, blade_pitch_offsets_deg=[0.0, 1.0]),
        ),
        analyse("oblique inflow 45 deg", ps.Rotor(geo),
                v_inf=10.0, inflow_angle_deg=45.0),
    ]

    print(f"\n質量アンバランスの理論値 U*Omega^2 = {unbalance * omega**2:.3f} N")
    print(f"  Fx の 1/rev 実測 = {cases[1]['Fx'].amplitude[0]:.3f} N")

    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, case in zip(axes.ravel(), cases):
        for i, c in ((0, "Fx"), (3, "Mx")):
            o, a = order_spectrum(case["psi"], case["result"].hub_wrench[:, i],
                                  n_order=6)
            ax.semilogy(o, np.maximum(a, 1e-9), label=c)
        ax.set_title(case["name"], fontsize=10)
        ax.set_xlabel("order [/rev]")
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"order spectra @ {RPM:.0f} rpm (B=2)")
    fig.savefig(OUT / "04_imbalance_orders.png", dpi=130)
    print(f"-> {OUT / '04_imbalance_orders.png'}")


if __name__ == "__main__":
    main()
