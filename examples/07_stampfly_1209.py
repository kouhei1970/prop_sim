"""例 7: StampFly 1209 プロペラ (31 mm 4 枚) の性能推定.

平面形は実機写真からの実測 (``prop_sim.presets.stampfly_1209``).
ピッチだけは型番 1209 の公称 0.9 inch を仮定している.

出力

    1. 回転数掃引 (推力・トルク・動力・Ct/Cp/FM・Reynolds 数)
    2. ホバー点 (機体重量から必要回転数と動力)
    3. モデル仮定の感度 (翼型・ピッチ・取付角)
    4. 前進飛行の 6 分力
    5. 6 分力計測の要求仕様 (BPF, アンバランス, ジャイロ)

    python examples/07_stampfly_1209.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import numpy as np
from scipy.optimize import brentq

import prop_sim as ps
from prop_sim.experiments import static_sweep
from prop_sim.presets import STAMPFLY_1209_MEASURED

OUT = Path(__file__).resolve().parent.parent / "results"
G = 9.80665
MASS_G = 36.8          # 機体全備重量の想定 [g] (実機に合わせて変更すること)


def hover_rpm(rotor, model, thrust_gf: float) -> float:
    def f(n):
        return model.solve(rotor, ps.OperatingPoint(rpm=n)).thrust / G * 1e3 - thrust_gf

    return brentq(f, 2000.0, 90000.0, xtol=5.0)


def main() -> None:
    geo = ps.stampfly_1209()
    rotor = ps.Rotor(geo)
    model = ps.BEMT()

    print("=" * 78)
    print("写真からの実測値")
    for k, v in STAMPFLY_1209_MEASURED.items():
        print(f"  {k:24s} {v}")
    print(f"\n{geo.summary()}")
    print(f"  ソリディティ sigma = {geo.solidity:.3f}, "
          f"面積比 = {geo.blade_area()/geo.disk_area:.3f}")

    print("\n翼断面 (切断面写真からの実測 + 薄翼理論)")
    print(f"  {ps.STAMPFLY_1209_SECTION.thin_airfoil_properties().summary()}")
    foil = ps.stampfly_1209_airfoil()
    print(f"  -> 翼型モデル @Re=1.2e4: cl_alpha={foil.cl_alpha:.2f}/rad, "
          f"alpha0={foil.alpha0_deg:+.2f} deg, cl_max={foil.cl_max:.2f}, "
          f"cd0={foil.cd0:.4f}")
    aa = np.deg2rad(np.linspace(-5, 15, 400))
    cl, cd, _ = foil.coefficients(aa, np.full_like(aa, 1.2e4), None)
    k = int(np.argmax(cl / cd))
    print(f"  -> 最大揚抗比 {cl[k]/cd[k]:.1f} @ alpha = {np.rad2deg(aa[k]):.1f} deg "
          f"(cl = {cl[k]:.2f})")

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 78)
    print("1) 静止推力 — 回転数掃引")
    rpm = np.arange(5000.0, 40001.0, 2500.0)
    res = static_sweep(rotor, model, rpm=rpm, include_sections=True)
    print(f"{'rpm':>6}{'T[gf]':>8}{'Q[mNm]':>8}{'P[W]':>7}{'Ct':>8}{'Cp':>8}"
          f"{'FM':>7}{'Re@75%':>8}{'a75[d]':>7}{'gf/W':>7}")
    for i in range(res.n_points):
        print(f"{res['rpm'][i]:6.0f}{res['thrust'][i]/G*1e3:8.2f}"
              f"{res['torque'][i]*1e3:8.3f}{res['power_shaft'][i]:7.3f}"
              f"{res['Ct'][i]:8.4f}{res['Cp'][i]:8.4f}{res['FM'][i]:7.3f}"
              f"{res['re75'][i]:8.0f}{res['alpha75_deg'][i]:7.2f}"
              f"{res['thrust'][i]/G*1e3/res['power_shaft'][i]:7.2f}")
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "07_stampfly_static.csv")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 78)
    print("2) ホバー点 (4 発, 軸動力)")
    print(f"{'機体重量[g]':>12}{'1発[gf]':>10}{'rpm':>9}{'P/発[W]':>10}{'合計[W]':>10}")
    for wt in (28.0, 32.0, MASS_G, 40.0, 45.0):
        n = hover_rpm(rotor, model, wt / 4.0)
        op = ps.OperatingPoint(rpm=n)
        p = model.solve(rotor, op).power(op)
        print(f"{wt:12.1f}{wt/4:10.2f}{n:9.0f}{p:10.3f}{4*p:10.2f}")
    n_hov = hover_rpm(rotor, model, MASS_G / 4.0)

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 78)
    print(f"3) モデル仮定の感度 ({MASS_G:.1f} g 機体 = 1 発 {MASS_G/4:.2f} gf)")
    print(f"{'仮定':<34}{'hover rpm':>11}{'P/発[W]':>10}{'FM':>8}")
    variants = [
        ("既定 (実測断面, P=0.9in)", ps.stampfly_1209()),
        ("断面: キャンバ効率 0.70", ps.stampfly_1209(
            airfoil=ps.stampfly_1209_airfoil(camber_efficiency=0.70))),
        ("断面: キャンバ効率 1.00", ps.stampfly_1209(
            airfoil=ps.stampfly_1209_airfoil(camber_efficiency=1.00))),
        ("断面: 剥離泡ペナルティ 2.2", ps.stampfly_1209(
            airfoil=ps.stampfly_1209_airfoil(bubble_penalty=2.2))),
        ("断面: 投影補正 厚み x0.85", ps.stampfly_1209(
            airfoil=ps.airfoil_from_section(
                ps.stampfly_1209_section(thickness_scale=0.85), reynolds_ref=1.2e4))),
        ("翼型: 汎用の低Re薄翼 (断面不使用)", ps.stampfly_1209(airfoil=ps.LOW_RE_THIN)),
        ("翼型: 平板", ps.stampfly_1209(airfoil=ps.FLAT_PLATE)),
        ("翼型: 高Re相当 (Clark Y)", ps.stampfly_1209(airfoil=ps.CLARK_Y)),
        ("ピッチ P=0.8 in", ps.stampfly_1209(pitch_in=0.8)),
        ("ピッチ P=1.0 in", ps.stampfly_1209(pitch_in=1.0)),
        ("翼端 washout -2 deg", ps.stampfly_1209(pitch_distribution="washout")),
    ]
    for nm, g2 in variants:
        r2 = ps.Rotor(g2)
        n2 = hover_rpm(r2, model, MASS_G / 4.0)
        op2 = ps.OperatingPoint(rpm=n2)
        s2 = model.solve(r2, op2)
        print(f"{nm:<34}{n2:11.0f}{s2.power(op2):10.3f}"
              f"{s2.coefficients(r2, op2)['FM']:8.3f}")
    for dc in (-1.0, 1.0):
        r2 = ps.Rotor(geo, collective_deg=dc)
        print(f"{'取付角 ' + f'{dc:+.1f} deg':<34}"
              f"{hover_rpm(r2, model, MASS_G/4.0):11.0f}")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 78)
    print(f"4) 前進飛行の 6 分力 ({n_hov:.0f} rpm 固定)")
    print(f"{'V[m/s]':>7}{'流入角':>7}{'J':>7}{'Fz[gf]':>8}{'Fx[gf]':>8}"
          f"{'Mx[uNm]':>9}{'My[uNm]':>9}{'Q[uNm]':>8}")
    for v in (0.0, 2.0, 4.0, 6.0, 8.0):
        for ang in ((0.0,) if v == 0 else (0.0, 30.0, 60.0, 90.0)):
            op = ps.OperatingPoint(rpm=n_hov, v_inf=v, inflow_angle_deg=ang)
            sol = model.solve(rotor, op)
            w = sol.wrench
            print(f"{v:7.1f}{ang:7.0f}{op.advance_ratio(geo.diameter):7.3f}"
                  f"{w.force[2]/G*1e3:8.2f}{w.force[0]/G*1e3:8.3f}"
                  f"{w.moment[0]*1e6:9.1f}{w.moment[1]*1e6:9.1f}"
                  f"{sol.torque()*1e6:8.1f}")
    print("  * 流入角 90 deg (横風/横進) で推力が増えるのが並進揚力")
    print("  * 流入角 0 deg (上昇) では前進率とともに推力が落ちる")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 78)
    print("5) 6 分力計測の要求仕様 (ホバー点)")
    om = n_hov * 2 * np.pi / 60
    sol = model.solve(rotor, ps.OperatingPoint(rpm=n_hov))
    jz = geo.polar_inertia
    print(f"  1/rev = {n_hov/60:.0f} Hz,  BPF(4/rev) = {4*n_hov/60:.0f} Hz,  "
          f"8/rev = {8*n_hov/60:.0f} Hz")
    print(f"  推力 {sol.thrust*1e3:.1f} mN,  トルク {sol.torque()*1e6:.0f} uN m,  "
          f"誘導速度 {sol.inflow.mean_v0():.2f} m/s")
    print(f"  角運動量 Jz*Omega = {jz*om*1e6:.1f} uN m s")
    print("\n  ジャイロモーメント (機体角速度 q)")
    for q in (1.0, 5.0, 10.0, 20.0):
        mg = jz * om * q
        print(f"    q={q:5.1f} rad/s -> {mg*1e6:8.1f} uN m "
              f"(空力トルクの {mg/sol.torque()*100:5.0f} %)")
    print("\n  質量アンバランスの 1/rev 面内力 (U*Omega^2)")
    for e_um in (1.0, 5.0, 10.0, 30.0):
        f = geo.mass * e_um * 1e-6 * om**2
        print(f"    重心ずれ {e_um:5.1f} um -> {f*1e3:7.2f} mN "
              f"(推力の {f/sol.thrust*100:5.1f} %)")
    print("\n  -> 6 分力計 + 治具の 1 次共振は BPF の 3 倍 (>6 kHz) が理想.")
    print("     届かない場合は共振の伝達関数で逆補正するか, BPF より十分低い")
    print("     帯域だけを信用して定常値を評価する.")

    # ------------------------------------------------------------- グラフ
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    ax[0].plot(res["rpm"], res["thrust"] / G * 1e3, "o-")
    ax[0].axhline(MASS_G / 4, color="r", ls="--", lw=1)
    ax[0].text(res["rpm"][0], MASS_G / 4, f" hover ({MASS_G:.0f} g / 4)",
               color="r", va="bottom", fontsize=8)
    ax[0].set_xlabel("rpm"); ax[0].set_ylabel("thrust [gf]")
    ax[1].plot(res["rpm"], res["power_shaft"], "o-")
    ax[1].set_xlabel("rpm"); ax[1].set_ylabel("shaft power [W]")
    ax[2].plot(res["rpm"], res["FM"], "o-", label="FM")
    ax[2].plot(res["rpm"], res["Ct"], "s-", label="Ct")
    ax[2].plot(res["rpm"], res["Cp"], "^-", label="Cp")
    ax[2].set_xlabel("rpm"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("StampFly 1209 (31 mm, 4 blades) — BEMT")
    fig.savefig(OUT / "07_stampfly_static.png", dpi=130)

    x = np.linspace(geo.hub_radius_ratio, 1.0, 200)
    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax2[0].plot(x, geo.chord(x) * 1e3)
    ax2[0].set_xlabel("r/R"); ax2[0].set_ylabel("chord [mm]")
    ax2[0].set_title("measured planform")
    ax2[1].plot(x, np.rad2deg(geo.twist(x)), label="twist (assumed P=0.9in)")
    ax2[1].plot(ps.presets.STAMPFLY_1209_SWEEP_DEG[:, 0],
                ps.presets.STAMPFLY_1209_SWEEP_DEG[:, 1], "o--",
                label="measured sweep")
    ax2[1].set_xlabel("r/R"); ax2[1].set_ylabel("angle [deg]"); ax2[1].legend(fontsize=8)
    for a in ax2:
        a.grid(alpha=0.3)
    fig2.savefig(OUT / "07_stampfly_geometry.png", dpi=130)
    print(f"\n-> {OUT / '07_stampfly_static.png'}")


if __name__ == "__main__":
    main()
