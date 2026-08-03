"""例 9: 実測推力によるモデル較正と, 仮説の切り分け.

StampFly 1209 の推力試験で得られた 1 点

    Omega = 1209 rad/s (11,545 rpm) のとき 推力 1.00 gf

に対してモデルを合わせ込み, 「どのパラメータがずれているのか」を
切り分けられるかを調べる.

要点: **1 点だけでは切り分けられない**. 取付角がずれている仮説と
低 Reynolds 数で性能が落ちている仮説はどちらもその 1 点に完全に合うが,
ホバー回転数の予測は 23 % 食い違う.

そのため **プリセット ``stampfly_1209()`` には較正を適用していない**.
このスクリプトは較正機能の使い方と, 1 点しかないときの危険性を示す
デモであって, 較正済みモデルを作るものではない. 回転数を振った試験結果
が揃ったら ``MEASUREMENTS`` を差し替えて 2 パラメータ同時較正に進む.

    python examples/09_calibrate_to_measurement.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import numpy as np
from scipy.optimize import brentq

import prop_sim as ps

OUT = Path(__file__).resolve().parent.parent / "results"
G = 9.80665
MEASUREMENTS = [ps.ThrustMeasurement(omega=1209.0, thrust_gf=1.0)]
HOVER_GF = 9.2          # 36.8 g / 4 発


def hover_rpm(rotor, model, gf=HOVER_GF):
    return brentq(
        lambda n: model.solve(rotor, ps.OperatingPoint(rpm=n)).thrust / G * 1e3 - gf,
        3000.0, 90000.0, xtol=5.0,
    )


def main() -> None:
    rotor = ps.Rotor(ps.stampfly_1209())
    model = ps.BEMT()
    m0 = MEASUREMENTS[0]

    print("=" * 74)
    print("実測")
    print(f"  Omega = {m0.omega:.0f} rad/s = {m0.rpm:.0f} rpm,  T = {m0.thrust_gf:.2f} gf"
          f" = {m0.thrust_n*1e3:.3f} mN")
    n, d = m0.rpm / 60.0, rotor.geometry.diameter
    print(f"  Ct = {m0.thrust_n/(1.225*n**2*d**4):.4f}")

    op = m0.operating_point()
    sol = model.solve(rotor, op)
    st = sol.sections
    k75 = int(np.argmin(np.abs(st.r_R - 0.75)))
    print("\n較正前のモデル")
    print(f"  T = {sol.thrust/G*1e3:.3f} gf,  Ct = {sol.coefficients(rotor,op)['Ct']:.4f}"
          f"  -> 実測の {sol.thrust/m0.thrust_n:.2f} 倍")
    print(f"  Re@0.75R = {st.reynolds[:,k75].mean():.0f}, "
          f"Re@翼端 = {st.reynolds[:,-1].mean():.0f}, "
          f"alpha@0.75R = {np.rad2deg(st.alpha[:,k75].mean()):.2f} deg")
    print("  -> この作動点は Re が 2,000-5,000 しかない. 翼型モデルが"
          "もっとも当てにならない領域.")

    # ------------------------------------------------------------ 仮説比較
    print("\n" + "=" * 74)
    print("1 点に合う複数の仮説")
    rpms = np.array([5000.0, 11545.0, 20000.0, 27300.0, 35000.0])
    out = ps.compare_hypotheses(
        rotor, model, MEASUREMENTS, rpm_range=rpms,
        parameters=("collective", "re_lift", "camber"),
    )
    for k, r in out["results"].items():
        v = list(r.parameters.values())[0]
        print(f"  {k:<11} = {v:+.4f}   残差 {abs(r.residual_gf[0])*1e3:6.3f} mgf"
              f"   {'(合わせきれない)' if abs(r.residual_gf[0]) > 1e-3 else ''}")

    print(f"\n  {'rpm':>7}{'Omega':>8}" + "".join(f"{k:>12}" for k in out["curves"])
          + f"{'食い違い':>10}")
    for i, rp in enumerate(rpms):
        print(f"  {rp:7.0f}{rp*2*np.pi/60:8.0f}"
              + "".join(f"{out['curves'][k][i]:12.3f}" for k in out["curves"])
              + f"{out['spread_pct'][i]:9.0f}%")
    print("  (推力 gf)")

    print("\n" + "=" * 74)
    print(f"ホバー ({HOVER_GF:.1f} gf/発) の予測は仮説でどれだけ変わるか")
    print(f"  {'仮説':<12}{'rpm':>9}{'rad/s':>9}{'P/発[W]':>10}{'4発合計[W]':>12}")
    for k, r in out["results"].items():
        if abs(r.residual_gf[0]) > 1e-3:
            continue
        nh = hover_rpm(r.rotor, model)
        oph = ps.OperatingPoint(rpm=nh)
        p = model.solve(r.rotor, oph).power(oph)
        print(f"  {k:<12}{nh:9.0f}{nh*2*np.pi/60:9.0f}{p:10.3f}{4*p:12.2f}")
    print("\n  -> 1 点の較正では ホバー回転数が 23 % 不定.")
    print("     回転数を変えた点をあと 2-3 点測れば切り分けられる.")
    print("     それまでプリセットには較正を適用しない (不定性がモデルの中に")
    print("     隠れてしまうため).")

    # --------------------------------------------- 2 点あればどうなるか (実演)
    print("\n" + "=" * 74)
    print("参考: 仮に 2 点あった場合の同時較正 (ここでは re_lift 仮説から生成した")
    print("      疑似データで手順を実演. 実測が増えたらここを差し替えること)")
    truth = out["results"]["re_lift"].rotor
    fake = [
        ps.ThrustMeasurement(
            rpm=float(rp),
            thrust_gf=model.solve(truth, ps.OperatingPoint(rpm=rp)).thrust / G * 1e3)
        for rp in (11545.0, 22000.0, 32000.0)
    ]
    res2 = ps.calibrate(rotor, model, fake, parameters=("collective", "re_lift"))
    print("  " + res2.report().replace("\n", "\n  "))
    print(f"  -> 真値 collective=0.000, re_lift={out['results']['re_lift'].parameters['re_lift']:.4f}")

    # ------------------------------------------------------------- グラフ
    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rr = np.linspace(4000.0, 36000.0, 22)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for k, r in out["results"].items():
        if abs(r.residual_gf[0]) > 1e-3:
            continue
        t = [model.solve(r.rotor, ps.OperatingPoint(rpm=x)).thrust / G * 1e3
             for x in rr]
        ax.plot(rr, t, label=f"calibrated: {k}")
    t0 = [model.solve(rotor, ps.OperatingPoint(rpm=x)).thrust / G * 1e3 for x in rr]
    ax.plot(rr, t0, "k--", lw=1, label="uncalibrated")
    ax.plot([m0.rpm], [m0.thrust_gf], "r*", ms=16, label="measurement")
    ax.axhline(HOVER_GF, color="gray", ls=":", lw=1)
    ax.text(rr[0], HOVER_GF, " hover (9.2 gf)", color="gray", fontsize=8,
            va="bottom")
    ax.set_xlabel("rpm"); ax.set_ylabel("thrust [gf]")
    ax.set_ylim(0, 18); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("one measurement, two equally good hypotheses")
    fig.savefig(OUT / "09_calibration.png", dpi=130)
    print(f"\n-> {OUT / '09_calibration.png'}")


if __name__ == "__main__":
    main()
