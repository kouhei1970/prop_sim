"""例 8: 空力モデル (Lv0 - LvS) の総当たり比較.

比較する対象

    Lv0  QuadraticModel  Ct/Cq 多項式
    Lv1  BET             翼素理論 + 一様インフロー
    Lv2  BEMT            翼素運動量理論 (既定)
    Lv3  LiftingLine     揚力線 + 渦後流 (軸流のみ)
    LvS  SurrogateModel  応答曲面 (BEMT / 渦法から同定)

見どころ

    1. 軸流での推力・トルク・計算コスト
    2. 半径方向の荷重分布 — 翼端損失の扱いの違いが出る
    3. BEMT が無視している旋回 (swirl) の寄与を渦法で切り分ける
    4. 斜め流入での 6 分力 (渦法は非対応)
    5. 代理モデルの学習元を変えると何が変わるか

    python examples/08_algorithm_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import time

import numpy as np

import prop_sim as ps

OUT = Path(__file__).resolve().parent.parent / "results"
RPM = 6000.0


def timed(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    return out, (time.perf_counter() - t0) / n


def main() -> None:
    geo = ps.from_diameter_pitch(10.0, 4.7)
    rotor = ps.Rotor(geo)
    print(rotor.summary())

    bemt = ps.BEMT()
    bet = ps.BET()
    vortex = ps.LiftingLine()
    vortex_nosw = ps.LiftingLine(include_swirl=False)

    print("\n代理モデルを同定中 (BEMT から) ...")
    j_train = np.linspace(0.0, 0.7, 8)
    ang_train = np.array([0.0, 20.0, 40.0, 60.0])
    sur, t_fit = timed(lambda: ps.SurrogateModel.fit(
        rotor, bemt, rpm=[4000.0, 7000.0, 10000.0],
        advance_ratios=j_train, inflow_angles_deg=ang_train))
    quad = ps.QuadraticModel.fit(rotor, bemt, rpm=RPM, advance_ratios=j_train)
    print(f"  {sur.summary()}")
    print(f"  同定 {t_fit:.1f} s, 学習誤差(RMS) = "
          + ", ".join(f"{k}={v:.2e}" for k, v in sur.training_error(rotor).items()))

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 76)
    print(f"1) 軸流 (ホバー @ {RPM:.0f} rpm)")
    models = [
        ("Lv0 quadratic", quad, 500),
        ("Lv1 BET", bet, 3),
        ("Lv2 BEMT", bemt, 3),
        ("Lv3 lifting-line", vortex, 1),
        ("Lv3 LL (swirl 無視)", vortex_nosw, 1),
        ("LvS surrogate", sur, 500),
    ]
    op = ps.OperatingPoint(rpm=RPM)
    base = bemt.solve(rotor, op)
    print(f"{'model':<22}{'T[N]':>9}{'ΔT':>8}{'Q[mNm]':>9}{'ΔQ':>8}"
          f"{'vi[m/s]':>9}{'FM':>7}{'cost':>10}")
    results = {}
    for nm, m, rep in models:
        sol, dt = timed(lambda m=m: m.solve(rotor, op), rep)
        results[nm] = (sol, dt)
        cf = sol.coefficients(rotor, op)
        print(f"{nm:<22}{sol.thrust:9.4f}"
              f"{(sol.thrust/base.thrust-1)*100:7.1f}%{sol.torque()*1e3:9.4f}"
              f"{(sol.torque()/base.torque()-1)*100:7.1f}%"
              f"{sol.inflow.mean_v0():9.3f}{cf.get('FM', np.nan):7.3f}"
              f"{_fmt_time(dt):>10}")

    ll = results["Lv3 lifting-line"][0]
    ll0 = results["Lv3 LL (swirl 無視)"][0]
    print(f"\n  渦法どうしの差 = 旋回 (swirl) の寄与: "
          f"推力 {(ll.thrust/ll0.thrust-1)*100:+.1f} %, "
          f"トルク {(ll.torque()/ll0.torque()-1)*100:+.1f} %")
    print(f"  BEMT vs 渦法(swirl 無視) = アニュラス独立 + Prandtl 損失の誤差: "
          f"{(ll0.thrust/base.thrust-1)*100:+.1f} %")
    print(f"  BEMT vs 渦法(swirl 込み) = BEMT の実効的な誤差: "
          f"{(ll.thrust/base.thrust-1)*100:+.1f} %")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 76)
    print("2) 半径方向の荷重分布 dT/dr [N/m]")
    print(f"{'r/R':>6}" + "".join(f"{k.split()[0]:>12}"
                                  for k in ("Lv1 BET", "Lv2 BEMT", "Lv3 LL")))
    grids = {}
    for nm in ("Lv1 BET", "Lv2 BEMT", "Lv3 lifting-line"):
        st = results[nm][0].sections
        grids[nm] = (st.r_R, st.dfz_dr.mean(axis=0) * rotor.n_blades)
    xq = np.linspace(0.25, 1.0, 16)
    for x in xq[::2]:
        row = "".join(f"{np.interp(x, *grids[nm]):12.2f}"
                      for nm in ("Lv1 BET", "Lv2 BEMT", "Lv3 lifting-line"))
        print(f"{x:6.2f}{row}")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 76)
    print("3) 前進率掃引 (軸流)")
    n = RPM / 60.0
    js = np.linspace(0.0, 0.7, 8)
    print(f"{'J':>6}" + "".join(f"{k.split(' ', 1)[1][:12]:>13}" for k, _, _ in models))
    err = {k: [] for k, _, _ in models}
    for j in js:
        opj = ps.OperatingPoint(rpm=RPM, v_inf=j * n * geo.diameter)
        ref = bemt.solve(rotor, opj).thrust
        row = ""
        for nm, m, _ in models:
            try:
                t = m.solve(rotor, opj).thrust
            except NotImplementedError:
                row += f"{'n/a':>13}"
                continue
            err[nm].append(abs(t - ref))
            row += f"{t:13.4f}"
        print(f"{j:6.2f}{row}")
    print("\n  BEMT に対する推力の平均絶対差 [N]")
    for nm, _, _ in models:
        if err[nm]:
            print(f"    {nm:<22}{np.mean(err[nm]):.4f}")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 76)
    print("4) 斜め流入の 6 分力 (V=8 m/s, 流入角 40 deg)")
    op_ob = ps.OperatingPoint(rpm=RPM, v_inf=8.0, inflow_angle_deg=40.0)
    print(f"{'model':<22}" + "".join(f"{c:>10}" for c in ps.COMPONENT_NAMES))
    for nm, m, _ in models:
        try:
            w = m.solve(rotor, op_ob).wrench.as_array()
        except NotImplementedError as e:
            print(f"{nm:<22}  -> 非対応: {str(e).splitlines()[0][:44]}")
            continue
        print(f"{nm:<22}" + "".join(f"{v:10.4f}" for v in w))

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 76)
    print("5) 代理モデルの学習元による違い (軸流のみ)")
    sur_ll = ps.SurrogateModel.fit(
        rotor, vortex, rpm=[4000.0, 7000.0, 10000.0],
        advance_ratios=np.linspace(0.0, 0.6, 5), inflow_angles_deg=np.array([0.0]))
    print(f"{'J':>6}{'BEMT':>10}{'sur(BEMT)':>11}{'渦法':>10}{'sur(渦法)':>11}")
    for j in (0.0, 0.2, 0.4, 0.6):
        opj = ps.OperatingPoint(rpm=RPM, v_inf=j * n * geo.diameter)
        print(f"{j:6.2f}{bemt.solve(rotor,opj).thrust:10.4f}"
              f"{sur.solve(rotor,opj).thrust:11.4f}"
              f"{vortex.solve(rotor,opj).thrust:10.4f}"
              f"{sur_ll.solve(rotor,opj).thrust:11.4f}")
    print("\n  -> 代理モデルは学習元の忠実度をそのまま受け継ぐ.")
    print("     速さは学習元によらず一定なので, 学習元こそが精度を決める.")

    # ------------------------------------------------------------- グラフ
    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    labels = {"Lv1 BET": "Lv1 BET", "Lv2 BEMT": "Lv2 BEMT",
              "Lv3 lifting-line": "Lv3 lifting-line",
              "Lv3 LL (swirl 無視)": "Lv3 LL (no swirl)"}
    for nm, lab in labels.items():
        st = results[nm][0].sections
        ax[0].plot(st.r_R, st.dfz_dr.mean(axis=0) * rotor.n_blades, label=lab)
    ax[0].set_xlabel("r/R"); ax[0].set_ylabel("dT/dr [N/m]")
    ax[0].set_title("radial thrust loading"); ax[0].legend(fontsize=8)

    for nm, m, _ in models:
        t = []
        for j in js:
            opj = ps.OperatingPoint(rpm=RPM, v_inf=j * n * geo.diameter)
            try:
                t.append(m.solve(rotor, opj).thrust)
            except NotImplementedError:
                t.append(np.nan)
        ax[1].plot(js, t, "o-", ms=3, label=nm)
    ax[1].set_xlabel("J"); ax[1].set_ylabel("thrust [N]")
    ax[1].set_title("advance ratio sweep"); ax[1].legend(fontsize=7)

    for nm, _, _ in models:
        sol, dt = results[nm]
        ax[2].scatter(dt * 1e3, abs(sol.thrust - ll.thrust), s=45)
        ax[2].annotate(nm, (dt * 1e3, abs(sol.thrust - ll.thrust)),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("cost per point [ms]")
    ax[2].set_ylabel("|T - T(lifting-line)| [N]")
    ax[2].set_title("cost vs deviation from Lv3")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"aerodynamic model comparison — {geo.name} @ {RPM:.0f} rpm")
    fig.savefig(OUT / "08_model_comparison.png", dpi=130)
    print(f"\n-> {OUT / '08_model_comparison.png'}")


def _fmt_time(dt: float) -> str:
    if dt < 1e-3:
        return f"{dt*1e6:.0f} us"
    if dt < 1.0:
        return f"{dt*1e3:.1f} ms"
    return f"{dt:.2f} s"


if __name__ == "__main__":
    main()
