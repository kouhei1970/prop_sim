"""例 10: OpenFOAM で翼断面のポーラを解き, 経験式と比べる.

やること

    1. 円柱 Re = 20 / 40 で検証 (文献値 Cd = 2.0-2.1 / 1.50-1.55)
    2. StampFly 1209 の実測断面のポーラを Re = 12,000 で解く
    3. `airfoil_from_section` の経験式との差を出す
    4. CFD ポーラを使ったロータ性能を BEMT で再計算して比較

前提

    OpenFOAM が入っていること (`prop_sim.cfd.find_openfoam()` で確認).
    入っていなければ 1. 2. は飛ばして経験式だけ表示する.

計算コスト

    この断面は Re = 12,000 では**定常解が存在しない** (剥離渦の放出) ので
    非定常 `pimpleFoam` が要る. 1 迎角あたり 10-30 分. 既定では迎角を
    3 点しか振らないので, 本格的なポーラが要るなら --alphas で増やして
    一晩回すこと.

    python examples/10_cfd_airfoil_polar.py
    python examples/10_cfd_airfoil_polar.py --alphas -4,0,4,8,12 --end-time 40
    python examples/10_cfd_airfoil_polar.py --skip-cfd     # 経験式だけ
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import numpy as np

import prop_sim as ps
from prop_sim.cfd import AirfoilCaseConfig, OpenFoamCase, find_openfoam
from prop_sim.section import SectionShape

OUT = Path(__file__).resolve().parent.parent / "results"
RE = 1.2e4


def cylinder_section(n=200):
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    t = 2.0 * np.sqrt(np.maximum(0.25 - (x - 0.5) ** 2, 0.0))
    return SectionShape(x=x, camber=np.zeros_like(x), thickness=t, name="cylinder")


def validate_cylinder(workdir):
    """既知解での検証. 格子・ソルバ・荷重積分をまとめて確かめる."""
    print("\n=== 1. 検証: 円柱の低 Reynolds 数流れ ===")
    print(f"{'Re':>5s} {'Cd':>8s} {'Cd_p':>8s} {'Cd_f':>8s} {'Cl':>10s}  文献値")
    ref = {20.0: "2.00-2.09", 40.0: "1.50-1.55"}
    for re in (20.0, 40.0):
        cfg = AirfoilCaseConfig(reynolds=re, n_surface=120, n_radial=70,
                                far_field_radius=40.0, radial_grading=800.0,
                                end_time=4000, write_interval=4000)
        case = OpenFoamCase(workdir / f"cyl{re:.0f}", cylinder_section(), cfg)
        case.write()
        case.run()
        r = case.coefficients()
        print(f"{re:5.0f} {r['Cd']:8.4f} {r['Cd_pressure']:8.4f} "
              f"{r['Cd_viscous']:8.4f} {r['Cl']:10.2e}  {ref[re]}")


def run_section_polar(workdir, alphas, end_time):
    """StampFly 断面のポーラ (非定常, 後半を時間平均)."""
    print(f"\n=== 2. StampFly 1209 断面のポーラ (Re = {RE:.0f}) ===")
    print("    定常解が無いので pimpleFoam で解いて時間平均する")
    rows = []
    for a in alphas:
        cfg = AirfoilCaseConfig(
            reynolds=RE, alpha_deg=float(a), solver="pimpleFoam",
            n_surface=160, n_radial=100, far_field_radius=20.0,
            radial_grading=500.0, delta_t=1e-4, max_courant=3.0,
            end_time=end_time, write_interval=end_time / 20.0,
            average_last_fraction=0.5,
        )
        case = OpenFoamCase(workdir / f"a{a:+05.1f}".replace("+", "p").replace(
            "-", "m"), ps.STAMPFLY_1209_SECTION, cfg)
        case.write()
        t0 = time.perf_counter()
        case.run(timeout=24 * 3600)
        r = case.coefficients()
        r["alpha_deg"] = float(a)
        r["wall_s"] = time.perf_counter() - t0
        rows.append(r)
        print(f"  alpha={a:+5.1f}  Cl={r['Cl']:7.4f} (±{r['Cl_std']:.4f})  "
              f"Cd={r['Cd']:7.5f} (±{r['Cd_std']:.5f})  Cm={r['CmPitch']:+7.4f}"
              f"   {r['wall_s']:.0f}s")
    return rows


def compare_with_empirical(rows, alphas):
    """経験式 (`airfoil_from_section`) との比較."""
    print("\n=== 3. 経験式との比較 ===")
    af = ps.stampfly_1209_airfoil()
    a = np.asarray(alphas, float)
    cl_e, cd_e, cm_e = af.coefficients(np.deg2rad(a), reynolds=np.full_like(a, RE))
    if rows is None:
        print(f"{'alpha':>6s} {'cl':>8s} {'cd':>8s} {'cm':>8s}  (経験式のみ)")
        for x, l, d, m in zip(a, cl_e, cd_e, cm_e):
            print(f"{x:6.1f} {l:8.4f} {d:8.5f} {m:8.4f}")
        return None
    print(f"{'alpha':>6s} | {'cl(CFD)':>8s} {'cl(経験)':>9s} {'差':>7s}"
          f" | {'cd(CFD)':>8s} {'cd(経験)':>9s} {'差':>7s}")
    for r, l, d in zip(rows, cl_e, cd_e):
        print(f"{r['alpha_deg']:6.1f} | {r['Cl']:8.4f} {l:9.4f} "
              f"{100*(r['Cl']/l-1):6.1f}% | {r['Cd']:8.5f} {d:9.5f} "
              f"{100*(r['Cd']/d-1):6.1f}%")
    return ps.TabulatedAirfoil(
        alpha_deg=np.array([r["alpha_deg"] for r in rows]),
        cl=np.array([r["Cl"] for r in rows]),
        cd=np.array([r["Cd"] for r in rows]),
        cm=np.array([r["CmPitch"] for r in rows]),
        name=f"StampFly 1209 CFD Re={RE:.0f}", reynolds_ref=RE, aspect_ratio=3.0)


def rotor_effect(cfd_airfoil):
    """CFD ポーラでロータ性能がどれだけ変わるか."""
    print("\n=== 4. ロータ性能への波及 (BEMT, ホバー) ===")
    model = ps.BEMT()
    base = ps.Rotor(ps.stampfly_1209())
    cfd = ps.Rotor(ps.stampfly_1209(airfoil=cfd_airfoil))
    print(f"{'rpm':>7s} | {'T(経験)[gf]':>12s} {'T(CFD)[gf]':>11s} {'差':>7s}"
          f" | {'P(経験)[W]':>11s} {'P(CFD)[W]':>10s}")
    for rpm in (12000.0, 20000.0, 30000.0, 40000.0):
        op = ps.OperatingPoint(rpm=rpm)
        a = model.solve(base, op)
        b = model.solve(cfd, op)
        print(f"{rpm:7.0f} | {a.thrust/9.80665*1e3:12.3f} "
              f"{b.thrust/9.80665*1e3:11.3f} {100*(b.thrust/a.thrust-1):6.1f}%"
              f" | {a.power(op):11.3f} {b.power(op):10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0,4,8",
                    help="迎角 [deg] をカンマ区切りで")
    ap.add_argument("--end-time", type=float, default=25.0,
                    help="非定常計算の終了時刻 [s] (コード長 1 m, U = 1 m/s)")
    ap.add_argument("--skip-cfd", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--workdir", default=str(OUT / "cfd"))
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    bashrc = find_openfoam()
    if args.skip_cfd or bashrc is None:
        if bashrc is None:
            print("OpenFOAM が見つからないので経験式だけ表示します")
        compare_with_empirical(None, alphas)
        return
    print(f"OpenFOAM: {bashrc or 'PATH 済み'}")

    if not args.skip_validation:
        validate_cylinder(workdir)
    rows = run_section_polar(workdir, alphas, args.end_time)
    cfd_airfoil = compare_with_empirical(rows, alphas)

    OUT.mkdir(exist_ok=True)
    csv = OUT / "cfd_polar_stampfly_1209.csv"
    with csv.open("w") as f:
        f.write("alpha_deg,cl,cd,cm,cl_std,cd_std,wall_s\n")
        for r in rows:
            f.write(f"{r['alpha_deg']},{r['Cl']},{r['Cd']},{r['CmPitch']},"
                    f"{r['Cl_std']},{r['Cd_std']},{r['wall_s']}\n")
    print(f"\n-> {csv}")

    if len(rows) >= 2:
        rotor_effect(cfd_airfoil)


if __name__ == "__main__":
    main()
