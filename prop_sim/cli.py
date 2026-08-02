"""コマンドラインインターフェイス.

例::

    python -m prop_sim static --diameter 10 --pitch 4.7 --rpm 2000:9000:500 \
        --out results/static.csv
    python -m prop_sim dynamic --diameter 10 --pitch 4.7 --rpm 6000 \
        --v-inf 8 --inflow-angle 25 --duration 0.3 --out results/dyn.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .drivetrain import Drivetrain
from .experiments import dynamic_run, static_sweep
from .geometry import from_diameter_pitch
from .models import get_model
from .rotor import Rotor, Unbalance
from .sensor import LoadCell, SensorConfig, TestRig


def _parse_range(text: str) -> np.ndarray:
    """``"5000"`` / ``"2000:9000:500"`` / ``"1,2,3"`` を配列に変換する."""
    text = str(text).strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = (stop - start) / 10.0 or 1.0
        else:
            start, stop, step = parts
        return np.arange(start, stop + 0.5 * step, step)
    if "," in text:
        return np.array([float(p) for p in text.split(",")])
    return np.array([float(text)])


def _build(args) -> tuple[Rotor, object, TestRig]:
    geo = from_diameter_pitch(
        args.diameter, args.pitch, n_blades=args.blades,
        chord_ratio_75=args.chord_ratio,
    )
    rotor = Rotor(
        geo,
        spin=args.spin,
        collective_deg=args.collective,
        blade_pitch_offsets_deg=(
            [0.0] * (args.blades - 1) + [args.pitch_imbalance]
            if args.pitch_imbalance else None
        ),
        unbalance=Unbalance(static=args.unbalance),
    )
    model = get_model(args.model)
    rig = TestRig(
        sensor_offset=np.array([0.0, 0.0, -args.sensor_offset]),
        load_cell=LoadCell(
            SensorConfig(
                full_scale=np.array(args.full_scale, dtype=float),
                sample_rate=args.sample_rate,
                adc_bits=args.adc_bits,
                seed=args.seed,
            )
        ),
    )
    return rotor, model, rig


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("propeller")
    g.add_argument("--diameter", type=float, default=10.0, help="直径 [inch]")
    g.add_argument("--pitch", type=float, default=4.7, help="ピッチ [inch]")
    g.add_argument("--blades", type=int, default=2, help="ブレード枚数")
    g.add_argument("--chord-ratio", type=float, default=0.155,
                   help="0.75R のコード比 c/R")
    g.add_argument("--spin", type=int, default=1, choices=[1, -1])
    g.add_argument("--collective", type=float, default=0.0,
                   help="ピッチオフセット [deg]")
    g.add_argument("--model", default="bemt", help="bemt / bet / quadratic")

    g = p.add_argument_group("imperfections")
    g.add_argument("--unbalance", type=float, default=0.0,
                   help="静アンバランス [kg m]")
    g.add_argument("--pitch-imbalance", type=float, default=0.0,
                   help="1 枚だけピッチをずらす [deg]")

    g = p.add_argument_group("test rig")
    g.add_argument("--sensor-offset", type=float, default=0.12,
                   help="ハブからセンサ原点までの距離 [m]")
    g.add_argument("--full-scale", type=float, nargs=6,
                   default=[50, 50, 100, 2, 2, 2], help="6 分力計の定格")
    g.add_argument("--sample-rate", type=float, default=10000.0, help="[Hz]")
    g.add_argument("--adc-bits", type=int, default=16)
    g.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="出力 CSV")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prop-sim", description="プロペラ 6 分力試験シミュレータ"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ps_ = sub.add_parser("static", help="静的掃引")
    _common(ps_)
    ps_.add_argument("--rpm", default="2000:9000:1000")
    ps_.add_argument("--v-inf", default="0")
    ps_.add_argument("--inflow-angle", default="0")
    ps_.add_argument("--drivetrain", action="store_true",
                     help="モータ・ESC の推定値も出力する")

    pd_ = sub.add_parser("dynamic", help="動的 (時間領域) 試験")
    _common(pd_)
    pd_.add_argument("--rpm", type=float, default=6000.0)
    pd_.add_argument("--v-inf", type=float, default=0.0)
    pd_.add_argument("--inflow-angle", type=float, default=0.0)
    pd_.add_argument("--duration", type=float, default=0.2)
    pd_.add_argument("--sim-rate", type=float, default=None)

    args = parser.parse_args(argv)
    rotor, model, rig = _build(args)

    if args.command == "static":
        res = static_sweep(
            rotor, model,
            rpm=_parse_range(args.rpm),
            v_inf=_parse_range(args.v_inf),
            inflow_angle_deg=_parse_range(args.inflow_angle),
            rig=rig,
            drivetrain=Drivetrain() if args.drivetrain else None,
            progress=True,
        )
        print(rotor.summary())
        print(res.summary())
        for w in res.warnings:
            print("  warning:", w)
        if args.out:
            res.to_csv(args.out)
            print("wrote", args.out)
        else:
            print(f"{'rpm':>8} {'V':>6} {'Fz[N]':>9} {'Mz[Nm]':>10} {'Ct':>8} {'eta':>7}")
            for i in range(res.n_points):
                print(
                    f"{res['rpm'][i]:8.0f} {res['v_inf'][i]:6.1f} "
                    f"{res['Fz'][i]:9.4f} {res['Mz'][i]:10.5f} "
                    f"{res['Ct'][i]:8.4f} {res['eta'][i]:7.3f}"
                )
    else:
        res = dynamic_run(
            rotor, model,
            duration=args.duration,
            rpm=args.rpm,
            v_inf=args.v_inf,
            inflow_angle_deg=args.inflow_angle,
            rig=rig,
            sim_rate=args.sim_rate,
            progress=True,
        )
        print(rotor.summary())
        print(res.summary())
        if args.out:
            res.to_csv(args.out)
            print("wrote", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
