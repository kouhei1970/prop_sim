"""SNS 用のカード画像 (OG image) を生成する.

    pip install playwright        # ブラウザ本体は同梱済み
    python docs/social/build_card.py

出力

    docs/social/og-card.png       1280 x 640  リポジトリの Social preview 用
    docs/social/og-card.html      その元 HTML (デバッグ用)

プロペラの絵は **実測形状から描いている**。コード長分布・ハブ比・翼端後退角は
``prop_sim.stampfly_1209()`` の値をそのまま使うので、飾りではなく
このリポジトリが扱っている実物の平面形になっている。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import prop_sim as ps

HERE = Path(__file__).resolve().parent
W, H = 1280, 640
TIP_SWEEP_DEG = 34.5          # 実測 (docs/STAMPFLY_1209.md)


def blade_path(cx: float, cy: float, radius_px: float, blade: int,
               n_blades: int) -> str:
    """実測のコード長分布から 1 枚のブレード輪郭を SVG path にする."""
    g = ps.stampfly_1209()
    R = g.diameter / 2
    r_R = np.linspace(g.hub_radius_ratio, 1.0, 60)
    c_R = g.chord(r_R) / R                       # コード長 / 翼半径
    # 中心線の後退角: 根元 0 -> 翼端 TIP_SWEEP_DEG (2 乗で効かせる)
    t = (r_R - r_R[0]) / (1.0 - r_R[0])
    sweep = np.deg2rad(TIP_SWEEP_DEG) * t**2
    base = 2.0 * np.pi * blade / n_blades
    # 弧長 = コード長 とみなして角度に直す
    half = 0.5 * c_R / np.maximum(r_R, 1e-6)

    def xy(ang, rr):
        return (cx + radius_px * rr * np.cos(ang),
                cy + radius_px * rr * np.sin(ang))

    lead = xy(base + sweep - half, r_R)
    trail = xy(base + sweep + half, r_R)
    pts = list(zip(*lead)) + list(zip(*trail))[::-1]
    d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f} "
    d += " ".join(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
    return d + " Z"


def build_svg(cx, cy, r_px, n_blades=4) -> str:
    blades = "\n".join(
        f'<path d="{blade_path(cx, cy, r_px, b, n_blades)}" '
        f'fill="url(#bladeGrad)" stroke="#7fb3ff" stroke-width="1.2" '
        f'stroke-opacity="0.55"/>'
        for b in range(n_blades)
    )
    rings = "\n".join(
        f'<circle cx="{cx}" cy="{cy}" r="{r_px*k:.1f}" fill="none" '
        f'stroke="#3a5578" stroke-width="1" stroke-dasharray="2 6"/>'
        for k in (0.5, 0.75, 1.0)
    )
    # ダクト (ベルマウス外径 36.2 / プロペラ 31.21)
    duct = (f'<circle cx="{cx}" cy="{cy}" r="{r_px*36.2/31.21:.1f}" fill="none" '
            f'stroke="#8fb0ff" stroke-width="2" stroke-opacity="0.45"/>')
    ticks = "\n".join(
        f'<line x1="{cx + r_px*1.19*np.cos(np.deg2rad(a)):.1f}" '
        f'y1="{cy + r_px*1.19*np.sin(np.deg2rad(a)):.1f}" '
        f'x2="{cx + r_px*1.25*np.cos(np.deg2rad(a)):.1f}" '
        f'y2="{cy + r_px*1.25*np.sin(np.deg2rad(a)):.1f}" '
        f'stroke="#43648f" stroke-width="{2 if a % 90 == 0 else 1}"/>'
        for a in range(0, 360, 15)
    )
    return f"""<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <radialGradient id="bladeGrad" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#2f6fd0" stop-opacity="0.95"/>
      <stop offset="100%" stop-color="#173a6b" stop-opacity="0.9"/>
    </radialGradient>
  </defs>
  {rings}
  {duct}
  {ticks}
  {blades}
  <circle cx="{cx}" cy="{cy}" r="{r_px*0.20:.1f}" fill="#0f1720"
          stroke="#7fb3ff" stroke-width="1.5"/>
</svg>"""


CARD = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{W}px; height:{H}px; overflow:hidden;
    background:
      radial-gradient(1100px 640px at 78% 50%, #16233a 0%, transparent 62%),
      linear-gradient(160deg, #0a0f16 0%, #0d141d 55%, #0a1017 100%);
    color:#e7edf4;
    font-family: "Liberation Sans","DejaVu Sans","Noto Sans CJK JP",
      "IPAGothic","Hiragino Sans",sans-serif;
    position:relative; }}
  .grid {{ position:absolute; inset:0;
    background-image:
      linear-gradient(#1b2a3d 1px, transparent 1px),
      linear-gradient(90deg, #1b2a3d 1px, transparent 1px);
    background-size: 40px 40px; opacity:.32; }}
  .art {{ position:absolute; right:-40px; top:-30px; opacity:.95; }}
  .wrap {{ position:relative; padding:62px 68px; height:100%;
    display:flex; flex-direction:column; justify-content:space-between; }}
  .eyebrow {{ font-family:"DejaVu Sans Mono","Liberation Mono",monospace;
    font-size:19px; letter-spacing:.30em; text-transform:uppercase;
    color:#6f9bd8; }}
  h1 {{ font-size:63px; line-height:1.16; font-weight:800;
    letter-spacing:-.022em; max-width:730px; }}
  h1 .hl {{ color:#8fb0ff; }}
  .sub {{ font-size:23px; line-height:1.62; color:#a9b7c8;
    max-width:640px; margin-top:22px; }}
  .stats {{ display:flex; gap:52px; align-items:flex-end; }}
  .stat .v {{ font-family:"DejaVu Sans Mono","Liberation Mono",monospace;
    font-size:37px; font-weight:700; color:#fff; letter-spacing:-.02em; }}
  .stat .v span {{ font-size:17px; color:#7f8ea3; margin-left:5px; }}
  .stat .k {{ font-size:15px; color:#7f8ea3; margin-top:5px; }}
  .repo {{ font-family:"DejaVu Sans Mono","Liberation Mono",monospace;
    font-size:20px; color:#6f9bd8; }}
  .rule {{ height:4px; width:96px; margin:26px 0 0;
    background:repeating-linear-gradient(90deg,#8fb0ff 0 3px,transparent 3px 8px); }}
</style></head><body>
  <div class="grid"></div>
  <div class="art">{svg}</div>
  <div class="wrap">
    <div>
      <div class="eyebrow">Virtual Wind Tunnel</div>
      <div class="rule"></div>
      <h1 style="margin-top:30px">31&nbsp;mm のプロペラの<br><span class="hl">6 分力</span>を丸ごと測る</h1>
      <div class="sub">{sub}</div>
    </div>
    <div class="stats">
      {stats}
      <div style="flex:1"></div>
      <div class="repo">github.com/kouhei1970/prop_sim</div>
    </div>
  </div>
</body></html>"""

SUB = ("実測形状のプロペラを、静止から前進飛行まで。"
       "空力から 6 分力計の出力までを丸ごと計算する"
       "無人機プロペラ試験のシミュレータ。")

STATS = [
    ("6", "分力 Fx Fy Fz Mx My Mz", ""),
    ("5", "段階の空力モデルを比較", ""),
    ("200", "件のテストで物理性質を検証", ""),
]


def main() -> None:
    svg = build_svg(cx=W * 0.795, cy=H * 0.50, r_px=232)
    stats = "\n".join(
        f'<div class="stat"><div class="v">{v}<span>{u}</span></div>'
        f'<div class="k">{k}</div></div>' for v, k, u in STATS)
    html = CARD.format(W=W, H=H, svg=svg, sub=SUB, stats=stats)
    src = HERE / "og-card.html"
    src.write_text(html)

    from playwright.sync_api import sync_playwright
    out = HERE / "og-card.png"
    # 同梱の Chromium を明示的に指す (playwright のビルド番号と一致しないため)
    exe = next((str(p) for p in (
        Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome"),
        Path("/opt/pw-browsers/chromium/chrome-linux/chrome"),
    ) if p.is_file()), None)
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=exe)
        pg = b.new_page(viewport={"width": W, "height": H},
                        device_scale_factor=1)
        pg.goto(src.as_uri())
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(out))
        b.close()
    print(f"-> {out}  ({W}x{H})")


if __name__ == "__main__":
    main()
