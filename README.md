# prop_sim — 無人機プロペラ 6 分力試験シミュレータ

回転するプロペラの **6 分力 (Fx, Fy, Fz, Mx, My, Mz)** を計測する実験を、
空力から 6 分力計の出力までまるごとシミュレートする Python パッケージ。

- 空力モデルは **Lv0（Ct/Cq 多項式）から Lv3（揚力線+渦後流）まで**を実装し、
  同じインターフェイスで差し替えられる。既定は **翼素運動量理論 (BEMT)**。
  外部 CFD/実測データの取り込み口もある → [docs/MODELS.md](docs/MODELS.md)
- **静的データ**（回転数・風速・流入角の掃引）と
  **動的データ**（時間領域、1/rev・B/rev・過渡応答）の両方を出力
- 「真値」だけでなく **6 分力計を通した計測値**（軸間干渉・ノイズ・
  構造共振・A/D 量子化・エイリアシング）を再現する

```
真の空力荷重 ──┐
慣性・ジャイロ ─┼→ ハブ 6 分力 → 作用点移動 → 6 分力計 → A/D → 「計測値」
アンバランス ──┘                            (干渉/共振/ノイズ)
```

## インストール

```bash
pip install -e .          # numpy, scipy
pip install -e ".[plot]"  # + matplotlib (例のグラフ用)
```

## 使い方

### プリセット形状

実測に基づくプロペラ形状を同梱している。

| 名前 | 内容 |
|------|------|
| `stampfly_1209` | M5Stack StampFly 用 1209 (31 mm 4 枚)。**平面形と翼断面**を実機写真から実測 → [docs/STAMPFLY_1209.md](docs/STAMPFLY_1209.md) |

```python
rotor = ps.Rotor(ps.stampfly_1209())          # 既定 P=0.9 in, 実測断面の翼型
rotor = ps.Rotor(ps.stampfly_1209(pitch_in=1.0, pitch_distribution="washout"))
ps.STAMPFLY_1209_SECTION.thin_airfoil_properties()   # 実測断面の薄翼理論特性
```

```bash
python -m prop_sim static --preset stampfly_1209 --rpm 10000:40000:5000
```

```python
import numpy as np
import prop_sim as ps
from prop_sim.experiments import static_sweep, dynamic_run
from prop_sim.sensor import TestRig

rotor = ps.Rotor(ps.from_diameter_pitch(10.0, 4.7))   # 10x4.7, 2 枚
model = ps.BEMT()

# --- 静的: 静止推力試験
op = ps.OperatingPoint(rpm=6000)
sol = model.solve(rotor, op)
print(sol.thrust, sol.torque(), sol.coefficients(rotor, op))

# --- 静的: 条件掃引 + 6 分力計
rig = TestRig(sensor_offset=np.array([0.0, 0.0, -0.12]))
res = static_sweep(rotor, model, rpm=np.arange(2000, 10001, 500), rig=rig)
res.to_csv("static.csv")

# --- 動的: 斜め流入での時間波形
dyn = dynamic_run(rotor, model, duration=0.2, rpm=7000,
                  v_inf=10.0, inflow_angle_deg=30.0, rig=rig)
dyn.to_csv("dynamic.csv")
```

コマンドラインからも:

```bash
python -m prop_sim static  --diameter 10 --pitch 4.7 --rpm 2000:10000:500 --out static.csv
python -m prop_sim dynamic --diameter 10 --pitch 4.7 --rpm 7000 \
    --v-inf 10 --inflow-angle 30 --duration 0.3 --out dynamic.csv
```

## 例

| ファイル | 内容 |
|---------|------|
| `examples/01_static_thrust_sweep.py` | 静止推力試験。Ct, Cp, FM, g/W, 電流 |
| `examples/02_wind_tunnel_map.py` | 前進率 × 流入角の 6 分力マップ |
| `examples/03_dynamic_throttle_step.py` | スロットルステップ。動的インフロー・慣性反トルク・センサ共振 |
| `examples/04_imbalance_diagnostics.py` | 質量/空力アンバランスの次数分析 |
| `examples/05_model_comparison.py` | BEMT / BET / 代理モデルの比較と同定 |
| `examples/06_measurement_error_budget.py` | 計測系の誤差要因の切り分け |
| `examples/07_stampfly_1209.py` | StampFly 1209 (31 mm 4 枚) の性能推定と計測要求 |
| `examples/08_algorithm_comparison.py` | Lv0〜LvS の総当たり比較（精度・コスト・BEMT の誤差分解） |

```bash
python examples/01_static_thrust_sweep.py     # → results/ に CSV と PNG
```

## 座標系と符号

ハブ座標系（`prop_sim/frames.py`）:

- `z` : シャフト軸。**推力が正**になる向き
- `x` : 方位角 `psi = 0` の基準方向（ディスク面内）
- `y` : `z × x`（右手系）。方位角は +x → +y が正
- 回転方向は `Rotor.spin`（`+1` = +z まわり右ねじ / `-1` = 逆）

出力される 6 分力は **「プロペラアセンブリが試験台に及ぼすレンチ」** で、
実際のロードセルの読みと一致する。定常状態では空力レンチに等しい。

- `Fz > 0` : 推力
- `Mz < 0` : `spin=+1` のときの反トルク（回転を妨げる向き）
- `OperatingPoint.inflow_angle_deg` : シャフト軸と進行方向のなす角
  （0 = 軸流、90 = 完全な横流れ）

回転方向を反転すると 6 分力が x–z 面に関して厳密に鏡映になることを
テストで確認している（`test_reversed_spin_mirrors_hub_moment_in_edgewise_flow`）。

## 何がモデル化されているか

**空力**

- 翼素運動量理論（Prandtl 翼端/ハブ損失、Glauert の斜め流入運動量式）
- 揚力線 + 渦後流（Biot–Savart、軸流のみ）— BEMT の検証用の参照解
- 応答曲面による代理モデル / 外部 CFD・実測データの取り込み
- 翼断面形状 → 翼型モデルの生成（薄翼理論 + 低 Re 粘性補正、`prop_sim.section`）
- 方位角方向の線形インフロー分布（Drees / Pitt / Coleman）
- 360 deg 翼型ポーラ（線形 + Viterna 外挿）、Reynolds・圧縮性補正
- 動的インフロー（Pitt–Peters 系の見かけ質量に基づく 1 次遅れ）
- 試験台の角運動（`body_rate`）による流入変化

**機械系**

- 回転慣性による反トルク `-J dOmega/dt`
- ジャイロモーメント `-omega_body × (J Omega e_z)`
- 質量アンバランス（静・偶力） → 1/rev の `U Omega^2`
- ブレード間のピッチ差・取付方位ずれ（空力アンバランス）
- BLDC モータ（Kv, 巻線抵抗, 無負荷電流）+ ESC + ベアリング損失

**計測系**

- 作用点のずれによるモーメント換算、取付角度誤差
- 軸間干渉（クロストーク）と校正の残差
- 感度誤差・オフセット・ランダムウォーク・温度ドリフト
- 構造共振（チャネルごとの 2 次系）
- 抗エイリアスフィルタ、サンプリング、A/D 量子化・飽和

**後処理**（`prop_sim.postproc`）

- 次数分析（オーダートラッキング）、回転同期平均、FFT、次数スペクトル
- Ct / Cq / Cp / eta / CT / CP / FM の換算

## 6 分力に何が現れるか

2 枚ブレードで実際に計算した結果（`examples/04`）:

| 状態 | Fx, Fy | Fz | Mx, My | Mz |
|------|--------|----|--------|-----|
| 健全・軸流 | 0 | 一定 | 0 | 一定 |
| 質量アンバランス | **1/rev** = `U Omega^2` | 0 | 0 | 0 |
| 空力アンバランス（1 枚 +1 deg） | 1/rev | 平均値がずれる | **1/rev** | 平均値がずれる |
| 斜め流入 45 deg | 定常 + **2/rev** | 定常 + 2/rev | 定常 + **2/rev** | 定常 |

斜め流入ではブレード 1 枚あたりは 1/rev で変動するが、B 枚合わせた
ハブ荷重には **B の倍数の次数だけが残る**（回転系 → 静止系の次数変換）。
シミュレータはこの性質を正しく再現する。

## モデルどうしの比較

10x4.7 2 枚・6000 rpm ホバーで実際に走らせた結果（`examples/08`）:

| モデル | 推力 [N] | BEMT との差 | コスト |
|--------|---------|-----------|-------|
| Lv0 Ct/Cq 多項式 | 4.329 | +0.8 % | 216 µs |
| Lv1 BET（翼端損失なし） | 4.542 | +5.8 % | 26 ms |
| Lv2 BEMT | 4.293 | — | 22 ms |
| Lv3 揚力線+渦後流（旋回込み） | 4.096 | **−4.6 %** | 0.56 s |
| Lv3 同（旋回を無視） | 4.267 | **−0.6 %** | 1.1 s |
| LvS 応答曲面（BEMT から同定） | 4.266 | −0.6 % | 260 µs |

下 2 行の差が **BEMT の誤差分解**になっている。BEMT が無視している後流の
旋回が推力を 4 % 押し上げており、残る近似（アニュラス独立 + Prandtl 翼端損失）
の誤差は 0.6 % しかない。

## 精度について

`from_diameter_pitch()` が生成するのは "D x P" 表記からの**代表的な**分布であり、
特定メーカーの実測形状ではない（`stampfly_1209` プリセットは写真実測の平面形を
持つが、ピッチ分布は公称値からの仮定）。実機と比べる場合は

1. 実測のコード長・ねじり角分布を `PropellerGeometry` に与える
2. 翼型を実測断面から作る（`airfoil_from_section`）か、`TabulatedAirfoil`
   （XFOIL / 風洞データ）に置き換える
3. 静止推力の 1 点で `collective_deg` を較正する

の順に精度が上がる。とくに 30 mm 級の微小プロペラは翼端でも Re < 2×10⁴ で、
翼型ポーラの選び方だけで推力が ±20 % 動く（`examples/07` に感度を出力）。
既定では `LOW_RE_THIN`（Re ~ 10⁴ の薄翼相当）を使う。

BEMT 自体の限界（ボルテックスリング状態、大流入角、多ロータ干渉、動的失速）は
[docs/MODELS.md](docs/MODELS.md) にまとめてある。

## 開発

```bash
python -m pytest -q
```

テストは数値の一致だけでなく、運動量理論との整合、rpm^2 則、回転方向の
鏡映対称性、アンバランス振幅の理論値 `U Omega^2`、ハブ荷重の次数変換と
いった**物理的性質**を検証している。

## ライセンス

MIT
