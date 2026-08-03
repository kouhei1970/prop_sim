"""参照データ (メーカ公表値・文献値).

**いずれも回転数を含まない**点に注意. 回転数の無いデータでは空力モデルを
較正できない (推力とトルクの両方が回転数の関数であり, 電気入力からは
モータ効率と分離できないため). できるのは

* 反証テスト — モデルが要求するモータ効率が 1 を超えないか
  (:func:`prop_sim.calibration.consistency_check`)
* 回転数によらない量 ``FM x eta_motor = P_ideal / P_elec`` の比較

の 2 つ.
"""

from .gemfan_1209_4 import GEMFAN_1209_4, gemfan_1209_4_points

__all__ = ["GEMFAN_1209_4", "gemfan_1209_4_points"]
