"""試験シナリオ (静的掃引 / 動的時間領域)."""

from .dynamic import DynamicResult, dynamic_run
from .static import StaticResult, static_sweep

__all__ = ["static_sweep", "StaticResult", "dynamic_run", "DynamicResult"]
