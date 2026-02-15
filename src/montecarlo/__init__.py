"""
ロト予測ツール - モンテカルロ・シミュレーション モジュール

重み付きランダム抽選を大量試行し、頻出パターンを抽出する。

使用方法:
    python -m src.montecarlo [--game loto6|loto7|miniloto] [--trials N]
"""

from src.montecarlo.simulator import MonteCarloSimulator
from src.montecarlo.analyzer import (
    analyze_top_combinations,
    analyze_number_frequency,
    print_report,
)

__all__ = [
    "MonteCarloSimulator",
    "analyze_top_combinations",
    "analyze_number_frequency",
    "print_report",
]
