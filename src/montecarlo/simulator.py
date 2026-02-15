"""
ロト予測ツール - モンテカルロ・シミュレーション エンジン

重み付きランダム抽選を大量試行し、頻出パターンを抽出する。
"""

import random
from typing import Callable, Optional

from src.common import LOTTERY_CONFIG


class MonteCarloSimulator:
    """
    モンテカルロ・シミュレーションによるロト番号予測エンジン。

    指定された重みに基づいて非重複の数字セットをN回抽選し、
    結果をソート済みタプルのリストとして返す。

    使用例:
        >>> from src.common.weights import calculate_frequency_weights
        >>> weights = calculate_frequency_weights(data, "LOTO6")
        >>> sim = MonteCarloSimulator("LOTO6", weights, trials=100_000)
        >>> results = sim.run()
    """

    def __init__(
        self,
        game_key: str,
        weights: dict[int, float],
        trials: int = 100_000,
    ) -> None:
        """
        Args:
            game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）
            weights: {数字: 重み} の辞書
            trials: シミュレーション試行回数
        """
        game_key = game_key.upper()
        if game_key not in LOTTERY_CONFIG:
            raise ValueError(f"不正なゲームキー: '{game_key}' (有効: {', '.join(LOTTERY_CONFIG.keys())})")

        self.game_key = game_key
        self.config = LOTTERY_CONFIG[game_key]
        self.pick_size: int = self.config["pick_size"]
        self.range_max: int = self.config["range_max"]
        self.trials = trials

        # 重み辞書からリストを構築（random.choices用）
        self.population: list[int] = list(range(1, self.range_max + 1))
        self.weights_list: list[float] = [weights.get(num, 1.0) for num in self.population]

    def run(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        progress_interval: int = 10_000,
    ) -> list[tuple[int, ...]]:
        """
        シミュレーションを実行する。

        Args:
            progress_callback: 進行状況通知関数 fn(current, total)
            progress_interval: コールバック呼び出し間隔（試行回数）

        Returns:
            各試行結果のソート済みタプルのリスト
        """
        results: list[tuple[int, ...]] = []

        for i in range(self.trials):
            # 重み付き非重複抽選
            picked = self._weighted_sample()
            results.append(tuple(sorted(picked)))

            # 進行状況の通知
            if progress_callback and (i + 1) % progress_interval == 0:
                progress_callback(i + 1, self.trials)

        return results

    def _weighted_sample(self) -> set[int]:
        """
        重み付きで非重複のpick_size個の数字を抽選する。

        random.choices は重複を許すため、
        while ループでユニークな数字が揃うまで繰り返す。
        """
        picked: set[int] = set()
        while len(picked) < self.pick_size:
            chosen = random.choices(
                self.population,
                weights=self.weights_list,
                k=1,
            )[0]
            picked.add(chosen)
        return picked
