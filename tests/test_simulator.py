"""
テスト - モンテカルロ・シミュレーション エンジン
"""

import pytest

from src.common import LOTTERY_CONFIG
from src.montecarlo.simulator import MonteCarloSimulator


class TestMonteCarloSimulator:
    """MonteCarloSimulator のテスト"""

    @pytest.fixture
    def uniform_weights_loto6(self) -> dict[int, float]:
        """ロト6用の均一重み"""
        return {n: 1.0 for n in range(1, 44)}

    @pytest.fixture
    def uniform_weights_loto7(self) -> dict[int, float]:
        """ロト7用の均一重み"""
        return {n: 1.0 for n in range(1, 38)}

    @pytest.fixture
    def uniform_weights_miniloto(self) -> dict[int, float]:
        """ミニロト用の均一重み"""
        return {n: 1.0 for n in range(1, 32)}

    def test_result_count(self, uniform_weights_loto6):
        """結果の件数が試行回数と一致すること"""
        trials = 100
        sim = MonteCarloSimulator("LOTO6", uniform_weights_loto6, trials=trials)
        results = sim.run()
        assert len(results) == trials

    def test_loto6_pick_size(self, uniform_weights_loto6):
        """ロト6の各結果が6個の数字を含むこと"""
        sim = MonteCarloSimulator("LOTO6", uniform_weights_loto6, trials=50)
        results = sim.run()
        for combo in results:
            assert len(combo) == 6, f"ロト6は6個: {combo}"

    def test_loto7_pick_size(self, uniform_weights_loto7):
        """ロト7の各結果が7個の数字を含むこと"""
        sim = MonteCarloSimulator("LOTO7", uniform_weights_loto7, trials=50)
        results = sim.run()
        for combo in results:
            assert len(combo) == 7, f"ロト7は7個: {combo}"

    def test_miniloto_pick_size(self, uniform_weights_miniloto):
        """ミニロトの各結果が5個の数字を含むこと"""
        sim = MonteCarloSimulator("MINILOTO", uniform_weights_miniloto, trials=50)
        results = sim.run()
        for combo in results:
            assert len(combo) == 5, f"ミニロトは5個: {combo}"

    def test_numbers_in_range(self, uniform_weights_loto6):
        """数字がゲームの有効範囲内であること"""
        sim = MonteCarloSimulator("LOTO6", uniform_weights_loto6, trials=100)
        results = sim.run()
        for combo in results:
            for num in combo:
                assert 1 <= num <= 43, f"範囲外の数字: {num}"

    def test_no_duplicates(self, uniform_weights_loto6):
        """各結果に重複数字がないこと"""
        sim = MonteCarloSimulator("LOTO6", uniform_weights_loto6, trials=100)
        results = sim.run()
        for combo in results:
            assert len(set(combo)) == len(combo), f"重複あり: {combo}"

    def test_results_sorted(self, uniform_weights_loto6):
        """各結果が昇順ソートされていること"""
        sim = MonteCarloSimulator("LOTO6", uniform_weights_loto6, trials=100)
        results = sim.run()
        for combo in results:
            assert combo == tuple(sorted(combo)), f"未ソート: {combo}"

    def test_weighted_bias(self):
        """極端な重み付けが結果に反映されること"""
        # 数字1に極端に大きな重みをつける
        weights = {n: 0.001 for n in range(1, 44)}
        weights[1] = 1000.0
        weights[2] = 1000.0
        weights[3] = 1000.0
        weights[4] = 1000.0
        weights[5] = 1000.0
        weights[6] = 1000.0

        sim = MonteCarloSimulator("LOTO6", weights, trials=100)
        results = sim.run()

        # 高重みの数字（1〜6）が大半の結果に含まれるはず
        count_with_1 = sum(1 for combo in results if 1 in combo)
        assert count_with_1 > 80, f"数字1の出現が少なすぎる: {count_with_1}/100"

    def test_invalid_game_key(self):
        """不正なゲームキーでValueError"""
        with pytest.raises(ValueError, match="不正なゲームキー"):
            MonteCarloSimulator("INVALID", {1: 1.0})

    def test_progress_callback(self, uniform_weights_loto6):
        """進行状況コールバックが正しく呼ばれること"""
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        sim = MonteCarloSimulator("LOTO6", uniform_weights_loto6, trials=100)
        sim.run(progress_callback=callback, progress_interval=25)

        # 100試行を25間隔 → 4回呼ばれるはず
        assert len(progress_calls) == 4
        assert progress_calls[-1] == (100, 100)
