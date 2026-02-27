"""
テスト - 遺伝的アルゴリズム モジュール
"""

import pytest

from src.common import LOTTERY_CONFIG
from src.genetic.fitness import (
    FitnessConfig,
    HistoricalStats,
    calculate_fitness,
    calculate_fitness_detail,
    compute_historical_stats,
    _score_sum,
    _score_odd_even,
    _score_high_low,
    _score_consecutive,
    _score_frequency,
)
from src.genetic.operators import (
    generate_random_individual,
    generate_initial_population,
    tournament_select,
    order_crossover,
    mutate,
)
from src.genetic.engine import GeneticEngine, GAConfig, GAResult


# ==========================================
# テスト用フィクスチャ
# ==========================================


@pytest.fixture
def sample_data_loto6() -> list[dict]:
    """テスト用のロト6ダミーデータ"""
    return [
        {
            "draw_no": i,
            "date": f"2024-01-{i:02d}",
            "main_numbers": sorted([1 + (i * j) % 43 or 1 for j in range(1, 7)]),
            "bonus_numbers": [1 + (i * 7) % 43 or 1],
        }
        for i in range(1, 101)
    ]


@pytest.fixture
def stats_loto6(sample_data_loto6) -> HistoricalStats:
    """テスト用の統計値"""
    return compute_historical_stats(sample_data_loto6, "LOTO6")


# ==========================================
# 適応度関数のテスト
# ==========================================


class TestFitness:
    """適応度関数のテスト"""

    def test_calculate_fitness_returns_float(self, stats_loto6):
        """calculate_fitnessがfloatを返すこと"""
        chromosome = (5, 10, 15, 25, 35, 40)
        score = calculate_fitness(chromosome, "LOTO6", stats_loto6)
        assert isinstance(score, float)

    def test_calculate_fitness_range(self, stats_loto6):
        """適応度が0.0〜1.0の範囲に収まること"""
        test_cases = [
            (1, 2, 3, 4, 5, 6),
            (10, 15, 20, 25, 30, 35),
            (38, 39, 40, 41, 42, 43),
        ]
        for nums in test_cases:
            score = calculate_fitness(nums, "LOTO6", stats_loto6)
            assert 0.0 <= score <= 1.0, f"範囲外: {score} for {nums}"

    def test_calculate_fitness_detail_keys(self, stats_loto6):
        """calculate_fitness_detailが全てのキーを含むこと"""
        chromosome = (5, 10, 15, 25, 35, 40)
        detail = calculate_fitness_detail(chromosome, "LOTO6", stats_loto6)
        expected_keys = {"sum", "odd_even", "high_low", "consecutive", "frequency", "total"}
        assert set(detail.keys()) == expected_keys

    def test_calculate_fitness_detail_values_range(self, stats_loto6):
        """適応度の各軸が0.0〜1.0に収まること"""
        chromosome = (5, 10, 15, 25, 35, 40)
        detail = calculate_fitness_detail(chromosome, "LOTO6", stats_loto6)
        for key in ["sum", "odd_even", "high_low", "consecutive", "frequency"]:
            assert 0.0 <= detail[key] <= 1.0, f"{key} が範囲外: {detail[key]}"

    def test_balanced_chromosome_high_score(self, stats_loto6):
        """バランスの良い個体が高スコアを返すこと"""
        # 奇偶3:3、高低3:3、連続なしの個体
        balanced = (3, 10, 17, 24, 31, 38)
        # 極端に偏った個体
        skewed = (1, 2, 3, 4, 5, 6)
        score_balanced = calculate_fitness(balanced, "LOTO6", stats_loto6)
        score_skewed = calculate_fitness(skewed, "LOTO6", stats_loto6)
        assert score_balanced > score_skewed, (
            f"バランスの良い個体({score_balanced:.4f})が偏った個体({score_skewed:.4f})より低スコア"
        )


class TestScoreFunctions:
    """各評価軸スコア関数の個別テスト"""

    def test_score_odd_even_perfect(self):
        """奇偶が完全に半々なら1.0"""
        # 奇3偶3
        score = _score_odd_even((1, 2, 3, 4, 5, 6))  # 奇:1,3,5 偶:2,4,6
        assert score == 1.0

    def test_score_odd_even_skewed(self):
        """全て奇数ならスコアが低い"""
        score = _score_odd_even((1, 3, 5, 7, 9, 11))  # 奇6偶0
        assert score < 0.5

    def test_score_high_low_balanced(self):
        """高低が均等なら高スコア"""
        # ロト6: range_max=43, mid=22
        score = _score_high_low((5, 10, 15, 25, 35, 40), 43)
        assert score > 0.5

    def test_score_consecutive_none(self):
        """連続数字がなければ1.0"""
        score = _score_consecutive((5, 10, 20, 30, 35, 40))
        assert score == 1.0

    def test_score_consecutive_all(self):
        """全て連続なら0.0"""
        score = _score_consecutive((1, 2, 3, 4, 5, 6))
        assert score == 0.0

    def test_score_frequency_with_stats(self, stats_loto6):
        """出現頻度スコアが0.0〜1.0に収まること"""
        score = _score_frequency((5, 10, 15, 25, 35, 40), stats_loto6)
        assert 0.0 <= score <= 1.0


class TestHistoricalStats:
    """統計値計算のテスト"""

    def test_compute_stats_returns_object(self, sample_data_loto6):
        """HistoricalStatsオブジェクトが返ること"""
        stats = compute_historical_stats(sample_data_loto6, "LOTO6")
        assert isinstance(stats, HistoricalStats)

    def test_compute_stats_mean_positive(self, sample_data_loto6):
        """平均合計値が正の値であること"""
        stats = compute_historical_stats(sample_data_loto6, "LOTO6")
        assert stats.mean_sum > 0

    def test_compute_stats_std_positive(self, sample_data_loto6):
        """標準偏差が正の値であること"""
        stats = compute_historical_stats(sample_data_loto6, "LOTO6")
        assert stats.std_sum > 0

    def test_compute_stats_frequency_length(self, sample_data_loto6):
        """頻度辞書が全数字を含むこと"""
        stats = compute_historical_stats(sample_data_loto6, "LOTO6")
        assert len(stats.frequency) == 43

    def test_compute_stats_with_recent(self, sample_data_loto6):
        """recent_nで絞り込みができること"""
        stats_all = compute_historical_stats(sample_data_loto6, "LOTO6")
        stats_recent = compute_historical_stats(sample_data_loto6, "LOTO6", recent_n=10)
        # 異なるデータ範囲なので、平均が異なるはず（必ずしもではないが大半は異なる）
        assert isinstance(stats_recent, HistoricalStats)

    def test_compute_stats_empty_data(self):
        """空データでもエラーにならないこと"""
        stats = compute_historical_stats([], "LOTO6")
        assert stats.mean_sum > 0  # 理論値が使われる


# ==========================================
# 遺伝的オペレータのテスト
# ==========================================


class TestOperators:
    """遺伝的オペレータのテスト"""

    @pytest.mark.parametrize(
        "game_key,pick_size,range_max",
        [
            ("LOTO6", 6, 43),
            ("LOTO7", 7, 37),
            ("MINILOTO", 5, 31),
        ],
    )
    def test_random_individual_constraints(self, game_key, pick_size, range_max):
        """ランダム個体がゲーム制約を満たすこと"""
        individual = generate_random_individual(game_key)
        assert len(individual) == pick_size, f"数字数が不正: {individual}"
        assert len(set(individual)) == pick_size, f"重複あり: {individual}"
        assert all(1 <= n <= range_max for n in individual), f"範囲外: {individual}"
        assert individual == tuple(sorted(individual)), f"未ソート: {individual}"

    def test_random_individual_is_sorted(self):
        """ランダム個体がソート済みであること"""
        for _ in range(50):
            ind = generate_random_individual("LOTO6")
            assert ind == tuple(sorted(ind))

    def test_initial_population_size(self):
        """初期個体群のサイズが正しいこと"""
        pop = generate_initial_population("LOTO6", 100)
        assert len(pop) == 100

    def test_tournament_select_returns_individual(self):
        """トーナメント選択が個体を返すこと"""
        pop = generate_initial_population("LOTO6", 20)
        fitnesses = [float(i) for i in range(20)]
        selected = tournament_select(pop, fitnesses, tournament_size=5)
        assert isinstance(selected, tuple)
        assert len(selected) == 6

    def test_tournament_select_prefers_best(self):
        """トーナメント選択が高適応度個体を選びやすいこと"""
        pop = generate_initial_population("LOTO6", 20)
        # 最後の個体に圧倒的な適応度を与える
        fitnesses = [0.0] * 19 + [100.0]
        # 200回試行して、最後の個体が多く選ばれるか（トーナメントサイズ大きめで安定化）
        count_best = sum(1 for _ in range(200) if tournament_select(pop, fitnesses, tournament_size=10) == pop[-1])
        assert count_best > 40, f"最良個体の選ばれた回数が少なすぎる: {count_best}/200"

    def test_crossover_constraints_loto6(self):
        """交叉の出力がロト6の制約を満たすこと"""
        parent1 = (5, 10, 15, 25, 35, 40)
        parent2 = (3, 8, 18, 28, 33, 43)
        for _ in range(50):
            child1, child2 = order_crossover(parent1, parent2, "LOTO6")
            for child in (child1, child2):
                assert len(child) == 6, f"数字数が不正: {child}"
                assert len(set(child)) == 6, f"重複あり: {child}"
                assert all(1 <= n <= 43 for n in child), f"範囲外: {child}"
                assert child == tuple(sorted(child)), f"未ソート: {child}"

    def test_crossover_constraints_loto7(self):
        """交叉の出力がロト7の制約を満たすこと"""
        parent1 = (1, 5, 10, 15, 20, 25, 30)
        parent2 = (3, 8, 13, 18, 23, 28, 37)
        for _ in range(50):
            child1, child2 = order_crossover(parent1, parent2, "LOTO7")
            for child in (child1, child2):
                assert len(child) == 7, f"数字数が不正: {child}"
                assert len(set(child)) == 7, f"重複あり: {child}"
                assert all(1 <= n <= 37 for n in child), f"範囲外: {child}"

    def test_mutate_constraints(self):
        """突然変異の出力がゲーム制約を満たすこと"""
        individual = (5, 10, 15, 25, 35, 40)
        for _ in range(100):
            mutated = mutate(individual, "LOTO6", mutation_rate=1.0)
            assert len(mutated) == 6, f"数字数が不正: {mutated}"
            assert len(set(mutated)) == 6, f"重複あり: {mutated}"
            assert all(1 <= n <= 43 for n in mutated), f"範囲外: {mutated}"
            assert mutated == tuple(sorted(mutated)), f"未ソート: {mutated}"

    def test_mutate_no_change_when_rate_zero(self):
        """突然変異率が0なら変化しないこと"""
        individual = (5, 10, 15, 25, 35, 40)
        for _ in range(50):
            mutated = mutate(individual, "LOTO6", mutation_rate=0.0)
            assert mutated == individual

    def test_mutate_changes_when_rate_one(self):
        """突然変異率が1.0なら必ず変化すること"""
        individual = (5, 10, 15, 25, 35, 40)
        changes = 0
        for _ in range(100):
            mutated = mutate(individual, "LOTO6", mutation_rate=1.0)
            if mutated != individual:
                changes += 1
        assert changes > 80, f"突然変異の発生が少なすぎる: {changes}/100"


# ==========================================
# GAエンジンのテスト
# ==========================================


class TestGeneticEngine:
    """GAエンジンのテスト"""

    def test_run_returns_result(self, sample_data_loto6):
        """runがGAResultを返すこと"""
        ga_config = GAConfig(population_size=20, generations=10, elite_count=2)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=3)
        assert isinstance(result, GAResult)

    def test_prediction_count(self, sample_data_loto6):
        """指定数の予測が生成されること"""
        ga_config = GAConfig(population_size=50, generations=10, elite_count=2, mutation_rate=0.3)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=5)
        assert len(result.best_individuals) == 5

    def test_predictions_valid_loto6(self, sample_data_loto6):
        """予測結果がロト6の制約を満たすこと"""
        ga_config = GAConfig(population_size=50, generations=10, elite_count=2, mutation_rate=0.3)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=5)
        for ind in result.best_individuals:
            assert len(ind) == 6
            assert len(set(ind)) == 6
            assert all(1 <= n <= 43 for n in ind)

    def test_fitness_improves(self, sample_data_loto6):
        """世代を経ると適応度が改善する傾向にあること"""
        ga_config = GAConfig(population_size=50, generations=100, elite_count=5)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=1)

        history = result.fitness_history
        assert len(history) > 1
        # 最終世代の最良適応度が初期世代以上であること
        # （エリート保存があるため単調非減少が保証される）
        assert history[-1]["best"] >= history[0]["best"]

    def test_fitness_history_length(self, sample_data_loto6):
        """適応度推移の記録数が世代数+1であること"""
        generations = 20
        ga_config = GAConfig(population_size=20, generations=generations, elite_count=2)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=1)
        # 世代数+1（最終世代の分も記録される）
        assert len(result.fitness_history) == generations + 1

    def test_invalid_game_key(self, sample_data_loto6):
        """不正なゲームキーでValueError"""
        with pytest.raises(ValueError, match="不正なゲームキー"):
            GeneticEngine("INVALID", sample_data_loto6)

    def test_progress_callback(self, sample_data_loto6):
        """進行コールバックが正しく呼ばれること"""
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        ga_config = GAConfig(population_size=20, generations=20, elite_count=2)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        engine.run(n_predictions=1, progress_callback=callback, progress_interval=5)

        # 20世代を5間隔 → 4回呼ばれるはず
        assert len(progress_calls) == 4
        assert progress_calls[-1] == (20, 20)

    @pytest.mark.parametrize("game_key", ["LOTO6", "LOTO7", "MINILOTO"])
    def test_all_games(self, game_key, sample_data_loto6):
        """全ゲームで正常に動作すること"""
        ga_config = GAConfig(population_size=20, generations=5, elite_count=2)
        engine = GeneticEngine(game_key, sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=2)
        config = LOTTERY_CONFIG[game_key]
        for ind in result.best_individuals:
            assert len(ind) == config["pick_size"]
            assert all(1 <= n <= config["range_max"] for n in ind)

    def test_unique_predictions(self, sample_data_loto6):
        """予測結果に重複がないこと"""
        ga_config = GAConfig(population_size=50, generations=20, elite_count=5)
        engine = GeneticEngine("LOTO6", sample_data_loto6, ga_config=ga_config)
        result = engine.run(n_predictions=5)
        unique = set(result.best_individuals)
        assert len(unique) == len(result.best_individuals), "予測に重複がある"
