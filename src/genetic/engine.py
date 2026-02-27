"""
ロト予測ツール - 遺伝的アルゴリズム エンジン

進化のメインループを管理する。
世代ごとに「選択→交叉→突然変異→評価→エリート保存」を繰り返し、
最も適応度の高い個体群を返す。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.common import LOTTERY_CONFIG
from src.genetic.fitness import (
    FitnessConfig,
    HistoricalStats,
    calculate_fitness,
    calculate_fitness_detail,
    compute_historical_stats,
)
from src.genetic.operators import (
    generate_initial_population,
    tournament_select,
    order_crossover,
    mutate,
)


@dataclass
class GAConfig:
    """GA パラメータ設定"""

    population_size: int = 200
    """個体群のサイズ"""

    generations: int = 500
    """世代数"""

    crossover_rate: float = 0.8
    """交叉率"""

    mutation_rate: float = 0.1
    """突然変異率"""

    elite_count: int = 10
    """エリート保存数"""

    tournament_size: int = 5
    """トーナメント選択のサイズ"""


@dataclass
class GAResult:
    """GA 実行結果"""

    best_individuals: list[tuple[int, ...]]
    """最良個体のリスト（上位N体、適応度降順）"""

    best_fitness: float
    """最良個体の適応度"""

    best_fitness_detail: dict[str, float]
    """最良個体の適応度詳細（各評価軸のスコア）"""

    fitness_history: list[dict[str, float]]
    """世代ごとの適応度推移 [{"best": float, "average": float, "worst": float}, ...]"""

    final_population: list[tuple[int, ...]]
    """最終世代の全個体"""

    final_fitnesses: list[float]
    """最終世代の全適応度"""

    generations_run: int
    """実際に実行した世代数"""

    ga_config: GAConfig = field(default_factory=GAConfig)
    """使用したGA設定"""


class GeneticEngine:
    """
    遺伝的アルゴリズムによるロト番号予測エンジン。

    使用例:
        >>> from src.common.data_loader import load_lottery_data
        >>> data = load_lottery_data("LOTO6")
        >>> engine = GeneticEngine("LOTO6", data)
        >>> result = engine.run(n_predictions=5)
    """

    def __init__(
        self,
        game_key: str,
        data: list[dict],
        ga_config: Optional[GAConfig] = None,
        fitness_config: Optional[FitnessConfig] = None,
        recent_n: Optional[int] = None,
    ) -> None:
        """
        Args:
            game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）
            data: load_lottery_data() の戻り値
            ga_config: GAパラメータ（None=デフォルト）
            fitness_config: 適応度関数の設定（None=デフォルト）
            recent_n: 統計計算に使う直近N回（None=全データ）
        """
        game_key = game_key.upper()
        if game_key not in LOTTERY_CONFIG:
            raise ValueError(f"不正なゲームキー: '{game_key}' (有効: {', '.join(LOTTERY_CONFIG.keys())})")

        self.game_key = game_key
        self.config = LOTTERY_CONFIG[game_key]
        self.data = data
        self.ga_config = ga_config or GAConfig()
        self.fitness_config = fitness_config or FitnessConfig()

        # 過去データから統計値を事前計算
        self.stats = compute_historical_stats(data, game_key, recent_n=recent_n)

    def run(
        self,
        n_predictions: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        progress_interval: int = 50,
    ) -> GAResult:
        """
        GA を実行し、予測結果を返す。

        Args:
            n_predictions: 最終的に返す予測番号セット数
            progress_callback: 進行状況通知関数 fn(current_gen, total_gen)
            progress_interval: コールバック呼び出し間隔（世代数）

        Returns:
            GAResult オブジェクト
        """
        gc = self.ga_config

        # 1. 初期個体群の生成
        population = generate_initial_population(self.game_key, gc.population_size)

        # 初期個体群の適応度を評価
        fitnesses = self._evaluate(population)

        # 世代ごとの適応度推移を記録
        fitness_history: list[dict[str, float]] = []

        # 2. 世代ループ
        for gen in range(gc.generations):
            # 適応度の統計を記録
            fitness_history.append(
                {
                    "best": max(fitnesses),
                    "average": sum(fitnesses) / len(fitnesses),
                    "worst": min(fitnesses),
                }
            )

            # エリート保存: 上位N体をそのまま次世代に残す
            elite_indices = sorted(
                range(len(fitnesses)),
                key=lambda i: fitnesses[i],
                reverse=True,
            )[: gc.elite_count]
            next_population: list[tuple[int, ...]] = [population[i] for i in elite_indices]

            # 残りの個体を交叉+突然変異で生成
            while len(next_population) < gc.population_size:
                # 選択
                parent1 = tournament_select(population, fitnesses, gc.tournament_size)
                parent2 = tournament_select(population, fitnesses, gc.tournament_size)

                # 交叉
                import random

                if random.random() < gc.crossover_rate:
                    child1, child2 = order_crossover(parent1, parent2, self.game_key)
                else:
                    child1, child2 = parent1, parent2

                # 突然変異
                child1 = mutate(child1, self.game_key, gc.mutation_rate)
                child2 = mutate(child2, self.game_key, gc.mutation_rate)

                next_population.append(child1)
                if len(next_population) < gc.population_size:
                    next_population.append(child2)

            population = next_population
            fitnesses = self._evaluate(population)

            # 進行状況の通知
            if progress_callback and (gen + 1) % progress_interval == 0:
                progress_callback(gen + 1, gc.generations)

        # 最終世代の適応度も記録
        fitness_history.append(
            {
                "best": max(fitnesses),
                "average": sum(fitnesses) / len(fitnesses),
                "worst": min(fitnesses),
            }
        )

        # 3. 結果の構築
        # 適応度上位N体を取得（重複を除いてユニークな組み合わせを選ぶ）
        ranked = sorted(
            zip(population, fitnesses),
            key=lambda x: x[1],
            reverse=True,
        )

        seen: set[tuple[int, ...]] = set()
        best_individuals: list[tuple[int, ...]] = []
        for individual, _fitness in ranked:
            if individual not in seen:
                seen.add(individual)
                best_individuals.append(individual)
            if len(best_individuals) >= n_predictions:
                break

        # 最良個体の詳細スコア
        best = best_individuals[0] if best_individuals else population[0]
        best_detail = calculate_fitness_detail(best, self.game_key, self.stats, self.fitness_config)

        return GAResult(
            best_individuals=best_individuals,
            best_fitness=max(fitnesses),
            best_fitness_detail=best_detail,
            fitness_history=fitness_history,
            final_population=population,
            final_fitnesses=fitnesses,
            generations_run=gc.generations,
            ga_config=gc,
        )

    def _evaluate(self, population: list[tuple[int, ...]]) -> list[float]:
        """個体群全体の適応度を計算する。"""
        return [
            calculate_fitness(
                individual,
                self.game_key,
                self.stats,
                self.fitness_config,
            )
            for individual in population
        ]
