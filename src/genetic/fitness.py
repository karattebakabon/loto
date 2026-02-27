"""
ロト予測ツール - 遺伝的アルゴリズム 適応度関数

個体（数字の組み合わせ）の「良さ」を 0.0〜1.0 のスコアで評価する。
5つの評価軸を重み付き合計して、複合的な適応度を算出する。

評価軸:
    1. 合計値スコア     — 過去の平均合計値に近いほど高評価
    2. 奇偶バランス     — 奇数・偶数が半々に近いほど高評価
    3. 高低バランス     — 上半分・下半分が均等なほど高評価
    4. 連続数字ペナルティ — 連続する数字が少ないほど高評価
    5. 出現頻度スコア   — 過去の頻出数字を多く含むほど高評価
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.common import LOTTERY_CONFIG


@dataclass
class FitnessConfig:
    """適応度関数の設定"""

    # 各評価軸の重み（合計 1.0 を推奨）
    weight_sum: float = 0.25
    """合計値スコアの重み"""

    weight_odd_even: float = 0.20
    """奇偶バランスの重み"""

    weight_high_low: float = 0.20
    """高低バランスの重み"""

    weight_consecutive: float = 0.15
    """連続数字ペナルティの重み"""

    weight_frequency: float = 0.20
    """出現頻度スコアの重み"""


@dataclass
class HistoricalStats:
    """過去データから事前計算した統計値"""

    mean_sum: float
    """過去の合計値の平均"""

    std_sum: float
    """過去の合計値の標準偏差"""

    frequency: dict[int, float] = field(default_factory=dict)
    """各数字の正規化された出現頻度（0.0〜1.0）"""


def compute_historical_stats(
    data: list[dict],
    game_key: str,
    recent_n: Optional[int] = None,
) -> HistoricalStats:
    """
    過去の当選データから、適応度計算に必要な統計値を事前計算する。

    Args:
        data: load_lottery_data() の戻り値
        game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）
        recent_n: 直近N回のみ使用（None=全データ）

    Returns:
        HistoricalStats オブジェクト
    """
    game_key = game_key.upper()
    config = LOTTERY_CONFIG[game_key]
    range_max = config["range_max"]

    # 直近N回に絞る
    target_data = data[-recent_n:] if recent_n is not None else data

    if not target_data:
        # データが空の場合: 理論上の中央値を使用
        pick_size = config["pick_size"]
        theoretical_mean = pick_size * (range_max + 1) / 2
        return HistoricalStats(
            mean_sum=theoretical_mean,
            std_sum=theoretical_mean * 0.1,  # 10%を仮の標準偏差とする
            frequency={n: 1.0 / range_max for n in range(1, range_max + 1)},
        )

    # 合計値の統計
    sums = [sum(d["main_numbers"]) for d in target_data]
    mean_sum = sum(sums) / len(sums)
    variance = sum((s - mean_sum) ** 2 for s in sums) / len(sums)
    std_sum = max(variance**0.5, 1.0)  # 0除算防止

    # 数字別の出現頻度
    counter: Counter = Counter()
    for d in target_data:
        counter.update(d["main_numbers"])

    # 正規化（最頻出を 1.0 に）
    max_count = max(counter.values()) if counter else 1
    frequency: dict[int, float] = {}
    for n in range(1, range_max + 1):
        frequency[n] = counter.get(n, 0) / max_count

    return HistoricalStats(
        mean_sum=mean_sum,
        std_sum=std_sum,
        frequency=frequency,
    )


def calculate_fitness(
    chromosome: tuple[int, ...],
    game_key: str,
    stats: HistoricalStats,
    fitness_config: Optional[FitnessConfig] = None,
) -> float:
    """
    個体の適応度を計算する。

    Args:
        chromosome: 数字の組み合わせ（ソート済みタプル）
        game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）
        stats: 事前計算された統計値
        fitness_config: 適応度の重み設定（None=デフォルト）

    Returns:
        適応度スコア（0.0〜1.0）
    """
    if fitness_config is None:
        fitness_config = FitnessConfig()

    config = LOTTERY_CONFIG[game_key.upper()]

    # 各評価軸のスコアを計算
    s_sum = _score_sum(chromosome, stats)
    s_odd_even = _score_odd_even(chromosome)
    s_high_low = _score_high_low(chromosome, config["range_max"])
    s_consecutive = _score_consecutive(chromosome)
    s_frequency = _score_frequency(chromosome, stats)

    # 重み付き合計
    fitness = (
        fitness_config.weight_sum * s_sum
        + fitness_config.weight_odd_even * s_odd_even
        + fitness_config.weight_high_low * s_high_low
        + fitness_config.weight_consecutive * s_consecutive
        + fitness_config.weight_frequency * s_frequency
    )

    return fitness


def calculate_fitness_detail(
    chromosome: tuple[int, ...],
    game_key: str,
    stats: HistoricalStats,
    fitness_config: Optional[FitnessConfig] = None,
) -> dict[str, float]:
    """
    個体の適応度を各評価軸ごとに詳細に返す（可視化・分析用）。

    Returns:
        {
            "sum": float,
            "odd_even": float,
            "high_low": float,
            "consecutive": float,
            "frequency": float,
            "total": float,
        }
    """
    if fitness_config is None:
        fitness_config = FitnessConfig()

    config = LOTTERY_CONFIG[game_key.upper()]

    scores = {
        "sum": _score_sum(chromosome, stats),
        "odd_even": _score_odd_even(chromosome),
        "high_low": _score_high_low(chromosome, config["range_max"]),
        "consecutive": _score_consecutive(chromosome),
        "frequency": _score_frequency(chromosome, stats),
    }

    scores["total"] = (
        fitness_config.weight_sum * scores["sum"]
        + fitness_config.weight_odd_even * scores["odd_even"]
        + fitness_config.weight_high_low * scores["high_low"]
        + fitness_config.weight_consecutive * scores["consecutive"]
        + fitness_config.weight_frequency * scores["frequency"]
    )

    return scores


# ==========================================
# 各評価軸のスコア計算（内部関数）
# ==========================================


def _score_sum(chromosome: tuple[int, ...], stats: HistoricalStats) -> float:
    """
    合計値スコア: 過去の平均合計値からのズレをガウス関数で評価。

    ズレが小さいほど 1.0 に近く、大きいほど 0.0 に近い。
    """
    total = sum(chromosome)
    # ガウス関数: exp(-0.5 * ((x - μ) / σ)^2)
    z = (total - stats.mean_sum) / stats.std_sum
    import math

    return math.exp(-0.5 * z * z)


def _score_odd_even(chromosome: tuple[int, ...]) -> float:
    """
    奇偶バランススコア: 奇数と偶数の比率が均等なほど高評価。

    理想は 50:50。最悪は 100:0 で 0.0。
    """
    n = len(chromosome)
    if n == 0:
        return 0.0
    odd_count = sum(1 for x in chromosome if x % 2 == 1)
    # 理想は n/2 個が奇数
    ideal = n / 2
    deviation = abs(odd_count - ideal) / ideal
    return max(0.0, 1.0 - deviation)


def _score_high_low(chromosome: tuple[int, ...], range_max: int) -> float:
    """
    高低バランススコア: 上半分・下半分の数字の比率が均等なほど高評価。

    境界値 = range_max / 2（切り上げ）
    """
    n = len(chromosome)
    if n == 0:
        return 0.0
    mid = (range_max + 1) / 2  # 例: ロト6→22, ロト7→19, ミニロト→16
    low_count = sum(1 for x in chromosome if x <= mid)
    ideal = n / 2
    deviation = abs(low_count - ideal) / ideal
    return max(0.0, 1.0 - deviation)


def _score_consecutive(chromosome: tuple[int, ...]) -> float:
    """
    連続数字ペナルティ: 連続する数字のペアが多いほど減点。

    連続ペア数 0 → 1.0、全ペア連続 → 0.0。
    """
    if len(chromosome) < 2:
        return 1.0
    sorted_nums = sorted(chromosome)
    consecutive_pairs = sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i + 1] - sorted_nums[i] == 1)
    max_pairs = len(chromosome) - 1
    return 1.0 - (consecutive_pairs / max_pairs)


def _score_frequency(
    chromosome: tuple[int, ...],
    stats: HistoricalStats,
) -> float:
    """
    出現頻度スコア: 過去の頻出数字を多く含むほど高評価。

    各数字の正規化頻度の平均。
    """
    if not chromosome:
        return 0.0
    total_freq = sum(stats.frequency.get(n, 0.0) for n in chromosome)
    return total_freq / len(chromosome)
