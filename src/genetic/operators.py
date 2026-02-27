"""
ロト予測ツール - 遺伝的アルゴリズム 遺伝的オペレータ

初期個体生成、選択、交叉、突然変異を提供する。
すべてのオペレータはゲームの制約（数字数・範囲・重複なし）を保証する。
"""

import random

from src.common import LOTTERY_CONFIG


def generate_random_individual(game_key: str) -> tuple[int, ...]:
    """
    ゲームの制約に従って、ランダムな個体を1体生成する。

    Args:
        game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）

    Returns:
        ソート済み整数タプル（例: (3, 7, 15, 22, 31, 40)）
    """
    config = LOTTERY_CONFIG[game_key.upper()]
    numbers = random.sample(range(1, config["range_max"] + 1), config["pick_size"])
    return tuple(sorted(numbers))


def generate_initial_population(
    game_key: str,
    population_size: int,
) -> list[tuple[int, ...]]:
    """
    初期個体群を生成する。

    Args:
        game_key: ゲームキー
        population_size: 個体群のサイズ

    Returns:
        個体のリスト
    """
    return [generate_random_individual(game_key) for _ in range(population_size)]


def tournament_select(
    population: list[tuple[int, ...]],
    fitnesses: list[float],
    tournament_size: int = 5,
) -> tuple[int, ...]:
    """
    トーナメント選択: 個体群からランダムにk個体を選び、最も適応度が高い個体を返す。

    Args:
        population: 個体群
        fitnesses: 各個体の適応度（populationと同じ順序）
        tournament_size: トーナメントに参加する個体数

    Returns:
        選ばれた個体
    """
    # ランダムにk個のインデックスを選ぶ（重複あり）
    k = min(tournament_size, len(population))
    candidates = random.sample(range(len(population)), k)

    # 最も適応度が高い個体を選択
    best_idx = max(candidates, key=lambda i: fitnesses[i])
    return population[best_idx]


def order_crossover(
    parent1: tuple[int, ...],
    parent2: tuple[int, ...],
    game_key: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    順序交叉（OX: Order Crossover）: 重複のない整数組み合わせに適した交叉手法。

    手順:
        1. 親1からランダムな区間の数字を子1にコピー
        2. 親2から、まだ使われていない数字を順に埋める
        3. 逆も同様にして子2を生成

    Args:
        parent1: 親個体1
        parent2: 親個体2
        game_key: ゲームキー

    Returns:
        (子個体1, 子個体2) の2体
    """
    size = len(parent1)

    # 交叉区間をランダムに決定
    start, end = sorted(random.sample(range(size), 2))

    child1 = _do_ox(parent1, parent2, start, end, game_key)
    child2 = _do_ox(parent2, parent1, start, end, game_key)

    return child1, child2


def _do_ox(
    donor: tuple[int, ...],
    filler: tuple[int, ...],
    start: int,
    end: int,
    game_key: str,
) -> tuple[int, ...]:
    """
    OX交叉の実行（片方向）。

    donorから区間[start, end]の数字を保持し、
    fillerから残りの数字を順番に埋める。
    結果はソートして返す。
    """
    # donorの区間を保持
    preserved = set(donor[start : end + 1])

    # fillerからpreserved以外の数字を順番に取得
    remaining = [n for n in filler if n not in preserved]

    # 子の構築: preserved + 必要な数だけremainingから取る
    needed = len(donor) - len(preserved)
    child_set = preserved | set(remaining[:needed])

    return tuple(sorted(child_set))


def mutate(
    individual: tuple[int, ...],
    game_key: str,
    mutation_rate: float = 0.1,
) -> tuple[int, ...]:
    """
    突然変異: 一定確率で1つの数字を、範囲内の未使用数字に置換する。

    Args:
        individual: 個体
        game_key: ゲームキー
        mutation_rate: 突然変異率（0.0〜1.0）

    Returns:
        突然変異後の個体（変異しなかった場合はそのまま返す）
    """
    if random.random() >= mutation_rate:
        return individual

    config = LOTTERY_CONFIG[game_key.upper()]
    range_max = config["range_max"]
    nums = list(individual)

    # 置換対象をランダムに1つ選ぶ
    idx = random.randrange(len(nums))

    # 使用されていない数字の候補リスト
    used = set(nums)
    available = [n for n in range(1, range_max + 1) if n not in used]

    if available:
        nums[idx] = random.choice(available)

    return tuple(sorted(nums))
