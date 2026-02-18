"""
ロト予測ツール - クラスタリング分析 特徴量抽出モジュール

各開催回の当選番号セットから6次元の特徴量ベクトルを生成する。

特徴量:
    1. sum_total  — 本数字の合計値
    2. mean       — 本数字の平均値
    3. std_dev    — 本数字の標準偏差
    4. range_span — 最大値 − 最小値
    5. odd_ratio  — 奇数の割合（0.0〜1.0）
    6. high_ratio — 上半分（≥中央値）の割合（0.0〜1.0）
"""

import math

import numpy as np

from src.common import LOTTERY_CONFIG

# 特徴量名（レポート・可視化での表示用）
FEATURE_NAMES: list[str] = [
    "合計値",
    "平均値",
    "標準偏差",
    "範囲幅",
    "奇数率",
    "上半分率",
]


def get_feature_names() -> list[str]:
    """特徴量名のリストを返す"""
    return FEATURE_NAMES.copy()


def _extract_one(numbers: list[int], range_max: int) -> list[float]:
    """
    1開催回分の本数字から6次元の特徴量を計算する。

    Args:
        numbers: 本数字のリスト（昇順）
        range_max: 数字の最大値（ゲームの上限）

    Returns:
        6要素のfloatリスト
    """
    n = len(numbers)
    if n == 0:
        return [0.0] * 6

    # 1. 合計値
    sum_total = float(sum(numbers))

    # 2. 平均値
    mean = sum_total / n

    # 3. 標準偏差（母集団）
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = math.sqrt(variance)

    # 4. 範囲幅（最大 − 最小）
    range_span = float(numbers[-1] - numbers[0])

    # 5. 奇数の割合
    odd_count = sum(1 for x in numbers if x % 2 == 1)
    odd_ratio = odd_count / n

    # 6. 上半分（≥ 中央値）の割合
    # 中央値 = (range_max + 1) / 2（例: ロト6なら22）
    midpoint = (range_max + 1) / 2
    high_count = sum(1 for x in numbers if x >= midpoint)
    high_ratio = high_count / n

    return [sum_total, mean, std_dev, range_span, odd_ratio, high_ratio]


def extract_features(
    data: list[dict],
    game_key: str,
) -> np.ndarray:
    """
    当選データリストから特徴量行列を生成する。

    Args:
        data: load_lottery_data() の戻り値
        game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）

    Returns:
        numpy.ndarray — shape=(len(data), 6) の特徴量行列

    Raises:
        ValueError: 不正なゲームキーまたは空データ
    """
    game_key = game_key.upper()
    if game_key not in LOTTERY_CONFIG:
        raise ValueError(
            f"不正なゲームキー: '{game_key}' "
            f"(有効: {', '.join(LOTTERY_CONFIG.keys())})"
        )

    if not data:
        raise ValueError("データが空です")

    config = LOTTERY_CONFIG[game_key]
    range_max = config["range_max"]

    features = []
    for draw in data:
        row = _extract_one(draw["main_numbers"], range_max)
        features.append(row)

    return np.array(features, dtype=np.float64)
