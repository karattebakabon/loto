"""
ロト予測ツール - 重み計算モジュール

過去の当選データから各数字の出現頻度を分析し、
シミュレーション用の重み辞書を生成する。
"""

from collections import Counter
from typing import Optional

from src.common import LOTTERY_CONFIG


def calculate_frequency_weights(
    data: list[dict],
    game_key: str,
    recent_n: Optional[int] = None,
    bonus_weight: float = 0.3,
) -> dict[int, float]:
    """
    過去の当選データから各数字の出現頻度ベースの重みを計算する。

    重みの計算式:
        weight[num] = 基本重み(1.0)
                    + 本数字としての出現回数 × 1.0
                    + ボーナス数字としての出現回数 × bonus_weight

    Args:
        data: load_lottery_data() の戻り値（開催回昇順を想定）
        game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）
        recent_n: 直近N回のみ使用（None=全データ）
        bonus_weight: ボーナス数字出現1回あたりの重み加算値

    Returns:
        {数字: 重み} の辞書。全数字（1〜range_max）を含む。
    """
    game_key = game_key.upper()
    if game_key not in LOTTERY_CONFIG:
        raise ValueError(f"不正なゲームキー: '{game_key}' (有効: {', '.join(LOTTERY_CONFIG.keys())})")

    config = LOTTERY_CONFIG[game_key]
    range_max = config["range_max"]

    # 直近N回に絞る（データは開催回昇順を想定）
    target_data = data[-recent_n:] if recent_n is not None else data

    # 出現回数をカウント
    main_counter: Counter = Counter()
    bonus_counter: Counter = Counter()

    for draw in target_data:
        main_counter.update(draw["main_numbers"])
        bonus_counter.update(draw["bonus_numbers"])

    # 重み辞書の構築
    weights: dict[int, float] = {}
    for num in range(1, range_max + 1):
        weights[num] = (
            1.0  # 基本重み
            + main_counter[num] * 1.0  # 本数字出現による加算
            + bonus_counter[num] * bonus_weight  # ボーナス数字出現による加算
        )

    return weights
