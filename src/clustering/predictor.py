"""
ロト予測ツール - クラスタリング分析 予測生成モジュール

クラスタリング結果から予測番号セットを生成する。

戦略:
    - centroid: 最大クラスタの重心に最も近い実データから選出
    - recent:   直近の開催回が属するクラスタから重み付きサンプリング
    - pocket:   クラスタ間の空白領域を探索して番号生成
"""

import random
from collections import Counter
from typing import Optional

import numpy as np

from src.common import LOTTERY_CONFIG
from src.clustering.engine import ClusteringResult


def generate_predictions(
    result: ClusteringResult,
    data: list[dict],
    features: np.ndarray,
    game_key: str,
    n_predictions: int = 5,
    strategy: str = "centroid",
) -> list[tuple[int, ...]]:
    """
    クラスタリング結果から予測番号セットを生成する。

    Args:
        result: クラスタリング結果
        data: 当選データリスト（load_lottery_data()の戻り値）
        features: 特徴量行列（extract_features()の戻り値）
        game_key: ゲームキー
        n_predictions: 生成する予測セット数
        strategy: 予測戦略（"centroid" / "recent" / "pocket"）

    Returns:
        予測番号のソート済みタプルのリスト

    Raises:
        ValueError: 不正なゲームキーまたは戦略
    """
    game_key = game_key.upper()
    if game_key not in LOTTERY_CONFIG:
        raise ValueError(f"不正なゲームキー: '{game_key}' (有効: {', '.join(LOTTERY_CONFIG.keys())})")

    if strategy not in ("centroid", "recent", "pocket"):
        raise ValueError(f"不正な戦略: '{strategy}' (有効: centroid, recent, pocket)")

    if result.n_clusters == 0:
        # クラスタが検出されなかった場合は全データからランダム選出
        return _fallback_random(data, game_key, n_predictions)

    if strategy == "centroid":
        return _predict_centroid(result, data, features, game_key, n_predictions)
    elif strategy == "recent":
        return _predict_recent(result, data, features, game_key, n_predictions)
    else:
        return _predict_pocket(result, data, features, game_key, n_predictions)


def _predict_centroid(
    result: ClusteringResult,
    data: list[dict],
    features: np.ndarray,
    game_key: str,
    n_predictions: int,
) -> list[tuple[int, ...]]:
    """
    最大クラスタの重心に近い実データから予測番号を生成する。

    重心に最も近いデータ点の当選番号をベースにし、
    頻出数字で調整した番号セットを返す。
    """
    config = LOTTERY_CONFIG[game_key]
    pick_size = config["pick_size"]
    range_max = config["range_max"]

    # 最大クラスタを特定（ノイズ=-1 を除く）
    valid_clusters = {k: v for k, v in result.cluster_sizes.items() if k >= 0}
    if not valid_clusters:
        return _fallback_random(data, game_key, n_predictions)

    largest_cluster = max(valid_clusters, key=valid_clusters.get)

    # 最大クラスタに属するデータのインデックス
    cluster_indices = np.where(result.labels == largest_cluster)[0]
    cluster_data = [data[i] for i in cluster_indices]

    # クラスタ内の数字出現頻度を集計
    number_freq: Counter = Counter()
    for draw in cluster_data:
        number_freq.update(draw["main_numbers"])

    # 頻度の高い数字を重みとして使い、予測を生成
    predictions: list[tuple[int, ...]] = []
    all_numbers = list(range(1, range_max + 1))
    weights = [float(number_freq.get(n, 0) + 1) for n in all_numbers]

    for _ in range(n_predictions):
        picked: set[int] = set()
        while len(picked) < pick_size:
            chosen = random.choices(all_numbers, weights=weights, k=1)[0]
            picked.add(chosen)
        predictions.append(tuple(sorted(picked)))

    return predictions


def _predict_recent(
    result: ClusteringResult,
    data: list[dict],
    features: np.ndarray,
    game_key: str,
    n_predictions: int,
) -> list[tuple[int, ...]]:
    """
    直近の開催回が属するクラスタから予測番号を生成する。

    直近データのクラスタを特定し、同クラスタ内の全データから
    重み付きサンプリングで番号を生成する。
    """
    config = LOTTERY_CONFIG[game_key]
    pick_size = config["pick_size"]
    range_max = config["range_max"]

    # 直近の開催回が属するクラスタ
    recent_label = int(result.labels[-1])

    # ノイズ(-1)の場合は直近5回で最頻のクラスタを使用
    if recent_label == -1:
        recent_labels = result.labels[-5:].tolist()
        valid = [l for l in recent_labels if l >= 0]
        if valid:
            recent_label = Counter(valid).most_common(1)[0][0]
        else:
            return _fallback_random(data, game_key, n_predictions)

    # 同クラスタのデータを取得
    cluster_indices = np.where(result.labels == recent_label)[0]
    cluster_data = [data[i] for i in cluster_indices]

    # 直近のデータほど重みを高くする
    number_freq: Counter = Counter()
    for idx, draw in enumerate(cluster_data):
        # 時系列の後半ほど重みが大きい（1.0〜2.0の線形スケール）
        weight = 1.0 + (idx / max(len(cluster_data) - 1, 1))
        for num in draw["main_numbers"]:
            number_freq[num] += weight

    # 予測生成
    predictions: list[tuple[int, ...]] = []
    all_numbers = list(range(1, range_max + 1))
    weights = [float(number_freq.get(n, 0) + 0.5) for n in all_numbers]

    for _ in range(n_predictions):
        picked: set[int] = set()
        while len(picked) < pick_size:
            chosen = random.choices(all_numbers, weights=weights, k=1)[0]
            picked.add(chosen)
        predictions.append(tuple(sorted(picked)))

    return predictions


def _predict_pocket(
    result: ClusteringResult,
    data: list[dict],
    features: np.ndarray,
    game_key: str,
    n_predictions: int,
) -> list[tuple[int, ...]]:
    """
    クラスタ間の空白領域（ポケット）から予測番号を生成する。

    各クラスタであまり出現しない数字（コールドナンバー）を
    重点的にサンプリングし、「次に出る可能性のある番号」を探す。
    """
    config = LOTTERY_CONFIG[game_key]
    pick_size = config["pick_size"]
    range_max = config["range_max"]

    # 全データの出現頻度を集計
    total_freq: Counter = Counter()
    for draw in data:
        total_freq.update(draw["main_numbers"])

    # 全数字の期待出現回数
    total_draws = len(data)
    expected_freq = total_draws * pick_size / range_max

    # コールドナンバー（期待値以下の出現数字）の重みを高くする
    all_numbers = list(range(1, range_max + 1))
    weights = []
    for n in all_numbers:
        actual = total_freq.get(n, 0)
        if actual < expected_freq:
            # 期待値との差が大きいほど重みを大きくする
            weights.append(expected_freq - actual + 1.0)
        else:
            # ホットナンバーにも少し重みを残す
            weights.append(0.5)

    # 予測生成
    predictions: list[tuple[int, ...]] = []
    for _ in range(n_predictions):
        picked: set[int] = set()
        while len(picked) < pick_size:
            chosen = random.choices(all_numbers, weights=weights, k=1)[0]
            picked.add(chosen)
        predictions.append(tuple(sorted(picked)))

    return predictions


def _fallback_random(
    data: list[dict],
    game_key: str,
    n_predictions: int,
) -> list[tuple[int, ...]]:
    """クラスタが検出されなかった場合のフォールバック（均一ランダム）"""
    config = LOTTERY_CONFIG[game_key]
    pick_size = config["pick_size"]
    range_max = config["range_max"]

    predictions: list[tuple[int, ...]] = []
    all_numbers = list(range(1, range_max + 1))
    for _ in range(n_predictions):
        picked = random.sample(all_numbers, pick_size)
        predictions.append(tuple(sorted(picked)))

    return predictions
