"""
ロト予測ツール - クラスタリング分析 レポートモジュール

クラスタリング結果と予測番号をコンソールに出力する。
"""

from collections import Counter

import numpy as np

from src.clustering.engine import ClusteringResult
from src.clustering.feature_extractor import FEATURE_NAMES


def print_cluster_report(
    result: ClusteringResult,
    data: list[dict],
    config: dict,
) -> None:
    """
    クラスタリング結果のレポートをコンソールに出力する。

    Args:
        result: クラスタリング結果
        data: 当選データリスト
        config: LOTTERY_CONFIG[game_key]
    """
    game_name = config["name"]
    method_name = "K-Means" if result.method == "kmeans" else "DBSCAN"

    print()
    print("=" * 60)
    print(f"  🔬 {game_name} クラスタリング分析結果（{method_name}）")
    print("=" * 60)
    print(f"  データ数: {len(data):,} 回")
    print(f"  クラスタ数: {result.n_clusters}")
    if result.noise_count > 0:
        print(f"  ノイズ（外れ値）: {result.noise_count} 件")
    if result.inertia is not None:
        print(f"  イナーシャ: {result.inertia:,.2f}")
    print()

    # ── クラスタ別統計 ──
    print(f"  【クラスタ別サマリー】")
    print(f"  {'ID':>4}  {'件数':>6}  {'割合':>7}  {'代表的な番号（最頻出）':>30}")
    print(f"  {'─' * 4}  {'─' * 6}  {'─' * 7}  {'─' * 30}")

    total = len(data)
    for label in sorted(result.cluster_sizes.keys()):
        if label == -1:
            # ノイズ
            count = result.cluster_sizes[label]
            pct = (count / total) * 100
            print(f"  {'N/A':>4}  {count:>6}  {pct:>6.1f}%  （ノイズ — 外れ値）")
            continue

        count = result.cluster_sizes[label]
        pct = (count / total) * 100

        # このクラスタの最頻出数字を取得
        cluster_indices = np.where(result.labels == label)[0]
        number_freq: Counter = Counter()
        for idx in cluster_indices:
            number_freq.update(data[idx]["main_numbers"])

        # 上位数字を表示
        pick_size = config["pick_size"]
        top_nums = [str(n) for n, _ in number_freq.most_common(pick_size)]
        top_str = " - ".join(top_nums)

        print(f"  {label:>4}  {count:>6}  {pct:>6.1f}%  {top_str}")

    print()

    # ── クラスタ重心の特徴量 ──
    if result.centroids is not None and len(result.centroids) > 0:
        print(f"  【クラスタ重心の特徴量】")
        # ヘッダー
        header = f"  {'ID':>4}"
        for name in FEATURE_NAMES:
            header += f"  {name:>8}"
        print(header)
        print(f"  {'─' * 4}" + f"  {'─' * 8}" * len(FEATURE_NAMES))

        for i, centroid in enumerate(result.centroids):
            row = f"  {i:>4}"
            for val in centroid:
                row += f"  {val:>8.2f}"
            print(row)

        print()

    print("=" * 60)


def print_prediction_report(
    predictions: list[tuple[int, ...]],
    config: dict,
    strategy: str,
) -> None:
    """
    予測結果をコンソールに出力する。

    Args:
        predictions: 予測番号のリスト
        config: LOTTERY_CONFIG[game_key]
        strategy: 使用した予測戦略
    """
    game_name = config["name"]
    strategy_names = {
        "centroid": "重心（セントロイド）狙い",
        "recent": "直近クラスタ狙い",
        "pocket": "空白地帯（ポケット）狙い",
    }
    strategy_label = strategy_names.get(strategy, strategy)

    print()
    print("=" * 60)
    print(f"  🎯 {game_name} クラスタリング予測結果")
    print(f"  戦略: {strategy_label}")
    print("=" * 60)
    print()

    for i, numbers in enumerate(predictions, 1):
        nums_str = " - ".join(f"{n:2d}" for n in numbers)
        print(f"  予測{i:>2}: [ {nums_str} ]")

    print()
    print("=" * 60)
