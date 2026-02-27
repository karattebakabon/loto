"""
ロト予測ツール - 遺伝的アルゴリズム レポートモジュール

GA の進化過程と予測結果をコンソールに出力する。
"""

from src.genetic.engine import GAResult


# 適応度の評価軸名（日本語表示用）
_SCORE_LABELS: dict[str, str] = {
    "sum": "合計値",
    "odd_even": "奇偶バランス",
    "high_low": "高低バランス",
    "consecutive": "連続数字",
    "frequency": "出現頻度",
    "total": "総合スコア",
}


def print_evolution_report(
    result: GAResult,
    config: dict,
) -> None:
    """
    GA の進化過程のサマリーをコンソールに出力する。

    Args:
        result: GA 実行結果
        config: LOTTERY_CONFIG[game_key]
    """
    game_name = config["name"]
    gc = result.ga_config
    history = result.fitness_history

    print()
    print("=" * 60)
    print(f"  🧬 {game_name} 遺伝的アルゴリズム 進化レポート")
    print("=" * 60)
    print(f"  個体群サイズ: {gc.population_size}")
    print(f"  世代数: {result.generations_run}")
    print(f"  交叉率: {gc.crossover_rate}")
    print(f"  突然変異率: {gc.mutation_rate}")
    print(f"  エリート保存: {gc.elite_count}体")
    print()

    # ── 進化の推移 ──
    print(f"  【進化の推移】")
    print(f"  {'世代':>6}  {'最良':>8}  {'平均':>8}  {'最悪':>8}")
    print(f"  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    # 始点、中間点、終点をサンプリング表示
    n = len(history)
    if n <= 10:
        sample_indices = list(range(n))
    else:
        # 等間隔で最大10点をサンプル + 最終世代
        step = max(1, n // 9)
        sample_indices = list(range(0, n, step))
        if sample_indices[-1] != n - 1:
            sample_indices.append(n - 1)

    for i in sample_indices:
        gen_label = f"第{i}世代" if i < n - 1 else "最終"
        best = history[i]["best"]
        avg = history[i]["average"]
        worst = history[i]["worst"]
        print(f"  {gen_label:>6}  {best:>8.4f}  {avg:>8.4f}  {worst:>8.4f}")

    print()

    # ── 最良個体の詳細スコア ──
    print(f"  【最良個体の適応度内訳】")
    detail = result.best_fitness_detail
    for key in ["sum", "odd_even", "high_low", "consecutive", "frequency"]:
        label = _SCORE_LABELS.get(key, key)
        score = detail.get(key, 0.0)
        bar = "█" * int(score * 20)
        print(f"    {label:<10}: {score:.4f}  {bar}")

    print(f"    {'─' * 40}")
    total_score = detail.get("total", 0.0)
    print(f"    {'総合スコア':<10}: {total_score:.4f}  ★")
    print()

    # 改善率
    if n >= 2:
        initial_best = history[0]["best"]
        final_best = history[-1]["best"]
        if initial_best > 0:
            improvement = ((final_best - initial_best) / initial_best) * 100
            print(f"  📈 適応度改善率: {improvement:+.1f}%（第0世代 → 最終世代）")
        else:
            print(f"  📈 最終適応度: {final_best:.4f}")
    print()
    print("=" * 60)


def print_prediction_report(
    predictions: list[tuple[int, ...]],
    config: dict,
) -> None:
    """
    予測結果をコンソールに出力する。

    Args:
        predictions: 予測番号のリスト
        config: LOTTERY_CONFIG[game_key]
    """
    game_name = config["name"]

    print()
    print("=" * 60)
    print(f"  🎯 {game_name} 遺伝的アルゴリズム 予測結果")
    print("=" * 60)
    print()

    for i, numbers in enumerate(predictions, 1):
        nums_str = " - ".join(f"{n:2d}" for n in numbers)
        print(f"  予測{i:>2}: [ {nums_str} ]")

    print()
    print("=" * 60)
