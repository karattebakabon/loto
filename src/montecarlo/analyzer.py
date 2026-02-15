"""
ロト予測ツール - シミュレーション結果分析モジュール

モンテカルロ・シミュレーションの結果を集計・分析し、
コンソールにレポートを出力する。
"""

from collections import Counter


def analyze_top_combinations(
    results: list[tuple[int, ...]],
    top_n: int = 10,
) -> list[tuple[tuple[int, ...], int]]:
    """
    最も頻出した組み合わせトップNを返す。

    Args:
        results: シミュレーション結果（ソート済みタプルのリスト）
        top_n: 上位何件を返すか

    Returns:
        [(組み合わせタプル, 出現回数), ...] のリスト（降順）
    """
    counter = Counter(results)
    return counter.most_common(top_n)


def analyze_number_frequency(
    results: list[tuple[int, ...]],
    range_max: int,
) -> dict[int, int]:
    """
    各数字の出現回数を集計する。

    Args:
        results: シミュレーション結果
        range_max: 数字の最大値

    Returns:
        {数字: 出現回数} の辞書（1〜range_max）
    """
    counter: Counter = Counter()
    for combo in results:
        counter.update(combo)

    # 全数字を含む辞書を返す（出現0回も含む）
    return {num: counter.get(num, 0) for num in range(1, range_max + 1)}


def print_report(
    results: list[tuple[int, ...]],
    config: dict,
    top_n: int = 10,
) -> None:
    """
    シミュレーション結果のレポートをコンソールに出力する。

    Args:
        results: シミュレーション結果
        config: LOTTERY_CONFIG[game_key]
        top_n: 上位組み合わせの表示件数
    """
    total = len(results)
    game_name = config["name"]
    range_max = config["range_max"]

    print()
    print("=" * 60)
    print(f"  🎰 {game_name} モンテカルロ・シミュレーション結果")
    print("=" * 60)
    print(f"  試行回数: {total:,} 回")
    print()

    # ── 頻出組み合わせ ──
    top_combos = analyze_top_combinations(results, top_n)
    print(f"  【頻出組み合わせ トップ{top_n}】")
    print(f"  {'順位':>4}  {'組み合わせ':<30}  {'出現回数':>8}  {'割合':>8}")
    print(f"  {'─' * 4}  {'─' * 30}  {'─' * 8}  {'─' * 8}")
    for rank, (numbers, count) in enumerate(top_combos, 1):
        nums_str = " - ".join(f"{n:2d}" for n in numbers)
        pct = (count / total) * 100
        print(f"  {rank:>4}  {nums_str:<30}  {count:>8,}  {pct:>7.4f}%")

    print()

    # ── 個別数字の出現頻度 ──
    freq = analyze_number_frequency(results, range_max)
    # 出現回数で降順ソート
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    print(f"  【数字別出現頻度 トップ10 / ワースト10】")
    print()

    # トップ10
    print(f"  ▲ よく出る数字:")
    for num, count in sorted_freq[:10]:
        bar = "█" * int(count / max(freq.values()) * 20)
        pct = (count / total) * 100
        print(f"    {num:>2}: {count:>8,} ({pct:>5.2f}%) {bar}")

    print()

    # ワースト10
    print(f"  ▼ あまり出ない数字:")
    for num, count in sorted_freq[-10:]:
        bar = "█" * int(count / max(freq.values()) * 20)
        pct = (count / total) * 100
        print(f"    {num:>2}: {count:>8,} ({pct:>5.2f}%) {bar}")

    print()
    print("=" * 60)
