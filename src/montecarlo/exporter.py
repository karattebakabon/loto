"""
ロト予測ツール - シミュレーション結果エクスポーター

シミュレーション結果をCSV/JSON形式でファイルに保存する。
"""

import csv
import json
import os
from datetime import datetime
from typing import Optional

from src.montecarlo.analyzer import (
    analyze_top_combinations,
    analyze_number_frequency,
)


def _ensure_output_dir(output_dir: str) -> None:
    """出力ディレクトリが存在しない場合は作成する"""
    os.makedirs(output_dir, exist_ok=True)


def _generate_filename(game_key: str, ext: str) -> str:
    """タイムスタンプ付きのファイル名を生成する"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mc_{game_key.lower()}_{timestamp}.{ext}"


def export_csv(
    results: list[tuple[int, ...]],
    config: dict,
    game_key: str,
    trials: int,
    output_dir: str = "output",
    top_n: int = 50,
    filepath: Optional[str] = None,
) -> str:
    """
    シミュレーション結果をCSVファイルに保存する。

    出力ファイルは2つのセクションを含む:
    1. 頻出組み合わせトップN
    2. 個別数字の出現頻度

    Args:
        results: シミュレーション結果
        config: LOTTERY_CONFIG[game_key]
        game_key: ゲームキー
        trials: 試行回数
        output_dir: 出力ディレクトリ
        top_n: 上位組み合わせの保存件数
        filepath: 出力ファイルパス（省略時は自動生成）

    Returns:
        保存したファイルのパス
    """
    _ensure_output_dir(output_dir)

    if filepath is None:
        filepath = os.path.join(output_dir, _generate_filename(game_key, "csv"))

    range_max = config["range_max"]
    pick_size = config["pick_size"]
    top_combos = analyze_top_combinations(results, top_n)
    freq = analyze_number_frequency(results, range_max)
    total = len(results)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ── メタデータ ──
        writer.writerow(["# メタデータ"])
        writer.writerow(["ゲーム", config["name"]])
        writer.writerow(["試行回数", trials])
        writer.writerow(["実行日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        # ── 頻出組み合わせ ──
        writer.writerow(["# 頻出組み合わせ"])
        header = ["順位"] + [f"数字{i+1}" for i in range(pick_size)] + ["出現回数", "割合(%)"]
        writer.writerow(header)

        for rank, (numbers, count) in enumerate(top_combos, 1):
            pct = (count / total) * 100
            row = [rank] + list(numbers) + [count, f"{pct:.4f}"]
            writer.writerow(row)

        writer.writerow([])

        # ── 個別数字の出現頻度 ──
        writer.writerow(["# 個別数字の出現頻度"])
        writer.writerow(["数字", "出現回数", "割合(%)"])

        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for num, count in sorted_freq:
            pct = (count / total) * 100
            writer.writerow([num, count, f"{pct:.2f}"])

    return filepath


def export_json(
    results: list[tuple[int, ...]],
    config: dict,
    game_key: str,
    trials: int,
    output_dir: str = "output",
    top_n: int = 50,
    filepath: Optional[str] = None,
) -> str:
    """
    シミュレーション結果をJSONファイルに保存する。

    Args:
        results: シミュレーション結果
        config: LOTTERY_CONFIG[game_key]
        game_key: ゲームキー
        trials: 試行回数
        output_dir: 出力ディレクトリ
        top_n: 上位組み合わせの保存件数
        filepath: 出力ファイルパス（省略時は自動生成）

    Returns:
        保存したファイルのパス
    """
    _ensure_output_dir(output_dir)

    if filepath is None:
        filepath = os.path.join(output_dir, _generate_filename(game_key, "json"))

    range_max = config["range_max"]
    top_combos = analyze_top_combinations(results, top_n)
    freq = analyze_number_frequency(results, range_max)
    total = len(results)

    data = {
        "metadata": {
            "game": config["name"],
            "game_key": game_key,
            "range_max": range_max,
            "pick_size": config["pick_size"],
            "trials": trials,
            "timestamp": datetime.now().isoformat(),
        },
        "top_combinations": [
            {
                "rank": rank,
                "numbers": list(numbers),
                "count": count,
                "percentage": round((count / total) * 100, 4),
            }
            for rank, (numbers, count) in enumerate(top_combos, 1)
        ],
        "number_frequency": [
            {
                "number": num,
                "count": count,
                "percentage": round((count / total) * 100, 2),
            }
            for num, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return filepath
