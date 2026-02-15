"""
ロト予測ツール - CSVデータ読み込みモジュール

過去の当選番号CSVファイルを読み込み、
統一された辞書形式に変換する。

データソース: loto-life.net (Shift-JIS)
"""

import csv
import os
from typing import Optional

from src.common import LOTTERY_CONFIG

# ゲームキーからCSVファイル名へのマッピング
_CSV_FILENAMES: dict[str, str] = {
    "LOTO6": "loto6.csv",
    "LOTO7": "loto7.csv",
    "MINILOTO": "miniloto.csv",
}

# CSVカラムインデックス定義（0始まり）
# 共通: 列0=開催回, 列1=開催日
_COLUMN_MAP: dict[str, dict] = {
    "LOTO6": {
        "main_start": 2,   # 第1数字
        "main_end": 8,     # 第6数字（排他）
        "bonus_start": 8,  # ボーナス数字
        "bonus_end": 9,    # （排他）
    },
    "LOTO7": {
        "main_start": 2,   # 第1数字
        "main_end": 9,     # 第7数字（排他）
        "bonus_start": 9,  # ボーナス数字1
        "bonus_end": 11,   # ボーナス数字2（排他）
    },
    "MINILOTO": {
        "main_start": 2,   # 第1数字
        "main_end": 7,     # 第5数字（排他）
        "bonus_start": 7,  # ボーナス数字
        "bonus_end": 8,    # （排他）
    },
}


def _get_data_dir() -> str:
    """データディレクトリのパスを返す"""
    # プロジェクトルート/data/raw/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "raw")


def load_lottery_data(
    game_key: str,
    data_dir: Optional[str] = None,
) -> list[dict]:
    """
    指定ゲームのCSVファイルを読み込み、当選番号データを返す。

    Args:
        game_key: ゲームキー（"LOTO6", "LOTO7", "MINILOTO"）
        data_dir: CSVファイルのディレクトリパス（省略時はデフォルト）

    Returns:
        各開催回のデータを格納した辞書のリスト:
        [
            {
                "draw_no": int,          # 開催回
                "date": str,             # 開催日（YYYY-MM-DD）
                "main_numbers": [int],   # 本数字（昇順）
                "bonus_numbers": [int],  # ボーナス数字
            },
            ...
        ]

    Raises:
        ValueError: 不正なゲームキーが指定された場合
        FileNotFoundError: CSVファイルが見つからない場合
    """
    # ゲームキーの検証
    game_key = game_key.upper()
    if game_key not in LOTTERY_CONFIG:
        raise ValueError(
            f"不正なゲームキー: '{game_key}' "
            f"(有効: {', '.join(LOTTERY_CONFIG.keys())})"
        )

    # ファイルパスの組み立て
    if data_dir is None:
        data_dir = _get_data_dir()
    csv_path = os.path.join(data_dir, _CSV_FILENAMES[game_key])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    # カラム定義の取得
    col = _COLUMN_MAP[game_key]

    # CSVの読み込み（Shift-JIS）
    results: list[dict] = []
    with open(csv_path, encoding="shift_jis", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダー行をスキップ

        for row in reader:
            # 空行やデータ不足行をスキップ
            if len(row) < col["bonus_end"]:
                continue

            try:
                draw_no = int(row[0])
                date_str = row[1]
                main_numbers = sorted(
                    int(row[i]) for i in range(col["main_start"], col["main_end"])
                )
                bonus_numbers = sorted(
                    int(row[i]) for i in range(col["bonus_start"], col["bonus_end"])
                )
            except (ValueError, IndexError):
                # 数値変換に失敗した行はスキップ
                continue

            results.append({
                "draw_no": draw_no,
                "date": date_str,
                "main_numbers": main_numbers,
                "bonus_numbers": bonus_numbers,
            })

    return results
