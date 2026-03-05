"""
ロト予測ツール - 答え合わせスクリプト

保存済みの予測JSONと実際の当選番号を照合し、
一致数・的中等級を記録する。

使用方法:
    # 全ゲームの最新予測を照合（CSVから実際の当選番号を自動取得）
    python -m src.verification.verify_results

    # 特定ゲームを指定
    python -m src.verification.verify_results --game loto6

    # 当選番号を手動指定（まだCSVに反映されていない場合）
    python -m src.verification.verify_results --game loto6 --draw 2083 --winning 5 12 20 31 37 42 --bonus 8

    # 予測ファイルを直接指定
    python -m src.verification.verify_results --file data/predictions/prediction_loto6_20260305.json
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Optional, Any

from src.common import LOTTERY_CONFIG

# ==========================================
# 当選等級判定テーブル
# ==========================================

# (本数字一致数, ボーナス一致数) → 等級 (None=落選)
# ロト6 当選条件
_LOTO6_PRIZE: dict[tuple[int, int], str] = {
    (6, 0): "1等",
    (5, 1): "2等",
    (5, 0): "3等",
    (4, 0): "4等",
    (3, 0): "5等",
}

# ロト7 当選条件
_LOTO7_PRIZE: dict[tuple[int, int], str] = {
    (7, 0): "1等",
    (6, 1): "2等",  # ボーナスいずれか1個
    (6, 0): "3等",
    (5, 0): "4等",
    (4, 1): "5等",  # ボーナスいずれか1個
    (4, 0): "6等",
    (3, 1): "7等",  # ボーナスいずれか1個
}

# ミニロト 当選条件
_MINILOTO_PRIZE: dict[tuple[int, int], str] = {
    (5, 0): "1等",
    (4, 1): "2等",
    (4, 0): "3等",
    (3, 0): "4等",
}

_PRIZE_TABLE: dict[str, dict] = {
    "LOTO6": _LOTO6_PRIZE,
    "LOTO7": _LOTO7_PRIZE,
    "MINILOTO": _MINILOTO_PRIZE,
}


def judge_prize(
    predicted: list[int],
    winning: list[int],
    bonus: list[int],
    game_key: str,
) -> tuple[int, int, Optional[str]]:
    """
    予測番号と当選番号を照合し、等級を返す。

    Args:
        predicted: 予測した本数字リスト
        winning:   当選本数字リスト
        bonus:     当選ボーナス数字リスト
        game_key:  ゲームキー

    Returns:
        (本数字一致数, ボーナス一致数, 等級文字列 or None)
    """
    predicted_set = set(predicted)
    winning_set = set(winning)
    bonus_set = set(bonus)

    main_match = len(predicted_set & winning_set)
    bonus_match = len(predicted_set & bonus_set)

    prize_table = _PRIZE_TABLE[game_key]

    # 等級の判定：本数字の多い順に照合
    grade: Optional[str] = None
    for (req_main, req_bonus), g in sorted(prize_table.items(), key=lambda x: (-x[0][0], -x[0][1])):
        if main_match >= req_main and bonus_match >= req_bonus:
            grade = g
            break

    return main_match, bonus_match, grade


def _get_predictions_dir() -> str:
    """data/predictions/ ディレクトリのパスを返す"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "predictions")


def _get_results_dir() -> str:
    """data/results/ ディレクトリのパスを返す"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _get_data_dir() -> str:
    """data/raw/ ディレクトリのパスを返す"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "raw")


_CSV_FILENAMES: dict[str, str] = {
    "LOTO6": "loto6.csv",
    "LOTO7": "loto7.csv",
    "MINILOTO": "miniloto.csv",
}

# ロト7はボーナス2個
_BONUS_COLS: dict[str, tuple[int, int]] = {
    "LOTO6": (8, 9),  # [8]のみ
    "LOTO7": (9, 11),  # [9][10]
    "MINILOTO": (7, 8),  # [7]のみ
}
_MAIN_COLS: dict[str, tuple[int, int]] = {
    "LOTO6": (2, 8),
    "LOTO7": (2, 9),
    "MINILOTO": (2, 7),
}


def get_winning_numbers(
    game_key: str,
    draw_no: int,
    data_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    CSVから指定回の当選番号を取得する。

    Returns:
        {"draw_no": int, "date": str, "main": [int], "bonus": [int]}
        見つからない場合は None
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    csv_path = os.path.join(data_dir, _CSV_FILENAMES[game_key])
    if not os.path.exists(csv_path):
        return None

    main_start, main_end = _MAIN_COLS[game_key]
    bonus_start, bonus_end = _BONUS_COLS[game_key]

    with open(csv_path, encoding="shift_jis", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダースキップ
        for row in reader:
            if len(row) < bonus_end:
                continue
            try:
                if int(row[0]) == draw_no:
                    main = sorted(int(row[i]) for i in range(main_start, main_end))
                    bonus = sorted(int(row[i]) for i in range(bonus_start, bonus_end))
                    return {
                        "draw_no": draw_no,
                        "date": row[1],
                        "main": main,
                        "bonus": bonus,
                    }
            except (ValueError, IndexError):
                continue
    return None


def _find_latest_prediction(game_key: str, predictions_dir: str) -> Optional[str]:
    """指定ゲームの最新予測JSONを見つける"""
    game_key_lower = game_key.lower()
    candidates = []
    if not os.path.isdir(predictions_dir):
        return None

    for filename in os.listdir(predictions_dir):
        if filename.startswith(f"prediction_{game_key_lower}_") and filename.endswith(".json"):
            candidates.append(os.path.join(predictions_dir, filename))

    if not candidates:
        return None
    return max(candidates)  # 日付が末尾なので最大=最新


def verify_prediction_file(
    json_path: str,
    winning_draw_no: int,
    winning_main: list[int],
    winning_bonus: list[int],
    game_key: str,
) -> dict[str, Any]:
    """
    予測JSONと実際の当選番号を照合する。

    Returns:
        照合結果の辞書
    """
    with open(json_path, encoding="utf-8") as f:
        prediction = json.load(f)

    config = LOTTERY_CONFIG[game_key]
    verify_results: list[dict] = []

    for method_key, method_data in prediction.get("results", {}).items():
        method_preds = method_data.get("predictions", [])
        method_results = []

        for i, pred in enumerate(method_preds):
            numbers = pred.get("numbers", [])
            main_match, bonus_match, grade = judge_prize(numbers, winning_main, winning_bonus, game_key)
            method_results.append(
                {
                    "set": i + 1,
                    "predicted": numbers,
                    "main_match": main_match,
                    "bonus_match": bonus_match,
                    "grade": grade,
                    "hit": grade is not None,
                }
            )

        # 手法のサマリー
        hits = [r for r in method_results if r["hit"]]
        max_match = max((r["main_match"] for r in method_results), default=0)

        verify_results.append(
            {
                "method": method_key,
                "sets": len(method_results),
                "hit_count": len(hits),
                "max_main_match": max_match,
                "best_grade": hits[0]["grade"] if hits else None,
                "details": method_results,
            }
        )

    return {
        "prediction_file": os.path.basename(json_path),
        "game_key": game_key,
        "game_name": config["name"],
        "prediction_run_date": prediction.get("run_date"),
        "prediction_run_datetime": prediction.get("run_datetime"),
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "winning": {
            "draw_no": winning_draw_no,
            "date": None,  # 後から設定
            "main": winning_main,
            "bonus": winning_bonus,
        },
        "method_results": verify_results,
    }


def print_verify_report(result: dict[str, Any]) -> None:
    """照合結果をコンソールに出力する"""
    config = LOTTERY_CONFIG[result["game_key"]]
    winning = result["winning"]

    print(f"\n{'=' * 55}")
    print(f"🎯 {result['game_name']} 答え合わせ")
    print(f"   予測実行日: {result['prediction_run_date']}")
    print(f"{'=' * 55}")
    print(f"🏆 実際の当選番号（第{winning['draw_no']}回）")
    print(f"   本数字: {winning['main']}")
    if winning["bonus"]:
        print(f"   ボーナス: {winning['bonus']}")
    print()

    for mr in result["method_results"]:
        method_label = {
            "montecarlo": "モンテカルロ",
            "clustering": "クラスタリング",
            "genetic": "遺伝的アルゴリズム",
        }.get(mr["method"], mr["method"])

        print(f"📊 {method_label} ({mr['sets']}セット)")

        best = mr.get("best_grade")
        if best:
            print(f"   ⭐ 当選！最高等級: {best}")
        else:
            print(f"   最大一致数: {mr['max_main_match']}個（本数字）")

        for det in mr["details"]:
            mark = "🎉" if det["hit"] else "  "
            grade_str = f" [{det['grade']}]" if det["hit"] else ""
            print(
                f"   {mark} Set{det['set']:02d}: {det['predicted']}"
                f"  本数字{det['main_match']}個一致、ボーナス{det['bonus_match']}個一致{grade_str}"
            )
        print()


def save_verify_result(result: dict[str, Any], results_dir: str) -> str:
    """照合結果をJSONで保存する"""
    game_key = result["game_key"].lower()
    draw_no = result["winning"]["draw_no"]
    pred_date = (result.get("prediction_run_date") or "unknown").replace("-", "")
    filename = f"verify_{game_key}_draw{draw_no}_{pred_date}.json"
    path = os.path.join(results_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.verification.verify_results",
        description="予測結果と実際の当選番号を照合する",
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["loto6", "loto7", "miniloto"],
        default=None,
        help="対象ゲーム（省略時: 全ゲーム）",
    )
    parser.add_argument(
        "--draw",
        type=int,
        default=None,
        help="照合する当選回号（省略時: 最新）",
    )
    parser.add_argument(
        "--winning",
        nargs="+",
        type=int,
        default=None,
        help="当選本数字（手動指定: --winning 5 12 20 31 37 42）",
    )
    parser.add_argument(
        "--bonus",
        nargs="+",
        type=int,
        default=None,
        help="当選ボーナス数字（手動指定: --bonus 8）",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="予測JSONファイルを直接指定",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="予測JSONディレクトリ（省略時: data/predictions/）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predictions_dir = args.predictions_dir or _get_predictions_dir()
    results_dir = _get_results_dir()

    print("\n🔍 答え合わせ開始")

    # 対象ゲームの決定
    if args.game:
        games = [args.game.upper()]
    else:
        games = ["LOTO6", "LOTO7", "MINILOTO"]

    for game_key in games:
        config = LOTTERY_CONFIG[game_key]
        print(f"\n🎮 {config['name']}")

        # 予測ファイルの特定
        if args.file:
            json_path = args.file
        else:
            json_path = _find_latest_prediction(game_key, predictions_dir)
            if not json_path:
                print(f"   ⚠️  予測ファイルが見つかりません: {predictions_dir}")
                continue

        print(f"   予測ファイル: {os.path.basename(json_path)}")

        # 当選番号の取得
        if args.winning:
            # 手動指定
            winning_main = sorted(args.winning)
            winning_bonus = sorted(args.bonus) if args.bonus else []
            draw_no = args.draw or 0
            winning_date = "手動入力"
        else:
            # CSVから自動取得
            if args.draw:
                draw_no = args.draw
            else:
                # 最新回を特定 (予測ファイルの latest_draw_no + 1 を基準に探す)
                with open(json_path, encoding="utf-8") as f:
                    pred_data = json.load(f)
                base_draw = pred_data.get("data_summary", {}).get("latest_draw_no", 0)
                draw_no = base_draw + 1

            won = get_winning_numbers(game_key, draw_no)
            if not won:
                print(f"   ⚠️  第{draw_no}回の当選番号がCSVに見つかりません")
                print(f"      先に update_data を実行してデータを更新してください")
                print(f"      または --winning で手動入力してください")
                continue

            winning_main = won["main"]
            winning_bonus = won["bonus"]
            winning_date = won["date"]

        print(f"   当選番号（第{draw_no}回 {winning_date}）: 本数字={winning_main}, ボーナス={winning_bonus}")

        # 照合
        result = verify_prediction_file(
            json_path,
            winning_draw_no=draw_no,
            winning_main=winning_main,
            winning_bonus=winning_bonus,
            game_key=game_key,
        )
        result["winning"]["date"] = winning_date

        # 結果表示
        print_verify_report(result)

        # 保存
        saved = save_verify_result(result, results_dir)
        print(f"   💾 保存: {saved}")


if __name__ == "__main__":
    main()
