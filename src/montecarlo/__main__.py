"""
ロト予測ツール - モンテカルロ・シミュレーション CLIエントリーポイント

使用方法:
    python -m src.montecarlo [オプション]

実行例:
    # ロト6（デフォルト）
    python -m src.montecarlo

    # ロト7、試行50万回、直近100回分のデータ
    python -m src.montecarlo --game loto7 --trials 500000 --recent 100

    # ミニロト、トップ5表示
    python -m src.montecarlo --game miniloto --top 5
"""

import argparse
import sys
import time

from src.common import LOTTERY_CONFIG
from src.common.data_loader import load_lottery_data
from src.common.weights import calculate_frequency_weights
from src.montecarlo.simulator import MonteCarloSimulator
from src.montecarlo.analyzer import print_report


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        prog="python -m src.montecarlo",
        description="ロト予測 モンテカルロ・シミュレーション",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="loto6",
        choices=["loto6", "loto7", "miniloto"],
        help="対象ゲーム（デフォルト: loto6）",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100_000,
        help="シミュレーション試行回数（デフォルト: 100,000）",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=None,
        help="直近N回のデータのみ使用（省略時: 全データ）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="表示する上位組み合わせ数（デフォルト: 10）",
    )
    parser.add_argument(
        "--bonus-weight",
        type=float,
        default=0.3,
        help="ボーナス数字の重み係数（デフォルト: 0.3）",
    )
    return parser.parse_args()


def _progress_printer(current: int, total: int) -> None:
    """シミュレーション進行状況をコンソールに表示"""
    pct = current / total * 100
    print(f"\r  進行中... {current:>10,} / {total:,} ({pct:.1f}%)", end="", flush=True)


def main() -> None:
    """メイン処理"""
    args = _parse_args()
    game_key = args.game.upper()
    config = LOTTERY_CONFIG[game_key]

    print(f"\n🎲 {config['name']} モンテカルロ・シミュレーション")
    print(f"   範囲: 1〜{config['range_max']}  選択数: {config['pick_size']}個")
    print(f"   試行回数: {args.trials:,}")

    # 1. CSVデータの読み込み
    print(f"\n📂 過去データを読み込み中...")
    try:
        data = load_lottery_data(game_key)
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        print("   data/raw/ ディレクトリにCSVファイルを配置してください。", file=sys.stderr)
        sys.exit(1)

    total_draws = len(data)
    target_draws = args.recent if args.recent else total_draws
    print(f"   {total_draws:,}回分のデータを読み込みました")
    if args.recent:
        print(f"   直近{args.recent}回分を使用します")

    # 2. 重みの計算
    print(f"\n⚖️  重みを計算中...")
    weights = calculate_frequency_weights(
        data,
        game_key,
        recent_n=args.recent,
        bonus_weight=args.bonus_weight,
    )
    print(f"   ボーナス数字の重み係数: {args.bonus_weight}")

    # 3. シミュレーションの実行
    print(f"\n🎰 シミュレーション実行中...")
    start_time = time.time()

    simulator = MonteCarloSimulator(game_key, weights, trials=args.trials)
    results = simulator.run(
        progress_callback=_progress_printer,
        progress_interval=max(args.trials // 20, 1),  # 5%刻みで進捗表示
    )
    print()  # 改行（進捗表示の後）

    elapsed = time.time() - start_time
    print(f"   完了！ 実行時間: {elapsed:.2f}秒")

    # 4. 結果の表示
    print_report(results, config, top_n=args.top)


if __name__ == "__main__":
    main()
