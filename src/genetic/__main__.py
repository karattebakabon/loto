"""
ロト予測ツール - 遺伝的アルゴリズム CLIエントリーポイント

使用方法:
    python -m src.genetic [オプション]

実行例:
    # ロト6（デフォルト）
    python -m src.genetic

    # ロト7、世代数1000、個体群300
    python -m src.genetic --game loto7 --generations 1000 --population 300

    # ミニロト、予測10セット
    python -m src.genetic --game miniloto --predictions 10

    # 直近200回のデータを使用
    python -m src.genetic --recent 200

    # インタラクティブHTMLグラフを生成
    python -m src.genetic --visualize

    # 全部入り
    python -m src.genetic --game loto7 --generations 1000 --population 300 --visualize
"""

import argparse
import os
import sys
import time

from src.common import LOTTERY_CONFIG
from src.common.data_loader import load_lottery_data
from src.genetic.engine import GeneticEngine, GAConfig
from src.genetic.fitness import FitnessConfig
from src.genetic.analyzer import print_evolution_report, print_prediction_report
from src.genetic.visualizer import generate_ga_report_html


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        prog="python -m src.genetic",
        description="ロト予測 遺伝的アルゴリズム（GA）",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="loto6",
        choices=["loto6", "loto7", "miniloto"],
        help="対象ゲーム（デフォルト: loto6）",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=200,
        help="個体群のサイズ（デフォルト: 200）",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="世代数（デフォルト: 500）",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="交叉率（デフォルト: 0.8）",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="突然変異率（デフォルト: 0.1）",
    )
    parser.add_argument(
        "--elite",
        type=int,
        default=10,
        help="エリート保存数（デフォルト: 10）",
    )
    parser.add_argument(
        "--tournament",
        type=int,
        default=5,
        help="トーナメントサイズ（デフォルト: 5）",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=None,
        help="直近N回のデータのみ使用（省略時: 全データ）",
    )
    parser.add_argument(
        "--predictions",
        type=int,
        default=5,
        help="生成する予測セット数（デフォルト: 5）",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="plotly でインタラクティブHTMLグラフを生成",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="出力ディレクトリ（デフォルト: output/）",
    )
    return parser.parse_args()


def _progress_printer(current: int, total: int) -> None:
    """進化の進行状況をコンソールに表示"""
    pct = current / total * 100
    print(f"\r  進化中... 第{current:>5}世代 / {total}世代 ({pct:.1f}%)", end="", flush=True)


def main() -> None:
    """メイン処理"""
    args = _parse_args()
    game_key = args.game.upper()
    config = LOTTERY_CONFIG[game_key]

    print(f"\n🧬 {config['name']} 遺伝的アルゴリズム")
    print(f"   範囲: 1〜{config['range_max']}  選択数: {config['pick_size']}個")

    # 1. CSVデータの読み込み
    print(f"\n📂 過去データを読み込み中...")
    try:
        data = load_lottery_data(game_key)
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        print("   data/raw/ ディレクトリにCSVファイルを配置してください。", file=sys.stderr)
        sys.exit(1)

    total_draws = len(data)
    print(f"   {total_draws:,}回分のデータを読み込みました")
    if args.recent:
        print(f"   直近{args.recent}回分を使用します")

    # 2. GA設定の構築
    ga_config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_count=args.elite,
        tournament_size=args.tournament,
    )

    print(f"\n⚙️  GA パラメータ")
    print(f"   個体群: {ga_config.population_size}  世代: {ga_config.generations}")
    print(f"   交叉: {ga_config.crossover_rate}  突然変異: {ga_config.mutation_rate}")
    print(f"   エリート: {ga_config.elite_count}  トーナメント: {ga_config.tournament_size}")

    # 3. GA実行
    print(f"\n🔄 進化を開始...")
    start_time = time.time()

    engine = GeneticEngine(
        game_key,
        data,
        ga_config=ga_config,
        recent_n=args.recent,
    )
    result = engine.run(
        n_predictions=args.predictions,
        progress_callback=_progress_printer,
        progress_interval=max(ga_config.generations // 20, 1),
    )
    print()  # 改行（進捗表示の後）

    elapsed = time.time() - start_time
    print(f"   完了！ 実行時間: {elapsed:.2f}秒")

    # 4. 結果の表示
    print_evolution_report(result, config)
    print_prediction_report(result.best_individuals, config)

    # 5. 可視化（オプション）
    if args.visualize:
        print(f"\n📊 インタラクティブHTMLレポートを生成中...")
        html_path = generate_ga_report_html(
            result=result,
            config=config,
            game_key=game_key,
            predictions=result.best_individuals,
            output_dir=args.output_dir,
        )
        print(f"   ✅ HTML: {html_path}")
        abs_path = os.path.abspath(html_path).replace(os.sep, "/")
        print(f"   ブラウザで開いてください: file:///{abs_path}")

    print(f"\n✅ 分析完了！")


if __name__ == "__main__":
    main()
