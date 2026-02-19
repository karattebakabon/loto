"""
ロト予測ツール - クラスタリング分析 CLIエントリーポイント

使用方法:
    python -m src.clustering [オプション]

実行例:
    # ロト6（デフォルト: K-Means + DBSCAN 両方）
    python -m src.clustering

    # ロト7、K-Meansのみ
    python -m src.clustering --game loto7 --method kmeans

    # ミニロト、DBSCAN、予測10セット
    python -m src.clustering --game miniloto --method dbscan --predictions 10

    # 直近200回、重心狙い
    python -m src.clustering --recent 200 --strategy centroid

    # インタラクティブHTMLグラフを生成
    python -m src.clustering --visualize
"""

import argparse
import os
import sys
import time

from src.common import LOTTERY_CONFIG
from src.common.data_loader import load_lottery_data
from src.clustering.feature_extractor import extract_features
from src.clustering.engine import run_kmeans, run_dbscan, find_optimal_k
from src.clustering.predictor import generate_predictions
from src.clustering.analyzer import print_cluster_report, print_prediction_report
from src.clustering.visualizer import generate_cluster_report_html


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        prog="python -m src.clustering",
        description="ロト予測 クラスタリング分析（K-Means / DBSCAN）",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="loto6",
        choices=["loto6", "loto7", "miniloto"],
        help="対象ゲーム（デフォルト: loto6）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["kmeans", "dbscan", "both"],
        help="クラスタリング手法（デフォルト: both）",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="K-Meansのクラスタ数（省略時: エルボー法で自動推定）",
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
        "--strategy",
        type=str,
        default="centroid",
        choices=["centroid", "recent", "pocket"],
        help="予測戦略（デフォルト: centroid）",
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


def main() -> None:
    """メイン処理"""
    args = _parse_args()
    game_key = args.game.upper()
    config = LOTTERY_CONFIG[game_key]

    print(f"\n🔬 {config['name']} クラスタリング分析")
    print(f"   範囲: 1〜{config['range_max']}  選択数: {config['pick_size']}個")

    # 1. CSVデータの読み込み
    print(f"\n📂 過去データを読み込み中...")
    try:
        data = load_lottery_data(game_key)
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        print("   data/raw/ ディレクトリにCSVファイルを配置してください。", file=sys.stderr)
        sys.exit(1)

    # 直近N回に絞り込み
    if args.recent:
        data = data[-args.recent :]
        print(f"   直近{args.recent}回分を使用します")
    print(f"   {len(data):,}回分のデータを読み込みました")

    # 2. 特徴量抽出
    print(f"\n📐 特徴量を抽出中...")
    features = extract_features(data, game_key)
    print(f"   {features.shape[0]}件 × {features.shape[1]}次元の特徴量行列を生成")

    # 3. クラスタリング実行
    results_list = []

    if args.method in ("kmeans", "both"):
        print(f"\n🔵 K-Means クラスタリング実行中...")
        start_time = time.time()

        # クラスタ数の決定
        if args.clusters:
            n_clusters = args.clusters
            print(f"   指定されたクラスタ数: {n_clusters}")
        else:
            print(f"   エルボー法でクラスタ数を推定中...")
            elbow = find_optimal_k(features)
            n_clusters = elbow["optimal_k"]
            print(f"   最適クラスタ数: {n_clusters}")

        km_result = run_kmeans(features, n_clusters=n_clusters)
        elapsed = time.time() - start_time
        print(f"   完了！ ({elapsed:.2f}秒)")

        print_cluster_report(km_result, data, config)
        results_list.append(km_result)

    if args.method in ("dbscan", "both"):
        print(f"\n🟠 DBSCAN クラスタリング実行中...")
        start_time = time.time()

        db_result = run_dbscan(features)
        elapsed = time.time() - start_time
        print(f"   完了！ ({elapsed:.2f}秒)")
        print(f"   検出クラスタ数: {db_result.n_clusters}, ノイズ: {db_result.noise_count}件")

        print_cluster_report(db_result, data, config)
        results_list.append(db_result)

    # 4. 予測生成（最初の結果を使用）
    if results_list:
        primary_result = results_list[0]
        print(f"\n🎯 予測番号を生成中... (戦略: {args.strategy})")

        predictions = generate_predictions(
            primary_result,
            data,
            features,
            game_key,
            n_predictions=args.predictions,
            strategy=args.strategy,
        )

        print_prediction_report(predictions, config, args.strategy)

    # 5. 可視化（オプション）
    if args.visualize and results_list:
        print(f"\n📊 インタラクティブHTMLレポートを生成中...")
        for vis_result in results_list:
            html_path = generate_cluster_report_html(
                result=vis_result,
                features=features,
                data=data,
                config=config,
                game_key=game_key,
                predictions=predictions if results_list[0] is vis_result else None,
                strategy=args.strategy,
                output_dir=args.output_dir,
            )
            method_label = "K-Means" if vis_result.method == "kmeans" else "DBSCAN"
            print(f"   ✅ {method_label}: {html_path}")
            abs_path = os.path.abspath(html_path).replace(os.sep, "/")
            print(f"   ブラウザで開いてください: file:///{abs_path}")

    print(f"\n✅ 分析完了！")


if __name__ == "__main__":
    main()
