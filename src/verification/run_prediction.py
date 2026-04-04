"""
ロト予測ツール - 一括予測実行スクリプト

3ゲーム（ロト6/7/ミニロト）× 3手法（モンテカルロ・クラスタリング・GA）の
予測を一括実行し、JSON形式で保存する。

使用方法:
    python -m src.verification.run_prediction
    python -m src.verification.run_prediction --game loto6
    python -m src.verification.run_prediction --game loto6 loto7
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

from src.common import LOTTERY_CONFIG
from src.common.data_loader import load_lottery_data
from src.common.weights import calculate_frequency_weights
from src.montecarlo.simulator import MonteCarloSimulator
from src.montecarlo.analyzer import analyze_top_combinations
from src.clustering.feature_extractor import extract_features
from src.clustering.engine import run_kmeans, find_optimal_k
from src.clustering.predictor import generate_predictions as clustering_predict
from src.genetic.engine import GeneticEngine, GAConfig

# ==========================================
# デフォルトパラメータ
# ==========================================

# モンテカルロ: 試行回数
MC_TRIALS = 100_000
# モンテカルロ: 上位N件を予測として採用
MC_TOP_N = 10

# クラスタリング: 予測セット数
CLUSTER_PREDICTIONS = 5
# クラスタリング: 予測戦略
CLUSTER_STRATEGY = "centroid"
# クラスタリング: スライディングウィンドウ（直近N回のみ使用、案2）
CLUSTER_WINDOW = 200

# GA: 世代数
GA_GENERATIONS = 500
# GA: 個体群サイズ
GA_POPULATION = 200
# GA: 予測セット数
GA_PREDICTIONS = 5


def _get_predictions_dir() -> str:
    """data/predictions/ ディレクトリのパスを返す"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pred_dir = os.path.join(base_dir, "data", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    return pred_dir


def run_montecarlo(
    game_key: str,
    data: list[dict],
) -> dict[str, Any]:
    """モンテカルロ・シミュレーションを実行し、予測結果を返す"""
    print(f"   🎲 モンテカルロ ({game_key})... ", end="", flush=True)
    t0 = time.time()

    weights = calculate_frequency_weights(data, game_key)
    simulator = MonteCarloSimulator(game_key, weights, trials=MC_TRIALS)
    raw_results = simulator.run()

    # 頻出上位MC_TOP_N件を予測として採用
    top_combos = analyze_top_combinations(raw_results, top_n=MC_TOP_N)
    predictions = [
        {
            "numbers": list(combo),
            "score": float(count),
        }
        for combo, count in top_combos
    ]

    elapsed = time.time() - t0
    print(f"完了 ({elapsed:.1f}秒, {len(predictions)}セット)")

    return {
        "method": "montecarlo",
        "trials": MC_TRIALS,
        "predictions": predictions,
        "elapsed_sec": round(elapsed, 2),
    }


def run_clustering(
    game_key: str,
    data: list[dict],
) -> dict[str, Any]:
    """クラスタリング分析を実行し、予測結果を返す"""
    print(f"   🔬 クラスタリング ({game_key})... ", end="", flush=True)
    t0 = time.time()

    # 案2: スライディングウィンドウ（直近CLUSTER_WINDOW回のみ使用）
    window = min(CLUSTER_WINDOW, len(data))
    data_window = data[-window:]

    features = extract_features(data_window, game_key)

    # 案5: シルエットスコアで最適K選定
    elbow = find_optimal_k(features)
    n_clusters = elbow["optimal_k"]
    best_silhouette = elbow.get("best_silhouette")
    km_result = run_kmeans(features, n_clusters=n_clusters)

    preds = clustering_predict(
        km_result,
        data_window,
        features,
        game_key,
        n_predictions=CLUSTER_PREDICTIONS,
        strategy=CLUSTER_STRATEGY,
    )

    predictions = [{"numbers": sorted(p)} for p in preds]

    elapsed = time.time() - t0
    sil_str = f", sil={best_silhouette:.3f}" if best_silhouette is not None else ""
    print(f"完了 ({elapsed:.1f}秒, {len(predictions)}セット, K={n_clusters}{sil_str}, 直近{window}回)")

    return {
        "method": "clustering",
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
        "strategy": CLUSTER_STRATEGY,
        "window": window,
        "best_silhouette": round(best_silhouette, 4) if best_silhouette is not None else None,
        "predictions": predictions,
        "elapsed_sec": round(elapsed, 2),
    }


def run_genetic(
    game_key: str,
    data: list[dict],
) -> dict[str, Any]:
    """遺伝的アルゴリズムを実行し、予測結果を返す"""
    print(f"   🧬 GA ({game_key})... ", end="", flush=True)
    t0 = time.time()

    ga_config = GAConfig(
        population_size=GA_POPULATION,
        generations=GA_GENERATIONS,
    )
    engine = GeneticEngine(game_key, data, ga_config=ga_config)
    result = engine.run(n_predictions=GA_PREDICTIONS)

    predictions = [{"numbers": sorted(ind)} for ind in result.best_individuals]

    elapsed = time.time() - t0
    print(f"完了 ({elapsed:.1f}秒, {len(predictions)}セット)")

    return {
        "method": "genetic",
        "generations": GA_GENERATIONS,
        "population": GA_POPULATION,
        "predictions": predictions,
        "elapsed_sec": round(elapsed, 2),
    }


def run_all_predictions(
    game_key: str,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """
    指定ゲームで全手法の予測を実行する。

    Returns:
        {
            "game_key": str,
            "run_date": str,
            "data_summary": {...},
            "results": {
                "montecarlo": {...},
                "clustering": {...},
                "genetic": {...},
            }
        }
    """
    config = LOTTERY_CONFIG[game_key]
    print(f"\n🎯 {config['name']} 予測開始")

    # データ読み込み
    try:
        data = load_lottery_data(game_key)
    except FileNotFoundError as e:
        print(f"❌ データファイルが見つかりません: {e}", file=sys.stderr)
        sys.exit(1)

    # 最新回情報
    latest = data[-1]
    print(f"   データ: {len(data)}件  最終回: 第{latest['draw_no']}回 ({latest['date']})")
    print(f"   次回以降の抽選を予測します")

    results: dict[str, Any] = {}
    if methods is None:
        methods = ["montecarlo", "clustering", "genetic"]

    if "montecarlo" in methods:
        results["montecarlo"] = run_montecarlo(game_key, data)
    if "clustering" in methods:
        results["clustering"] = run_clustering(game_key, data)
    if "genetic" in methods:
        results["genetic"] = run_genetic(game_key, data)

    return {
        "game_key": game_key,
        "game_name": config["name"],
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
        "data_summary": {
            "total_draws": len(data),
            "latest_draw_no": latest["draw_no"],
            "latest_date": latest["date"],
        },
        "results": results,
    }


def save_prediction(prediction: dict[str, Any], output_dir: str) -> str:
    """
    予測結果をJSON形式で保存する。

    ファイル名: prediction_{GAME}_{YYYYMMDD}.json
    Returns: 保存先パス
    """
    os.makedirs(output_dir, exist_ok=True)
    game_key = prediction["game_key"].lower()
    run_date = prediction["run_date"].replace("-", "")
    filename = f"prediction_{game_key}_{run_date}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)

    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.verification.run_prediction",
        description="3ゲーム×3手法の一括予測実行",
    )
    parser.add_argument(
        "--game",
        nargs="+",
        choices=["loto6", "loto7", "miniloto"],
        default=None,
        help="予測対象ゲーム（省略時: 全ゲーム）",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["montecarlo", "clustering", "genetic"],
        default=None,
        help="使用する手法（省略時: 全手法）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（省略時: data/predictions/）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or _get_predictions_dir()

    games = [g.upper() for g in args.game] if args.game else ["LOTO6", "LOTO7", "MINILOTO"]
    methods = args.method  # None = 全手法

    print("\n🔮 ロト予測 一括実行")
    print(f"   対象ゲーム: {', '.join(games)}")
    print(f"   手法: {', '.join(methods) if methods else '全て'}")
    print(f"   出力先: {output_dir}")

    total_start = time.time()
    saved_paths = []

    for game_key in games:
        prediction = run_all_predictions(game_key, methods)
        path = save_prediction(prediction, output_dir)
        saved_paths.append(path)
        print(f"   💾 保存: {path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 50}")
    print(f"✅ 全予測完了（合計 {total_elapsed:.1f}秒）")
    print(f"   保存ファイル数: {len(saved_paths)}")
    for p in saved_paths:
        print(f"   📄 {p}")


if __name__ == "__main__":
    main()
