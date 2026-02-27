"""
ロト予測ツール - 遺伝的アルゴリズム 可視化モジュール

Plotly を使って GA の進化過程と予測結果を
インタラクティブな HTML レポートとして出力する。

グラフ:
    1. 適応度推移グラフ     — 世代ごとの最良/平均/最悪の折れ線
    2. 適応度レーダーチャート — 最良個体の各評価軸スコア
    3. 数字分布ヒートマップ — 最終世代全個体の各数字出現頻度
    4. 予測番号テーブル     — 推薦された番号セット
"""

import os
from collections import Counter
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.common import LOTTERY_CONFIG
from src.genetic.engine import GAResult


# 評価軸ラベル（レーダーチャート用）
_AXIS_LABELS = ["合計値", "奇偶バランス", "高低バランス", "連続数字", "出現頻度"]
_AXIS_KEYS = ["sum", "odd_even", "high_low", "consecutive", "frequency"]


def generate_ga_report_html(
    result: GAResult,
    config: dict,
    game_key: str,
    predictions: list[tuple[int, ...]],
    output_dir: str = "output",
) -> str:
    """
    GA の実行結果をインタラクティブ HTML レポートとして保存する。

    Args:
        result: GA 実行結果
        config: LOTTERY_CONFIG[game_key]
        game_key: ゲームキー
        predictions: 予測番号セットのリスト
        output_dir: 出力ディレクトリ

    Returns:
        生成した HTML ファイルのパス
    """
    game_name = config["name"]
    range_max = config["range_max"]

    # ── 1. 適応度推移グラフ ──
    fig_fitness = _create_fitness_chart(result, game_name)

    # ── 2. レーダーチャート ──
    fig_radar = _create_radar_chart(result, game_name)

    # ── 3. 数字分布ヒートマップ ──
    fig_heatmap = _create_number_heatmap(result, game_name, range_max)

    # ── 4. HTMLの組み立て ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gc = result.ga_config

    # 予測番号テーブル
    prediction_rows = ""
    for i, nums in enumerate(predictions, 1):
        nums_str = " - ".join(f"{n:02d}" for n in nums)
        prediction_rows += f"<tr><td>{i}</td><td>{nums_str}</td></tr>\n"

    # Plotly の JSON を事前に取得
    fitness_json = fig_fitness.to_json()
    radar_json = fig_radar.to_json()
    heatmap_json = fig_heatmap.to_json()

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{game_name} GA予測レポート</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', 'Hiragino Kaku Gothic ProN', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-size: 2em;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            font-size: 1.3em;
            margin-top: 0;
            color: #a8d8ea;
        }}
        .params {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .param-item {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .param-item .label {{
            font-size: 0.85em;
            color: #888;
        }}
        .param-item .value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #00d2ff;
        }}
        .chart-container {{
            width: 100%;
            margin: 10px 0;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 768px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px 16px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,210,255,0.1);
            color: #a8d8ea;
        }}
        tr:hover {{
            background: rgba(255,255,255,0.03);
        }}
        .footer {{
            text-align: center;
            color: #555;
            margin-top: 30px;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 {game_name} 遺伝的アルゴリズム 予測レポート</h1>
        <p class="subtitle">生成日時: {timestamp}</p>

        <!-- パラメータサマリー -->
        <div class="card">
            <h2>⚙️ パラメータ</h2>
            <div class="params">
                <div class="param-item">
                    <div class="label">個体群サイズ</div>
                    <div class="value">{gc.population_size}</div>
                </div>
                <div class="param-item">
                    <div class="label">世代数</div>
                    <div class="value">{result.generations_run}</div>
                </div>
                <div class="param-item">
                    <div class="label">交叉率</div>
                    <div class="value">{gc.crossover_rate}</div>
                </div>
                <div class="param-item">
                    <div class="label">突然変異率</div>
                    <div class="value">{gc.mutation_rate}</div>
                </div>
                <div class="param-item">
                    <div class="label">エリート保存</div>
                    <div class="value">{gc.elite_count}体</div>
                </div>
                <div class="param-item">
                    <div class="label">最終適応度</div>
                    <div class="value">{result.best_fitness:.4f}</div>
                </div>
            </div>
        </div>

        <!-- 適応度推移 -->
        <div class="card">
            <h2>📈 適応度推移</h2>
            <div class="chart-container" id="chart-fitness"></div>
        </div>

        <!-- レーダーチャート & ヒートマップ -->
        <div class="two-col">
            <div class="card">
                <h2>🎯 最良個体の適応度内訳</h2>
                <div class="chart-container" id="chart-radar"></div>
            </div>
            <div class="card">
                <h2>🔢 最終世代の数字分布</h2>
                <div class="chart-container" id="chart-heatmap"></div>
            </div>
        </div>

        <!-- 予測番号 -->
        <div class="card">
            <h2>🎰 予測番号</h2>
            <table>
                <thead>
                    <tr><th>No.</th><th>予測番号</th></tr>
                </thead>
                <tbody>
                    {prediction_rows}
                </tbody>
            </table>
        </div>

        <p class="footer">
            ロト予測ツール - 遺伝的アルゴリズム | {game_name} |
            個体群{gc.population_size} × {result.generations_run}世代
        </p>
    </div>

    <script>
        // 適応度推移
        var figFitness = {fitness_json};
        Plotly.newPlot('chart-fitness', figFitness.data, figFitness.layout, {{responsive: true}});

        // レーダーチャート
        var figRadar = {radar_json};
        Plotly.newPlot('chart-radar', figRadar.data, figRadar.layout, {{responsive: true}});

        // ヒートマップ
        var figHeatmap = {heatmap_json};
        Plotly.newPlot('chart-heatmap', figHeatmap.data, figHeatmap.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ga_{game_key.lower()}_{timestamp_file}.html"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath


def _create_fitness_chart(result: GAResult, game_name: str) -> go.Figure:
    """適応度推移の折れ線グラフを生成する。"""
    history = result.fitness_history
    gens = list(range(len(history)))
    best_vals = [h["best"] for h in history]
    avg_vals = [h["average"] for h in history]
    worst_vals = [h["worst"] for h in history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=gens,
            y=best_vals,
            mode="lines",
            name="最良",
            line=dict(color="#00d2ff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=avg_vals,
            mode="lines",
            name="平均",
            line=dict(color="#ffa500", width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=worst_vals,
            mode="lines",
            name="最悪",
            line=dict(color="#ff4444", width=1, dash="dash"),
            opacity=0.5,
        )
    )

    fig.update_layout(
        xaxis_title="世代",
        yaxis_title="適応度",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )

    return fig


def _create_radar_chart(result: GAResult, game_name: str) -> go.Figure:
    """最良個体の適応度レーダーチャートを生成する。"""
    detail = result.best_fitness_detail
    values = [detail.get(k, 0.0) for k in _AXIS_KEYS]
    # レーダーチャートは最初の値を末尾に追加して閉じる
    values_closed = values + [values[0]]
    labels_closed = _AXIS_LABELS + [_AXIS_LABELS[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(0,210,255,0.15)",
            line=dict(color="#00d2ff", width=2),
            name="適応度",
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=60, r=60, t=30, b=30),
        height=350,
    )

    return fig


def _create_number_heatmap(
    result: GAResult,
    game_name: str,
    range_max: int,
) -> go.Figure:
    """最終世代の全個体における各数字の出現頻度ヒートマップを生成する。"""
    # 各数字の出現回数をカウント
    counter: Counter = Counter()
    for individual in result.final_population:
        counter.update(individual)

    numbers = list(range(1, range_max + 1))
    counts = [counter.get(n, 0) for n in numbers]

    # 正規化（0〜1）
    max_count = max(counts) if counts else 1
    normalized = [c / max_count for c in counts]

    # 棒グラフで表示（ヒートマップ的な色彩）
    colors = [f"rgb({int(255 * (1 - v))}, {int(100 + 155 * v)}, {int(255 * v)})" for v in normalized]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=numbers,
            y=counts,
            marker_color=colors,
            text=[f"{n}" for n in numbers],
            hovertemplate="数字 %{x}: %{y}回<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="数字",
        yaxis_title="出現回数",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        bargap=0.1,
    )

    return fig
