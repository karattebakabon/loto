"""
ロト予測ツール - インタラクティブ可視化モジュール

plotly を使用してシミュレーション結果を
インタラクティブなHTMLグラフとして出力する。
"""

import os
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.montecarlo.analyzer import (
    analyze_top_combinations,
    analyze_number_frequency,
)


def _ensure_output_dir(output_dir: str) -> None:
    """出力ディレクトリが存在しない場合は作成する"""
    os.makedirs(output_dir, exist_ok=True)


def generate_report_html(
    results: list[tuple[int, ...]],
    config: dict,
    game_key: str,
    trials: int,
    output_dir: str = "output",
    top_n: int = 20,
    filepath: Optional[str] = None,
) -> str:
    """
    シミュレーション結果のインタラクティブHTMLレポートを生成する。

    含まれるグラフ:
    1. 数字別出現頻度（棒グラフ）
    2. 数字別出現頻度（ヒートマップ）
    3. 頻出組み合わせトップN（横棒グラフ）

    Args:
        results: シミュレーション結果
        config: LOTTERY_CONFIG[game_key]
        game_key: ゲームキー
        trials: 試行回数
        output_dir: 出力ディレクトリ
        top_n: 頻出組み合わせの表示件数
        filepath: 出力ファイルパス（省略時は自動生成）

    Returns:
        保存したHTMLファイルのパス
    """
    _ensure_output_dir(output_dir)

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"mc_{game_key.lower()}_{timestamp}.html")

    game_name = config["name"]
    range_max = config["range_max"]
    total = len(results)

    # データ準備
    freq = analyze_number_frequency(results, range_max)
    top_combos = analyze_top_combinations(results, top_n)

    # 数字順にソート（棒グラフ用）
    numbers_sorted = list(range(1, range_max + 1))
    counts_sorted = [freq[n] for n in numbers_sorted]
    pcts_sorted = [(freq[n] / total) * 100 for n in numbers_sorted]

    # 平均出現回数（期待値）
    pick_size = config["pick_size"]
    expected_per_number = (trials * pick_size) / range_max

    # ── カラーパレット ──
    bg_color = "#0d1117"
    card_color = "#161b22"
    text_color = "#e6edf3"
    accent_color = "#58a6ff"
    grid_color = "#30363d"
    hot_color = "#ff6b6b"
    cold_color = "#4ecdc4"

    # 棒の色分け（期待値より上=ホット、下=コールド）
    bar_colors = [hot_color if c > expected_per_number else cold_color for c in counts_sorted]

    # ═══════════════════════════════════════
    # グラフ1: 数字別出現頻度（棒グラフ）
    # ═══════════════════════════════════════
    fig_bar = go.Figure()

    # 期待値ライン
    fig_bar.add_hline(
        y=expected_per_number,
        line_dash="dash",
        line_color="#8b949e",
        line_width=1,
        annotation_text=f"期待値 ({expected_per_number:,.0f})",
        annotation_position="top right",
        annotation_font_color="#8b949e",
    )

    fig_bar.add_trace(
        go.Bar(
            x=numbers_sorted,
            y=counts_sorted,
            marker_color=bar_colors,
            marker_line_width=0,
            hovertemplate=("<b>数字 %{x}</b><br>出現回数: %{y:,}<br>割合: %{customdata:.2f}%<extra></extra>"),
            customdata=pcts_sorted,
        )
    )

    fig_bar.update_layout(
        title=dict(
            text=f"🎰 {game_name} 数字別出現頻度",
            font=dict(size=20, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            title="数字",
            tickmode="linear",
            dtick=1,
            gridcolor=grid_color,
            color=text_color,
        ),
        yaxis=dict(
            title="出現回数",
            gridcolor=grid_color,
            color=text_color,
        ),
        plot_bgcolor=card_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        hoverlabel=dict(
            bgcolor=card_color,
            font_size=13,
            font_color=text_color,
        ),
        margin=dict(l=60, r=30, t=60, b=40),
        height=450,
    )

    # ═══════════════════════════════════════
    # グラフ2: ヒートマップ
    # ═══════════════════════════════════════
    # 数字を10個ずつの行に配置
    cols_per_row = 10
    heatmap_rows = []
    heatmap_labels = []
    row_labels = []

    for start in range(1, range_max + 1, cols_per_row):
        end = min(start + cols_per_row, range_max + 1)
        row_data = []
        row_text = []
        for n in range(start, start + cols_per_row):
            if n <= range_max:
                row_data.append(freq[n])
                row_text.append(str(n))
            else:
                row_data.append(None)
                row_text.append("")
        heatmap_rows.append(row_data)
        heatmap_labels.append(row_text)
        row_labels.append(f"{start}-{min(start + cols_per_row - 1, range_max)}")

    fig_heat = go.Figure()

    fig_heat.add_trace(
        go.Heatmap(
            z=heatmap_rows,
            text=heatmap_labels,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            colorscale=[
                [0, "#1a1a2e"],
                [0.25, "#16213e"],
                [0.5, "#0f3460"],
                [0.75, "#e94560"],
                [1, "#ff6b6b"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="出現回数", font=dict(color=text_color)),
                tickfont=dict(color=text_color),
            ),
            hovertemplate=("数字: %{text}<br>出現回数: %{z:,}<extra></extra>"),
            ygap=3,
            xgap=3,
        )
    )

    fig_heat.update_layout(
        title=dict(
            text=f"🔥 {game_name} 出現頻度ヒートマップ",
            font=dict(size=20, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        yaxis=dict(
            ticktext=row_labels,
            tickvals=list(range(len(row_labels))),
            color=text_color,
            showgrid=False,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        hoverlabel=dict(
            bgcolor=card_color,
            font_size=13,
            font_color=text_color,
        ),
        margin=dict(l=60, r=80, t=60, b=30),
        height=max(200, len(row_labels) * 60 + 100),
    )

    # ═══════════════════════════════════════
    # グラフ3: 頻出組み合わせ（横棒グラフ）
    # ═══════════════════════════════════════
    combo_labels = [" - ".join(f"{n:02d}" for n in numbers) for numbers, _ in reversed(top_combos)]
    combo_counts = [count for _, count in reversed(top_combos)]
    combo_pcts = [(c / total) * 100 for c in combo_counts]

    fig_combo = go.Figure()

    fig_combo.add_trace(
        go.Bar(
            y=combo_labels,
            x=combo_counts,
            orientation="h",
            marker_color=accent_color,
            marker_line_width=0,
            hovertemplate=("<b>%{y}</b><br>出現回数: %{x:,}<br>割合: %{customdata:.4f}%<extra></extra>"),
            customdata=combo_pcts,
        )
    )

    fig_combo.update_layout(
        title=dict(
            text=f"🏆 {game_name} 頻出組み合わせ トップ{top_n}",
            font=dict(size=20, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            title="出現回数",
            gridcolor=grid_color,
            color=text_color,
        ),
        yaxis=dict(
            color=text_color,
            tickfont=dict(family="Consolas, monospace", size=11),
        ),
        plot_bgcolor=card_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        hoverlabel=dict(
            bgcolor=card_color,
            font_size=13,
            font_color=text_color,
        ),
        margin=dict(l=180, r=30, t=60, b=40),
        height=max(400, top_n * 28 + 100),
    )

    # ═══════════════════════════════════════
    # HTML結合
    # ═══════════════════════════════════════
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False)
    heat_html = fig_heat.to_html(full_html=False, include_plotlyjs=False)
    combo_html = fig_combo.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{game_name} モンテカルロ・シミュレーション結果</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: {bg_color};
            color: {text_color};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid {grid_color};
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        .header .meta {{
            color: #8b949e;
            font-size: 0.9em;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: {card_color};
            border: 1px solid {grid_color};
            border-radius: 8px;
            padding: 15px 25px;
            text-align: center;
        }}
        .stat-card .label {{
            color: #8b949e;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        .stat-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: {accent_color};
        }}
        .chart-section {{
            background: {card_color};
            border: 1px solid {grid_color};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
        }}
        .legend {{
            text-align: center;
            margin: 10px 0;
            font-size: 0.85em;
            color: #8b949e;
        }}
        .legend span {{
            margin: 0 10px;
        }}
        .hot {{ color: {hot_color}; }}
        .cold {{ color: {cold_color}; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #484f58;
            font-size: 0.8em;
            border-top: 1px solid {grid_color};
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎰 {game_name} モンテカルロ・シミュレーション</h1>
        <p class="meta">実行日時: {timestamp_str}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="label">ゲーム</div>
            <div class="value">{game_name}</div>
        </div>
        <div class="stat-card">
            <div class="label">数字範囲</div>
            <div class="value">1 〜 {range_max}</div>
        </div>
        <div class="stat-card">
            <div class="label">選択数</div>
            <div class="value">{pick_size}個</div>
        </div>
        <div class="stat-card">
            <div class="label">試行回数</div>
            <div class="value">{trials:,}</div>
        </div>
    </div>

    <div class="chart-section">
        {bar_html}
        <div class="legend">
            <span class="hot">■ ホット（期待値以上）</span>
            <span class="cold">■ コールド（期待値未満）</span>
        </div>
    </div>

    <div class="chart-section">
        {heat_html}
    </div>

    <div class="chart-section">
        {combo_html}
    </div>

    <footer>
        Loto Predictor - モンテカルロ・シミュレーション | Generated by loto-predictor
    </footer>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath
