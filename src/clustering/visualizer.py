"""
ロト予測ツール - クラスタリング分析 インタラクティブ可視化モジュール

plotly を使用してクラスタリング結果を
インタラクティブなHTMLグラフとして出力する。

含まれるグラフ:
    1. クラスタ散布図（PCA 2Dプロジェクション）
    2. クラスタサイズ分布（ドーナツチャート）
    3. 特徴量箱ひげ図（クラスタ別比較）
    4. 予測番号ヒートマップ
"""

import os
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.clustering.engine import ClusteringResult
from src.clustering.feature_extractor import FEATURE_NAMES


# ── カラーパレット（GitHub ダークテーマ準拠）──
_BG_COLOR = "#0d1117"
_CARD_COLOR = "#161b22"
_TEXT_COLOR = "#e6edf3"
_GRID_COLOR = "#30363d"
_MUTED_COLOR = "#8b949e"

# クラスタ用カラー（最大10クラスタ + ノイズ）
_CLUSTER_COLORS = [
    "#58a6ff",  # ブルー
    "#f78166",  # オレンジ
    "#3fb950",  # グリーン
    "#d2a8ff",  # パープル
    "#ff7b72",  # レッド
    "#79c0ff",  # ライトブルー
    "#ffa657",  # アンバー
    "#7ee787",  # ライトグリーン
    "#d0d7de",  # グレー
    "#f2cc60",  # イエロー
]
_NOISE_COLOR = "#484f58"  # ノイズ用（暗いグレー）


def _ensure_output_dir(output_dir: str) -> None:
    """出力ディレクトリが存在しない場合は作成する"""
    os.makedirs(output_dir, exist_ok=True)


def _get_cluster_color(label: int) -> str:
    """クラスタラベルに対応する色を返す"""
    if label == -1:
        return _NOISE_COLOR
    return _CLUSTER_COLORS[label % len(_CLUSTER_COLORS)]


def _build_scatter_chart(
    features: np.ndarray,
    result: ClusteringResult,
    data: list[dict],
    game_name: str,
) -> go.Figure:
    """
    グラフ1: PCA 2Dプロジェクションによるクラスタ散布図
    """
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(features)

    # 寄与率
    exp_var = pca.explained_variance_ratio_

    fig = go.Figure()

    # クラスタごとにトレースを追加
    unique_labels = sorted(set(result.labels.tolist()))
    for label in unique_labels:
        mask = result.labels == label
        indices = np.where(mask)[0]

        # ホバーテキストの構築
        hover_texts = []
        for idx in indices:
            draw = data[idx]
            nums = " - ".join(f"{n}" for n in draw["main_numbers"])
            hover_texts.append(
                f"第{draw['draw_no']}回<br>"
                f"番号: {nums}<br>"
                f"日付: {draw['date']}"
            )

        cluster_name = f"クラスタ {label}" if label >= 0 else "ノイズ（外れ値）"
        color = _get_cluster_color(label)

        fig.add_trace(
            go.Scatter(
                x=coords_2d[mask, 0],
                y=coords_2d[mask, 1],
                mode="markers",
                name=cluster_name,
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color=_BG_COLOR),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    # クラスタ重心をプロット（K-Meansの場合）
    if result.centroids is not None and result.method == "kmeans":
        centroids_2d = pca.transform(
            result.scaler.transform(
                result.centroids
            ) if result.scaler else result.centroids
        )
        # scalerがある場合は標準化後のcentroidsをPCA変換
        # 実際にはscaled centroidsをPCA変換する方が正確
        if result.centroids_scaled is not None:
            centroids_2d = pca.transform(result.centroids_scaled)

        fig.add_trace(
            go.Scatter(
                x=centroids_2d[:, 0],
                y=centroids_2d[:, 1],
                mode="markers+text",
                name="重心",
                marker=dict(
                    size=16,
                    color="white",
                    symbol="x",
                    line=dict(width=2, color="white"),
                ),
                text=[f"C{i}" for i in range(len(centroids_2d))],
                textposition="top center",
                textfont=dict(color="white", size=11),
                hovertemplate="クラスタ %{text} 重心<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"🗺️ {game_name} クラスタ散布図（PCA 2D）",
            font=dict(size=20, color=_TEXT_COLOR),
            x=0.5,
        ),
        xaxis=dict(
            title=f"第1主成分 (寄与率: {exp_var[0]:.1%})",
            gridcolor=_GRID_COLOR,
            color=_TEXT_COLOR,
            zeroline=False,
        ),
        yaxis=dict(
            title=f"第2主成分 (寄与率: {exp_var[1]:.1%})",
            gridcolor=_GRID_COLOR,
            color=_TEXT_COLOR,
            zeroline=False,
        ),
        plot_bgcolor=_CARD_COLOR,
        paper_bgcolor=_BG_COLOR,
        font=dict(color=_TEXT_COLOR),
        legend=dict(
            bgcolor=_CARD_COLOR,
            bordercolor=_GRID_COLOR,
            borderwidth=1,
            font=dict(color=_TEXT_COLOR),
        ),
        hoverlabel=dict(
            bgcolor=_CARD_COLOR,
            font_size=13,
            font_color=_TEXT_COLOR,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        height=550,
    )

    return fig


def _build_donut_chart(
    result: ClusteringResult,
    game_name: str,
) -> go.Figure:
    """
    グラフ2: クラスタサイズ分布（ドーナツチャート）
    """
    labels_sorted = sorted(result.cluster_sizes.keys())
    names = []
    values = []
    colors = []

    for label in labels_sorted:
        if label == -1:
            names.append("ノイズ")
        else:
            names.append(f"クラスタ {label}")
        values.append(result.cluster_sizes[label])
        colors.append(_get_cluster_color(label))

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=names,
            values=values,
            hole=0.5,
            marker=dict(
                colors=colors,
                line=dict(color=_BG_COLOR, width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color=_TEXT_COLOR),
            hovertemplate="<b>%{label}</b><br>件数: %{value:,}<br>割合: %{percent}<extra></extra>",
        )
    )

    total = sum(values)
    fig.update_layout(
        title=dict(
            text=f"📊 {game_name} クラスタサイズ分布",
            font=dict(size=20, color=_TEXT_COLOR),
            x=0.5,
        ),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br>回",
                x=0.5, y=0.5,
                font_size=18,
                font_color=_TEXT_COLOR,
                showarrow=False,
            )
        ],
        plot_bgcolor=_BG_COLOR,
        paper_bgcolor=_BG_COLOR,
        font=dict(color=_TEXT_COLOR),
        legend=dict(
            bgcolor=_CARD_COLOR,
            bordercolor=_GRID_COLOR,
            borderwidth=1,
            font=dict(color=_TEXT_COLOR),
        ),
        hoverlabel=dict(
            bgcolor=_CARD_COLOR,
            font_size=13,
            font_color=_TEXT_COLOR,
        ),
        margin=dict(l=30, r=30, t=60, b=30),
        height=400,
    )

    return fig


def _build_boxplot_chart(
    features: np.ndarray,
    result: ClusteringResult,
    game_name: str,
) -> go.Figure:
    """
    グラフ3: クラスタ別の特徴量分布（箱ひげ図）
    """
    fig = go.Figure()

    unique_labels = sorted(set(result.labels.tolist()))
    # ノイズを除く
    cluster_labels = [l for l in unique_labels if l >= 0]

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        for label in cluster_labels:
            mask = result.labels == label
            values = features[mask, feat_idx]
            color = _get_cluster_color(label)

            fig.add_trace(
                go.Box(
                    y=values,
                    name=f"C{label}",
                    legendgroup=f"cluster_{label}",
                    showlegend=(feat_idx == 0),  # 凡例は最初の特徴量のみ
                    marker_color=color,
                    line_color=color,
                    boxmean=True,
                    offsetgroup=f"cluster_{label}",
                    x=[feat_name] * len(values),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"📦 {game_name} クラスタ別 特徴量分布",
            font=dict(size=20, color=_TEXT_COLOR),
            x=0.5,
        ),
        xaxis=dict(
            title="特徴量",
            gridcolor=_GRID_COLOR,
            color=_TEXT_COLOR,
        ),
        yaxis=dict(
            title="値",
            gridcolor=_GRID_COLOR,
            color=_TEXT_COLOR,
        ),
        boxmode="group",
        plot_bgcolor=_CARD_COLOR,
        paper_bgcolor=_BG_COLOR,
        font=dict(color=_TEXT_COLOR),
        legend=dict(
            title=dict(text="クラスタ", font=dict(color=_TEXT_COLOR)),
            bgcolor=_CARD_COLOR,
            bordercolor=_GRID_COLOR,
            borderwidth=1,
            font=dict(color=_TEXT_COLOR),
        ),
        hoverlabel=dict(
            bgcolor=_CARD_COLOR,
            font_size=13,
            font_color=_TEXT_COLOR,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        height=500,
    )

    return fig


def _build_prediction_heatmap(
    predictions: list[tuple[int, ...]],
    config: dict,
    game_name: str,
    strategy: str,
) -> go.Figure:
    """
    グラフ4: 予測番号ヒートマップ

    予測セット×数字のマトリクスで、選ばれた番号をハイライト。
    """
    range_max = config["range_max"]

    # マトリクス構築（予測セット × 全数字）
    n_preds = len(predictions)
    matrix = np.zeros((n_preds, range_max), dtype=int)
    for i, combo in enumerate(predictions):
        for num in combo:
            matrix[i, num - 1] = 1

    # 各数字が何セットに選ばれたか（下部のサマリー行用）
    num_freq = matrix.sum(axis=0)

    # サマリー行を追加
    matrix_with_summary = np.vstack([matrix, num_freq.reshape(1, -1)])

    # Y軸ラベル
    y_labels = [f"予測 {i+1}" for i in range(n_preds)] + ["合計"]
    x_labels = [str(n) for n in range(1, range_max + 1)]

    # テキスト表示（選ばれた番号のみ数字を表示）
    text_matrix = []
    for i in range(n_preds):
        row = []
        for j in range(range_max):
            row.append(str(j + 1) if matrix[i, j] else "")
        text_matrix.append(row)
    # サマリー行
    summary_text = [str(int(v)) if v > 0 else "" for v in num_freq]
    text_matrix.append(summary_text)

    strategy_names = {
        "centroid": "重心狙い",
        "recent": "直近クラスタ",
        "pocket": "ポケット狙い",
    }

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=matrix_with_summary,
            x=x_labels,
            y=y_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            colorscale=[
                [0, "#1a1a2e"],
                [0.5, "#0f3460"],
                [1, "#58a6ff"],
            ],
            showscale=False,
            hovertemplate="数字: %{x}<br>%{y}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"🎯 {game_name} 予測番号マップ（{strategy_names.get(strategy, strategy)}）",
            font=dict(size=20, color=_TEXT_COLOR),
            x=0.5,
        ),
        xaxis=dict(
            title="数字",
            tickmode="linear",
            dtick=1,
            side="bottom",
            color=_TEXT_COLOR,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            color=_TEXT_COLOR,
            autorange="reversed",
        ),
        plot_bgcolor=_BG_COLOR,
        paper_bgcolor=_BG_COLOR,
        font=dict(color=_TEXT_COLOR),
        hoverlabel=dict(
            bgcolor=_CARD_COLOR,
            font_size=13,
            font_color=_TEXT_COLOR,
        ),
        margin=dict(l=80, r=30, t=60, b=60),
        height=max(250, n_preds * 40 + 150),
    )

    return fig


def generate_cluster_report_html(
    result: ClusteringResult,
    features: np.ndarray,
    data: list[dict],
    config: dict,
    game_key: str,
    predictions: list[tuple[int, ...]] | None = None,
    strategy: str = "centroid",
    output_dir: str = "output",
    filepath: str | None = None,
) -> str:
    """
    クラスタリング結果のインタラクティブHTMLレポートを生成する。

    Args:
        result: クラスタリング結果
        features: 特徴量行列（標準化前）
        data: 当選データリスト
        config: LOTTERY_CONFIG[game_key]
        game_key: ゲームキー
        predictions: 予測番号リスト（省略時は予測マップなし）
        strategy: 予測戦略名
        output_dir: 出力ディレクトリ
        filepath: 出力ファイルパス（省略時は自動生成）

    Returns:
        保存したHTMLファイルのパス
    """
    _ensure_output_dir(output_dir)

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = result.method
        filepath = os.path.join(
            output_dir,
            f"cluster_{method}_{game_key.lower()}_{timestamp}.html",
        )

    game_name = config["name"]
    range_max = config["range_max"]
    pick_size = config["pick_size"]
    method_name = "K-Means" if result.method == "kmeans" else "DBSCAN"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── グラフ生成 ──
    # StandardScaler で標準化してPCAにかけるため、
    # scaler を通した特徴量を使う
    if result.scaler is not None:
        features_for_pca = result.scaler.transform(features)
    else:
        features_for_pca = features

    fig_scatter = _build_scatter_chart(features_for_pca, result, data, game_name)
    fig_donut = _build_donut_chart(result, game_name)
    fig_box = _build_boxplot_chart(features, result, game_name)

    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)
    donut_html = fig_donut.to_html(full_html=False, include_plotlyjs=False)
    box_html = fig_box.to_html(full_html=False, include_plotlyjs=False)

    # 予測ヒートマップ（予測がある場合のみ）
    pred_section = ""
    if predictions:
        fig_pred = _build_prediction_heatmap(predictions, config, game_name, strategy)
        pred_html = fig_pred.to_html(full_html=False, include_plotlyjs=False)
        pred_section = f"""
    <div class="chart-section">
        {pred_html}
    </div>"""

    # ── 統計カード ──
    stat_cards = f"""
        <div class="stat-card">
            <div class="label">ゲーム</div>
            <div class="value">{game_name}</div>
        </div>
        <div class="stat-card">
            <div class="label">手法</div>
            <div class="value">{method_name}</div>
        </div>
        <div class="stat-card">
            <div class="label">データ数</div>
            <div class="value">{len(data):,}</div>
        </div>
        <div class="stat-card">
            <div class="label">クラスタ数</div>
            <div class="value accent">{result.n_clusters}</div>
        </div>"""

    if result.noise_count > 0:
        stat_cards += f"""
        <div class="stat-card">
            <div class="label">ノイズ</div>
            <div class="value warn">{result.noise_count}</div>
        </div>"""

    if result.inertia is not None:
        stat_cards += f"""
        <div class="stat-card">
            <div class="label">イナーシャ</div>
            <div class="value">{result.inertia:,.0f}</div>
        </div>"""

    # ── HTML組み立て ──
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{game_name} クラスタリング分析結果（{method_name}）</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: {_BG_COLOR};
            color: {_TEXT_COLOR};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid {_GRID_COLOR};
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #58a6ff, #d2a8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .header .meta {{
            color: {_MUTED_COLOR};
            font-size: 0.9em;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: {_CARD_COLOR};
            border: 1px solid {_GRID_COLOR};
            border-radius: 8px;
            padding: 15px 25px;
            text-align: center;
            min-width: 120px;
        }}
        .stat-card .label {{
            color: {_MUTED_COLOR};
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        .stat-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: {_TEXT_COLOR};
        }}
        .stat-card .value.accent {{
            color: #58a6ff;
        }}
        .stat-card .value.warn {{
            color: #f78166;
        }}
        .chart-section {{
            background: {_CARD_COLOR};
            border: 1px solid {_GRID_COLOR};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
        }}
        .chart-row {{
            display: flex;
            gap: 25px;
            margin-bottom: 25px;
        }}
        .chart-row .chart-section {{
            flex: 1;
            margin-bottom: 0;
        }}
        @media (max-width: 900px) {{
            .chart-row {{
                flex-direction: column;
            }}
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #484f58;
            font-size: 0.8em;
            border-top: 1px solid {_GRID_COLOR};
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 {game_name} クラスタリング分析</h1>
        <p class="meta">{method_name} | 実行日時: {timestamp_str}</p>
    </div>

    <div class="stats">
        {stat_cards}
    </div>

    <div class="chart-row">
        <div class="chart-section">
            {scatter_html}
        </div>
        <div class="chart-section">
            {donut_html}
        </div>
    </div>

    <div class="chart-section">
        {box_html}
    </div>
    {pred_section}

    <footer>
        Loto Predictor - クラスタリング分析（{method_name}） | Generated by loto-predictor
    </footer>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath
