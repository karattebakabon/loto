"""
ロト予測ツール - クラスタリング分析 エンジン

K-Means / DBSCAN を統一インターフェースで提供する。
特徴量はStandardScalerで標準化してから処理する。
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    """クラスタリング結果を格納するデータクラス"""

    method: str
    """使用した手法（"kmeans" / "dbscan"）"""

    labels: np.ndarray
    """各データ点のクラスタラベル（DBSCANのノイズは-1）"""

    n_clusters: int
    """検出されたクラスタ数（ノイズを除く）"""

    centroids: np.ndarray | None = None
    """クラスタ重心（標準化前のスケール）。DBSCANの場合はNone"""

    centroids_scaled: np.ndarray | None = None
    """クラスタ重心（標準化後のスケール、内部計算用）"""

    scaler: StandardScaler | None = None
    """使用したStandardScaler（逆変換用）"""

    inertia: float | None = None
    """K-Meansのイナーシャ（クラスタ内二乗和）"""

    cluster_sizes: dict[int, int] = field(default_factory=dict)
    """各クラスタのデータ点数"""

    noise_count: int = 0
    """ノイズとして検出されたデータ点数（DBSCAN用）"""


def run_kmeans(
    features: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> ClusteringResult:
    """
    K-Meansクラスタリングを実行する。

    Args:
        features: 特徴量行列（shape=(n_samples, n_features)）
        n_clusters: クラスタ数
        random_state: 乱数シード

    Returns:
        ClusteringResult
    """
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means実行
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(features_scaled)

    # 重心を元のスケールに逆変換
    centroids_original = scaler.inverse_transform(km.cluster_centers_)

    # クラスタサイズの集計
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    return ClusteringResult(
        method="kmeans",
        labels=labels,
        n_clusters=n_clusters,
        centroids=centroids_original,
        centroids_scaled=km.cluster_centers_,
        scaler=scaler,
        inertia=km.inertia_,
        cluster_sizes=cluster_sizes,
    )


def run_dbscan(
    features: np.ndarray,
    eps: float | None = None,
    min_samples: int = 5,
) -> ClusteringResult:
    """
    DBSCANクラスタリングを実行する。

    Args:
        features: 特徴量行列
        eps: 近傍半径（Noneの場合はk-距離法で自動推定）
        min_samples: コアポイントの最小近傍数

    Returns:
        ClusteringResult
    """
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # eps自動推定（k-距離法）
    if eps is None:
        eps = _estimate_eps(features_scaled, min_samples)

    # DBSCAN実行
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features_scaled)

    # クラスタ数とノイズ数の集計
    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    noise_count = int(np.sum(labels == -1))

    # クラスタサイズの集計
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    # 各クラスタの重心を計算（ノイズを除く）
    centroids_scaled = None
    centroids_original = None
    if n_clusters > 0:
        centroid_list = []
        for label in sorted(unique_labels - {-1}):
            mask = labels == label
            centroid_list.append(features_scaled[mask].mean(axis=0))
        centroids_scaled = np.array(centroid_list)
        centroids_original = scaler.inverse_transform(centroids_scaled)

    return ClusteringResult(
        method="dbscan",
        labels=labels,
        n_clusters=n_clusters,
        centroids=centroids_original,
        centroids_scaled=centroids_scaled,
        scaler=scaler,
        cluster_sizes=cluster_sizes,
        noise_count=noise_count,
    )


def _estimate_eps(features_scaled: np.ndarray, min_samples: int) -> float:
    """
    k-距離法でDBSCANのepsパラメータを推定する。

    k-最近傍の距離を降順ソートし、「エルボー」（曲率最大点）を
    eps として採用する。

    Args:
        features_scaled: 標準化済み特徴量行列
        min_samples: DBSCANのmin_samplesと同じ値をkとして使用

    Returns:
        推定されたeps値
    """
    k = min(min_samples, len(features_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(features_scaled)
    distances, _ = nn.kneighbors(features_scaled)

    # k番目の最近傍距離を降順ソート
    k_distances = np.sort(distances[:, -1])[::-1]

    # 簡易エルボー検出: 二次差分の最大点
    if len(k_distances) < 3:
        return float(np.median(k_distances))

    diffs = np.diff(k_distances)
    second_diffs = np.diff(diffs)
    # 最大の曲率変化点をエルボーとする
    elbow_idx = int(np.argmax(np.abs(second_diffs))) + 1
    eps = float(k_distances[elbow_idx])

    # 安全マージン（epsが極端に小さい場合の補正）
    return max(eps, 0.1)


def find_optimal_k(
    features: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
) -> dict:
    """
    エルボー法でK-Meansの最適クラスタ数を推定する。

    Args:
        features: 特徴量行列
        k_range: 探索するkの範囲（デフォルト: 2〜10）
        random_state: 乱数シード

    Returns:
        {
            "k_values": [2, 3, ...],
            "inertias": [float, ...],
            "optimal_k": int,
        }
    """
    if k_range is None:
        max_k = min(10, len(features) - 1)
        k_range = range(2, max_k + 1)

    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    k_values = list(k_range)
    inertias = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(features_scaled)
        inertias.append(km.inertia_)

    # エルボー検出（二次差分の最大点）
    optimal_k = k_values[0]
    if len(inertias) >= 3:
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
        elbow_idx = second_diffs.index(max(second_diffs)) + 1
        optimal_k = k_values[elbow_idx]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "optimal_k": optimal_k,
    }
