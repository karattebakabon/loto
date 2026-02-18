"""
テスト - クラスタリング分析モジュール
"""

import pytest
import numpy as np

from src.common import LOTTERY_CONFIG
from src.common.data_loader import load_lottery_data
from src.clustering.feature_extractor import extract_features, get_feature_names, _extract_one
from src.clustering.engine import run_kmeans, run_dbscan, find_optimal_k, ClusteringResult
from src.clustering.predictor import generate_predictions


# ==========================================
# テスト用フィクスチャ
# ==========================================


@pytest.fixture
def loto6_data():
    """ロト6の実データ"""
    return load_lottery_data("LOTO6")


@pytest.fixture
def loto6_features(loto6_data):
    """ロト6の特徴量行列"""
    return extract_features(loto6_data, "LOTO6")


@pytest.fixture
def loto6_kmeans_result(loto6_features):
    """ロト6のK-Means結果"""
    return run_kmeans(loto6_features, n_clusters=5)


# ==========================================
# 特徴量抽出のテスト
# ==========================================


class TestFeatureExtractor:
    """特徴量抽出モジュールのテスト"""

    def test_feature_names_count(self):
        """特徴量名が6個であること"""
        names = get_feature_names()
        assert len(names) == 6

    def test_feature_names_immutable(self):
        """get_feature_names()がコピーを返すこと"""
        names1 = get_feature_names()
        names2 = get_feature_names()
        assert names1 == names2
        names1.append("ダミー")
        assert len(get_feature_names()) == 6

    def test_extract_one_basic(self):
        """1開催回分の特徴量が正しく計算されること"""
        numbers = [3, 12, 18, 25, 33, 41]
        result = _extract_one(numbers, range_max=43)

        assert len(result) == 6
        # 合計値
        assert result[0] == 132.0
        # 平均値
        assert result[1] == pytest.approx(22.0)
        # 標準偏差（正の値であること）
        assert result[2] > 0
        # 範囲幅
        assert result[3] == 38.0
        # 奇数率（3, 25, 33, 41 = 4個 → 4/6）
        assert result[4] == pytest.approx(4 / 6)
        # 上半分率（≥22: 25, 33, 41 = 3個 → 3/6）
        assert result[5] == pytest.approx(3 / 6)

    def test_extract_one_empty(self):
        """空の数字リストでゼロベクトルを返すこと"""
        result = _extract_one([], range_max=43)
        assert result == [0.0] * 6

    def test_extract_features_shape_loto6(self, loto6_data):
        """ロト6の特徴量行列の形状が正しいこと"""
        features = extract_features(loto6_data, "LOTO6")
        assert features.shape == (len(loto6_data), 6)
        assert features.dtype == np.float64

    def test_extract_features_shape_loto7(self):
        """ロト7の特徴量行列の形状が正しいこと"""
        data = load_lottery_data("LOTO7")
        features = extract_features(data, "LOTO7")
        assert features.shape == (len(data), 6)

    def test_extract_features_shape_miniloto(self):
        """ミニロトの特徴量行列の形状が正しいこと"""
        data = load_lottery_data("MINILOTO")
        features = extract_features(data, "MINILOTO")
        assert features.shape == (len(data), 6)

    def test_extract_features_no_nan(self, loto6_features):
        """特徴量にNaNが含まれないこと"""
        assert not np.any(np.isnan(loto6_features))

    def test_extract_features_invalid_key(self, loto6_data):
        """不正なゲームキーでValueError"""
        with pytest.raises(ValueError, match="不正なゲームキー"):
            extract_features(loto6_data, "INVALID")

    def test_extract_features_empty_data(self):
        """空データでValueError"""
        with pytest.raises(ValueError, match="データが空"):
            extract_features([], "LOTO6")


# ==========================================
# クラスタリングエンジンのテスト
# ==========================================


class TestClusteringEngine:
    """クラスタリングエンジンのテスト"""

    def test_kmeans_basic(self, loto6_features):
        """K-Meansが正常に動作すること"""
        result = run_kmeans(loto6_features, n_clusters=3)

        assert isinstance(result, ClusteringResult)
        assert result.method == "kmeans"
        assert result.n_clusters == 3
        assert len(result.labels) == len(loto6_features)
        assert result.centroids is not None
        assert result.centroids.shape == (3, 6)
        assert result.inertia is not None
        assert result.inertia >= 0

    def test_kmeans_labels_valid(self, loto6_features):
        """K-Meansのラベルが有効な範囲であること"""
        result = run_kmeans(loto6_features, n_clusters=5)
        assert set(result.labels.tolist()).issubset(set(range(5)))

    def test_kmeans_cluster_sizes(self, loto6_features):
        """クラスタサイズの合計がデータ数と一致すること"""
        result = run_kmeans(loto6_features, n_clusters=4)
        total = sum(result.cluster_sizes.values())
        assert total == len(loto6_features)

    def test_dbscan_basic(self, loto6_features):
        """DBSCANが正常に動作すること"""
        result = run_dbscan(loto6_features)

        assert isinstance(result, ClusteringResult)
        assert result.method == "dbscan"
        assert result.n_clusters >= 0
        assert len(result.labels) == len(loto6_features)

    def test_dbscan_noise_count(self, loto6_features):
        """DBSCANのノイズ数とクラスタサイズが整合すること"""
        result = run_dbscan(loto6_features)

        total_in_clusters = sum(v for k, v in result.cluster_sizes.items() if k >= 0)
        assert total_in_clusters + result.noise_count == len(loto6_features)

    def test_find_optimal_k(self, loto6_features):
        """エルボー法が辞書を返すこと"""
        result = find_optimal_k(loto6_features, k_range=range(2, 8))

        assert "k_values" in result
        assert "inertias" in result
        assert "optimal_k" in result
        assert result["optimal_k"] >= 2
        assert len(result["k_values"]) == len(result["inertias"])

    def test_find_optimal_k_inertia_decreasing(self, loto6_features):
        """イナーシャがk増加に伴い単調減少すること"""
        result = find_optimal_k(loto6_features, k_range=range(2, 8))
        inertias = result["inertias"]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1], f"イナーシャが減少していない: k={result['k_values'][i]}"


# ==========================================
# 予測生成のテスト
# ==========================================


class TestPredictor:
    """予測生成モジュールのテスト"""

    @pytest.mark.parametrize("strategy", ["centroid", "recent", "pocket"])
    def test_prediction_count(self, loto6_kmeans_result, loto6_data, loto6_features, strategy):
        """予測数が指定通りであること"""
        predictions = generate_predictions(
            loto6_kmeans_result,
            loto6_data,
            loto6_features,
            "LOTO6",
            n_predictions=3,
            strategy=strategy,
        )
        assert len(predictions) == 3

    @pytest.mark.parametrize("strategy", ["centroid", "recent", "pocket"])
    def test_prediction_pick_size(self, loto6_kmeans_result, loto6_data, loto6_features, strategy):
        """各予測が正しい個数の数字を含むこと"""
        predictions = generate_predictions(
            loto6_kmeans_result,
            loto6_data,
            loto6_features,
            "LOTO6",
            n_predictions=5,
            strategy=strategy,
        )
        for combo in predictions:
            assert len(combo) == 6, f"ロト6は6個: {combo}"

    @pytest.mark.parametrize("strategy", ["centroid", "recent", "pocket"])
    def test_prediction_range(self, loto6_kmeans_result, loto6_data, loto6_features, strategy):
        """予測番号がゲームの範囲内であること"""
        predictions = generate_predictions(
            loto6_kmeans_result,
            loto6_data,
            loto6_features,
            "LOTO6",
            n_predictions=5,
            strategy=strategy,
        )
        for combo in predictions:
            for num in combo:
                assert 1 <= num <= 43, f"範囲外: {num}"

    @pytest.mark.parametrize("strategy", ["centroid", "recent", "pocket"])
    def test_prediction_no_duplicates(self, loto6_kmeans_result, loto6_data, loto6_features, strategy):
        """予測番号に重複がないこと"""
        predictions = generate_predictions(
            loto6_kmeans_result,
            loto6_data,
            loto6_features,
            "LOTO6",
            n_predictions=5,
            strategy=strategy,
        )
        for combo in predictions:
            assert len(set(combo)) == len(combo), f"重複あり: {combo}"

    @pytest.mark.parametrize("strategy", ["centroid", "recent", "pocket"])
    def test_prediction_sorted(self, loto6_kmeans_result, loto6_data, loto6_features, strategy):
        """予測番号が昇順ソートされていること"""
        predictions = generate_predictions(
            loto6_kmeans_result,
            loto6_data,
            loto6_features,
            "LOTO6",
            n_predictions=5,
            strategy=strategy,
        )
        for combo in predictions:
            assert combo == tuple(sorted(combo)), f"未ソート: {combo}"

    def test_prediction_invalid_strategy(self, loto6_kmeans_result, loto6_data, loto6_features):
        """不正な戦略でValueError"""
        with pytest.raises(ValueError, match="不正な戦略"):
            generate_predictions(
                loto6_kmeans_result,
                loto6_data,
                loto6_features,
                "LOTO6",
                strategy="invalid",
            )

    def test_prediction_invalid_game_key(self, loto6_kmeans_result, loto6_data, loto6_features):
        """不正なゲームキーでValueError"""
        with pytest.raises(ValueError, match="不正なゲームキー"):
            generate_predictions(
                loto6_kmeans_result,
                loto6_data,
                loto6_features,
                "INVALID",
            )

    def test_prediction_loto7(self):
        """ロト7でも正しく動作すること"""
        data = load_lottery_data("LOTO7")
        features = extract_features(data, "LOTO7")
        result = run_kmeans(features, n_clusters=3)
        predictions = generate_predictions(
            result,
            data,
            features,
            "LOTO7",
            n_predictions=3,
        )
        for combo in predictions:
            assert len(combo) == 7
            for num in combo:
                assert 1 <= num <= 37

    def test_prediction_miniloto(self):
        """ミニロトでも正しく動作すること"""
        data = load_lottery_data("MINILOTO")
        features = extract_features(data, "MINILOTO")
        result = run_kmeans(features, n_clusters=3)
        predictions = generate_predictions(
            result,
            data,
            features,
            "MINILOTO",
            n_predictions=3,
        )
        for combo in predictions:
            assert len(combo) == 5
            for num in combo:
                assert 1 <= num <= 31
