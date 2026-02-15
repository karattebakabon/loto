"""
テスト - CSVデータ読み込みモジュール
"""

import pytest

from src.common import LOTTERY_CONFIG
from src.common.data_loader import load_lottery_data


class TestLoadLotteryData:
    """load_lottery_data() のテスト"""

    def test_load_loto6(self):
        """ロト6のCSVを正常に読み込めること"""
        data = load_lottery_data("LOTO6")
        assert len(data) > 0, "データが1件以上読み込まれること"

        # 最初の1件を検証
        first = data[0]
        assert "draw_no" in first
        assert "date" in first
        assert "main_numbers" in first
        assert "bonus_numbers" in first

        # 本数字は6個
        assert len(first["main_numbers"]) == 6
        # ボーナス数字は1個
        assert len(first["bonus_numbers"]) == 1

        # 数字が範囲内
        for n in first["main_numbers"]:
            assert 1 <= n <= 43, f"ロト6の数字は1〜43: {n}"

    def test_load_loto7(self):
        """ロト7のCSVを正常に読み込めること"""
        data = load_lottery_data("LOTO7")
        assert len(data) > 0

        first = data[0]
        # 本数字は7個
        assert len(first["main_numbers"]) == 7
        # ボーナス数字は2個
        assert len(first["bonus_numbers"]) == 2

        for n in first["main_numbers"]:
            assert 1 <= n <= 37, f"ロト7の数字は1〜37: {n}"

    def test_load_miniloto(self):
        """ミニロトのCSVを正常に読み込めること"""
        data = load_lottery_data("MINILOTO")
        assert len(data) > 0

        first = data[0]
        # 本数字は5個
        assert len(first["main_numbers"]) == 5
        # ボーナス数字は1個
        assert len(first["bonus_numbers"]) == 1

        for n in first["main_numbers"]:
            assert 1 <= n <= 31, f"ミニロトの数字は1〜31: {n}"

    def test_case_insensitive_key(self):
        """ゲームキーが大文字小文字を問わないこと"""
        data = load_lottery_data("loto6")
        assert len(data) > 0

    def test_invalid_game_key(self):
        """不正なゲームキーでValueError"""
        with pytest.raises(ValueError, match="不正なゲームキー"):
            load_lottery_data("INVALID")

    def test_data_order(self):
        """データが開催回順に並んでいること"""
        data = load_lottery_data("LOTO6")
        draw_numbers = [d["draw_no"] for d in data]
        assert draw_numbers == sorted(draw_numbers), "開催回が昇順であること"

    def test_main_numbers_sorted(self):
        """本数字が昇順ソートされていること"""
        data = load_lottery_data("LOTO6")
        for d in data[:10]:  # 先頭10件をチェック
            assert d["main_numbers"] == sorted(d["main_numbers"])

    def test_data_count(self):
        """十分なデータ件数が読み込まれること"""
        # ログに記載: ロト6=2,076回、ロト7=664回、ミニロト=1,373回
        data_loto6 = load_lottery_data("LOTO6")
        assert len(data_loto6) >= 2000, f"ロト6: {len(data_loto6)}件"

        data_loto7 = load_lottery_data("LOTO7")
        assert len(data_loto7) >= 600, f"ロト7: {len(data_loto7)}件"

        data_mini = load_lottery_data("MINILOTO")
        assert len(data_mini) >= 1300, f"ミニロト: {len(data_mini)}件"
