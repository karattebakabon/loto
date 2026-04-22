"""
ロト予測ツール - CSVデータ自動更新スクリプト

loto-life.net から最新の当選番号CSVをダウンロードし、
既存の data/raw/ CSVと差分マージする。

使用方法:
    python -m src.verification.update_data
    python -m src.verification.update_data --game loto6
    python -m src.verification.update_data --dry-run
"""

import argparse
import csv
import io
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional

# ==========================================
# 定数
# ==========================================

# loto-life.net CSVダウンロードURL
_DOWNLOAD_URLS: dict[str, str] = {
    "LOTO6": "https://loto-life.net/csv/loto6",
    "LOTO7": "https://loto-life.net/csv/loto7",
    "MINILOTO": "https://loto-life.net/csv/mini",
}

# 既存CSVファイル名
_CSV_FILENAMES: dict[str, str] = {
    "LOTO6": "loto6.csv",
    "LOTO7": "loto7.csv",
    "MINILOTO": "miniloto.csv",
}

REQUEST_TIMEOUT = 30  # 秒


def _get_data_dir() -> str:
    """data/raw/ ディレクトリのパスを返す"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "raw")


def _download_csv(game_key: str) -> list[list[str]]:
    """
    loto-life.net から最新CSVをダウンロードし、行リストを返す。

    Returns:
        [ [col0, col1, ...], ... ]  ヘッダー含む
    """
    url = _DOWNLOAD_URLS[game_key]
    print(f"   ダウンロード中: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://loto-life.net/",
    }
    last_err: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(5 * attempt)
            print(f"   リトライ {attempt}/2...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read()
            break
        except urllib.error.URLError as e:
            last_err = e
    else:
        raise ConnectionError(f"ダウンロード失敗: {last_err}") from last_err

    # Shift-JIS でデコード
    text = raw.decode("shift_jis", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row]
    return rows


def _load_existing_csv(game_key: str, data_dir: str) -> list[list[str]]:
    """
    既存の data/raw/xxx.csv を読み込む。
    Returns: ヘッダー含む行リスト
    """
    path = os.path.join(data_dir, _CSV_FILENAMES[game_key])
    if not os.path.exists(path):
        return []

    with open(path, encoding="shift_jis", newline="") as f:
        reader = csv.reader(f)
        return [row for row in reader if row]


def _get_draw_no(row: list[str]) -> Optional[int]:
    """行から開催回番号を取得する。変換失敗時はNone"""
    try:
        return int(row[0])
    except (ValueError, IndexError):
        return None


def merge_rows(
    existing: list[list[str]],
    downloaded: list[list[str]],
) -> tuple[list[list[str]], int]:
    """
    既存データと新規ダウンロードデータをマージする。

    Args:
        existing:   既存CSVの行リスト（ヘッダー含む）
        downloaded: 新規CSVの行リスト（ヘッダー含む）

    Returns:
        (マージ後の行リスト, 追加された行数)
    """
    if not downloaded:
        return existing, 0

    header = downloaded[0]

    # 既存の開催回セット
    existing_nos: set[int] = set()
    if len(existing) > 1:
        for row in existing[1:]:
            no = _get_draw_no(row)
            if no is not None:
                existing_nos.add(no)

    # 新規行を抽出
    new_rows: list[list[str]] = []
    for row in downloaded[1:]:
        no = _get_draw_no(row)
        if no is not None and no not in existing_nos:
            new_rows.append(row)

    if not existing:
        # 既存なし: ダウンロード全体をそのまま使う
        merged = downloaded
    else:
        # 既存あり: データ行を結合し、開催回で昇順ソート
        all_data = existing[1:] + new_rows
        all_data.sort(key=lambda r: _get_draw_no(r) or 0)
        merged = [header] + all_data

    return merged, len(new_rows)


def save_csv(rows: list[list[str]], game_key: str, data_dir: str) -> str:
    """
    マージ後データを Shift-JIS CSV として保存する。

    Returns:
        保存先パス
    """
    path = os.path.join(data_dir, _CSV_FILENAMES[game_key])
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerows(rows)
    csv_text = buf.getvalue()

    with open(path, "w", encoding="shift_jis", errors="ignore", newline="") as f:
        f.write(csv_text)

    return path


def update_game(
    game_key: str,
    data_dir: str,
    dry_run: bool = False,
) -> dict:
    """
    指定ゲームのCSVを更新する。

    Returns:
        {
            "game_key": str,
            "before_count": int,
            "after_count": int,
            "added_count": int,
            "latest_draw_no": int,
            "latest_date": str,
        }
    """
    print(f"\n🔄 {game_key} を更新中...")

    existing = _load_existing_csv(game_key, data_dir)
    before_count = max(0, len(existing) - 1)  # ヘッダー除く

    # ダウンロード
    downloaded = _download_csv(game_key)
    print(f"   サーバー側データ: {len(downloaded) - 1}件")

    # マージ
    merged, added = merge_rows(existing, downloaded)
    after_count = max(0, len(merged) - 1)

    # 最終回情報
    latest_row = merged[-1] if len(merged) > 1 else ["?", "?"]
    latest_draw_no = _get_draw_no(latest_row) or 0
    latest_date = latest_row[1] if len(latest_row) > 1 else "?"

    if not dry_run:
        path = save_csv(merged, game_key, data_dir)
        print(f"   保存先: {path}")
    else:
        print(f"   [DRY RUN] 保存スキップ")

    print(f"   更新前: {before_count}件  →  更新後: {after_count}件  (+{added}件)")
    print(f"   最終回: 第{latest_draw_no}回 ({latest_date})")

    return {
        "game_key": game_key,
        "before_count": before_count,
        "after_count": after_count,
        "added_count": added,
        "latest_draw_no": latest_draw_no,
        "latest_date": latest_date,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.verification.update_data",
        description="ロト当選番号CSVを loto-life.net から自動更新",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        choices=["loto6", "loto7", "miniloto"],
        help="更新するゲーム（省略時: 全ゲーム）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には保存しない（動作確認用）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="CSVディレクトリ（省略時: data/raw/）",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="ダウンロード失敗時にスキップして続行（リモートエージェント用）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or _get_data_dir()

    print("\n📥 ロト当選番号CSVデータ更新")
    print(f"   データディレクトリ: {data_dir}")
    if args.dry_run:
        print("   ⚠️  DRY RUN モード（ファイルは更新されません）")

    start_time = datetime.now()

    if args.game:
        games = [args.game.upper()]
    else:
        games = ["LOTO6", "LOTO7", "MINILOTO"]

    results = []
    errors = []
    for game_key in games:
        try:
            result = update_game(game_key, data_dir, dry_run=args.dry_run)
            results.append(result)
        except Exception as e:
            print(f"\n❌ {game_key} の更新に失敗: {e}", file=sys.stderr)
            errors.append(game_key)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'=' * 50}")
    print(f"✅ 更新完了（{elapsed:.1f}秒）")
    for r in results:
        status = f"+{r['added_count']}件" if r["added_count"] > 0 else "差分なし"
        print(f"   {r['game_key']:<10}: 第{r['latest_draw_no']}回 ({r['latest_date']}) [{status}]")
    if errors:
        if args.skip_on_error:
            print(f"\n⚠️  スキップ: {', '.join(errors)}（既存データで続行）", file=sys.stderr)
        else:
            print(f"\n❌ 失敗: {', '.join(errors)}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
