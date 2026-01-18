import pandas as pd
from pathlib import Path
from collections import Counter


def load_submission(filepath: Path) -> pd.DataFrame:
    """submissionファイルを読み込む"""
    df = pd.read_csv(filepath, header=None, names=["id", "pred1", "pred2"])
    return df


def ensemble_predictions(preds_list: list[tuple[int, int]], priority_order: list[int]) -> tuple[int, int]:
    """
    複数の予測からアンサンブルして2つの予測値を返す

    Args:
        preds_list: 各submissionからの予測値のリスト [(pred1, pred2), ...]
        priority_order: 優先順位（インデックスのリスト、先頭が最優先）

    Returns:
        アンサンブルされた2つの予測値 (top1, top2)
    """
    # すべての予測値を集める（どのsubmissionから来たかも記録）
    all_preds_with_source = []
    for source_idx, (p1, p2) in enumerate(preds_list):
        all_preds_with_source.append((p1, source_idx, 0))  # (値, ソースインデックス, ソース内の位置)
        all_preds_with_source.append((p2, source_idx, 1))

    # 各値の出現回数をカウント
    values = [p[0] for p in all_preds_with_source]
    counter = Counter(values)

    # ソート用のキーを作成
    # 1. 出現回数が多い順（降順）
    # 2. 優先順位が高い順（priority_orderでのインデックスが小さい順）
    # 3. ソース内の位置（pred1が先）
    def sort_key(item):
        value, source_idx, pos_in_source = item
        count = counter[value]
        priority = priority_order.index(source_idx)
        return (-count, priority, pos_in_source)

    # ソート
    sorted_preds = sorted(all_preds_with_source, key=sort_key)

    # 重複を除いて上位2つを選択
    selected = []
    seen = set()
    for value, _, _ in sorted_preds:
        if value not in seen:
            selected.append(value)
            seen.add(value)
            if len(selected) == 2:
                break

    return (selected[0], selected[1])


def main():
    # src/comp/ から実行する場合のパス
    base_dir = Path(__file__).parent.parent.parent
    submission_dir = base_dir / "submission"

    # 3つのsubmissionファイルを読み込む
    sub1 = load_submission(submission_dir / "sub1_gemini3flash.csv")
    sub3 = load_submission(submission_dir / "sub3_gemini3flash_temp0.csv")
    sub6 = load_submission(submission_dir / "sub6_gemini2-5flash-preview.csv")

    print(f"sub1: {len(sub1)} rows")
    print(f"sub3: {len(sub3)} rows")
    print(f"sub6: {len(sub6)} rows")

    # 優先順位: sub1(0) -> sub3(1) -> sub6(2)
    priority_order = [0, 1, 2]

    results = []
    for idx in range(len(sub1)):
        row_id = sub1.iloc[idx]["id"]

        preds_list = [
            (sub1.iloc[idx]["pred1"], sub1.iloc[idx]["pred2"]),
            (sub3.iloc[idx]["pred1"], sub3.iloc[idx]["pred2"]),
            (sub6.iloc[idx]["pred1"], sub6.iloc[idx]["pred2"]),
        ]

        top1, top2 = ensemble_predictions(preds_list, priority_order)
        results.append((row_id, top1, top2))

    # 結果をDataFrameに変換して保存
    result_df = pd.DataFrame(results, columns=["id", "pred1", "pred2"])
    output_path = submission_dir / "sub7_ensamble.csv"
    result_df.to_csv(output_path, index=False, header=False)

    print(f"\nSaved to {output_path}")
    print(f"Total rows: {len(result_df)}")

    # サンプル出力
    print("\nSample output (first 10 rows):")
    for i in range(min(10, len(results))):
        row_id, p1, p2 = results[i]
        s1 = (sub1.iloc[i]["pred1"], sub1.iloc[i]["pred2"])
        s3 = (sub3.iloc[i]["pred1"], sub3.iloc[i]["pred2"])
        s6 = (sub6.iloc[i]["pred1"], sub6.iloc[i]["pred2"])
        print(f"  ID {row_id}: sub1={s1}, sub3={s3}, sub6={s6} -> ({p1}, {p2})")


if __name__ == "__main__":
    main()
