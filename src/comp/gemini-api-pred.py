"""
Gemini API (gemini-3.0-flash) を使ったストーリーマッチング
base_stories.tsv と fiction_stories_test.tsv を与え、
どのstoryとどのstoryのペアがあらすじの元となっているかを一括推論する
20件ずつ17回に分けてリクエストを送信
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# .envからAPIキーを読み込む
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY が .env に設定されていません")

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)


def load_data(base_path: str, fiction_path: str):
    """データを読み込む"""
    base_df = pd.read_csv(base_path, sep="\t")
    fiction_df = pd.read_csv(fiction_path, sep="\t")
    return base_df, fiction_df


def create_prompt(base_df: pd.DataFrame, fiction_batch: pd.DataFrame) -> str:
    """Geminiに送るプロンプトを作成する（バッチ用）"""

    # base_storiesの情報を整形
    base_stories_text = "【元となる作品リスト（base_stories）】\n"
    for _, row in base_df.iterrows():
        base_stories_text += f"ID {row['id']}: {row['title']}\n"
        base_stories_text += f"あらすじ: {row['story']}\n\n"

    # fiction_storiesの情報を整形（元のIDを使用）
    fiction_stories_text = "【合成されたあらすじリスト（fiction_stories）】\n"
    fiction_ids = []
    for _, row in fiction_batch.iterrows():
        fiction_id = row['id']
        fiction_ids.append(fiction_id)
        fiction_stories_text += f"Fiction {fiction_id}:\n"
        fiction_stories_text += f"{row['story']}\n\n"

    id_list = ", ".join(map(str, fiction_ids))

    prompt = f"""あなたは物語分析のエキスパートです。

以下に「元となる作品リスト（base_stories）」と「合成されたあらすじリスト（fiction_stories）」を提示します。

各「合成されたあらすじ」は、「元となる作品リスト」の中から【ちょうど2つの作品】を混ぜ合わせて作られています。
固有名詞は削除されています。

あなたの仕事は、各fiction_storyについて、どの2つのbase_storyが元になっているかを特定することです。

{base_stories_text}

{fiction_stories_text}

【タスク】
上記のすべてのFiction（Fiction {id_list}）について、
それぞれどの2つのbase_story（IDで指定）が元になっているかを分析し、回答してください。

【分析のヒント】
- あらすじの構造、設定、特異なイベント、結末の類似性に注目してください
- 世界観（SF、戦争、現代など）の組み合わせを考慮してください
- 特徴的なアイテム、概念、キャラクター設定を手がかりにしてください

【回答形式】
各Fictionについて以下の形式で回答してください：

Fiction X: ID A, ID B
（分析理由を簡潔に記載）

...（すべてのFictionについて続ける）

それでは、すべてのFictionについて分析を開始してください。
"""
    return prompt


def query_gemini(prompt: str, model_name: str = "gemini-3-flash-preview") -> str:
    """Gemini APIにクエリを送信する"""
    try:
        model = genai.GenerativeModel(model_name)

        # 生成設定
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=32768,
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        return response.text
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return f"ERROR: {e}"


def main():
    base_path = "inputs/base_stories.tsv"
    fiction_path = "inputs/fiction_stories_test.tsv"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("データ読み込み中...")
    print("=" * 80)

    base_df, fiction_df = load_data(base_path, fiction_path)

    print(f"Base stories: {len(base_df)} 件")
    print(f"Fiction stories: {len(fiction_df)} 件")
    print()

    # バッチ処理の設定
    batch_size = 10  # 1度に送るIDの数
    # 予測対象の区間を定義
    target_ranges = [
        (131, 140),
        (151, 160),
        (211, 220),
        (271, 280),
        (311, 320),
        (331, 340),
    ]

    # 各区間からIDリストを作成
    target_ids = []
    for start_id, end_id in target_ranges:
        target_ids.extend(range(start_id, end_id + 1))

    # 対象IDのデータをフィルタリング
    fiction_df_filtered = fiction_df[fiction_df['id'].isin(target_ids)]
    print(f"対象Fiction stories: {len(fiction_df_filtered)} 件")
    print(f"対象区間: {target_ranges}")
    print()

    # バッチ数を計算
    num_batches = (len(fiction_df_filtered) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # フィルタリング済みのfiction_dfから該当範囲を取得
        fiction_batch = fiction_df_filtered.iloc[start_idx:end_idx]

        if len(fiction_batch) == 0:
            print(f"Batch {batch_idx + 1}: データがありません。スキップします。")
            continue

        # IDの範囲を取得
        start_id = fiction_batch.iloc[0]['id']
        end_id = fiction_batch.iloc[-1]['id']
        id_range = f"{start_id}-{end_id}"

        print("=" * 80)
        print(f"Batch {batch_idx + 1}/{num_batches}: Fiction {id_range} ({len(fiction_batch)}件)")
        print("=" * 80)

        # プロンプト作成
        prompt = create_prompt(base_df, fiction_batch)
        print(f"プロンプトの文字数: {len(prompt)} 文字")

        # Gemini APIにリクエスト
        print("Gemini API (gemini-3-flash-preview) に問い合わせ中...")
        response = query_gemini(prompt)

        # デバッグ出力
        print()
        print("-" * 40)
        print("【Gemini APIの出力（デバッグ）】")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print()

        # ファイルに保存
        output_file = output_dir / f"gemini3flash_v7_{id_range}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Batch: {batch_idx + 1}/{num_batches}\n")
            f.write(f"Fiction IDs: {id_range}\n")
            f.write(f"件数: {len(fiction_batch)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(response)

        print(f"結果を保存しました: {output_file}")
        print()

        # 次のリクエストまで5秒待機（最後のバッチ以外）
        if batch_idx < num_batches - 1:
            print("5秒待機中...")
            time.sleep(5)
            print()

    print("=" * 80)
    print("全バッチ処理完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
