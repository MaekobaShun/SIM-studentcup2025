"""
【gemini-apiを使って提出する】
0: Pythonの環境構築をして必要なライブラリをインストールする()
1: .envファイルを作成して，GEMINI_API_KEY="your gemini api key" を設定する
2: 設定の部分を書き換える
    2-1: DATASET_TYPE を 'TRAIN' か 'TEST' に設定する(TRAINは例題データ，TESTは本番データ)
    2-2: MODEL_NAME を使用するgeminiモデル名に設定する
    2-3: MIN_ID, MAX_ID を対象とするfiction_storiesのID範囲に設定する
    2-4: BATCH_SIZE を一度に処理する件数に設定する
3: 以下のコマンドで実行する
    python src/comp/gemini-api-pred.py

【メモ】
今回はgemini-3-flash-preview(無料範囲で使える中で最もよいモデル)を使用する想定です
BATCH_SIZEを20にしたら時間がかかりすぎたため10にすることを推奨します
gemini-3-flash-previewは1日に20回までしか処理ができないため，1-170と171-340に分割して実行することを推奨します
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# =============================================================================
# 設定
# =============================================================================
DATASET_TYPE = "TEST"
MODEL_NAME = "gemini-3-flash-preview"
MIN_ID = 1
MAX_ID = 170
BATCH_SIZE = 10
# =============================================================================

# .envからAPIキーを読み込む
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY が .env に設定されていません")

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)


def load_data(base_path: str, fiction_path: str, dataset_type: str):
    """データを読み込む"""
    base_df = pd.read_csv(base_path, sep="\t")
    fiction_df = pd.read_csv(fiction_path, sep="\t")

    # TRAINデータの場合、行番号をIDとして追加（1-indexed）
    if dataset_type == "TRAIN":
        fiction_df = fiction_df.reset_index(drop=True)
        fiction_df["id"] = fiction_df.index + 1

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
        fiction_id = row["id"]
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


def query_gemini(prompt: str, model_name: str) -> str:
    """Gemini APIにクエリを送信する"""
    try:
        model = genai.GenerativeModel(model_name)

        # 生成設定
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
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

    # DATASET_TYPEに応じてファイルパスを切り替え
    if DATASET_TYPE == "TRAIN":
        fiction_path = "inputs/fiction_stories_practice.tsv"
    elif DATASET_TYPE == "TEST":
        fiction_path = "inputs/fiction_stories_test.tsv"
    else:
        raise ValueError(f"DATASET_TYPE は 'TRAIN' または 'TEST' を指定してください: {DATASET_TYPE}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"設定:")
    print(f"  DATASET_TYPE: {DATASET_TYPE}")
    print(f"  MODEL_NAME: {MODEL_NAME}")
    print(f"  MIN_ID: {MIN_ID}, MAX_ID: {MAX_ID}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print("=" * 80)
    print("データ読み込み中...")
    print("=" * 80)

    base_df, fiction_df = load_data(base_path, fiction_path, DATASET_TYPE)

    print(f"Base stories: {len(base_df)} 件")
    print(f"Fiction stories: {len(fiction_df)} 件")
    print()

    # 対象IDのデータをフィルタリング
    target_ids = list(range(MIN_ID, MAX_ID + 1))
    fiction_df_filtered = fiction_df[fiction_df["id"].isin(target_ids)]
    print(f"対象Fiction stories: {len(fiction_df_filtered)} 件 (ID {MIN_ID}-{MAX_ID})")
    print()

    # バッチ数を計算
    num_batches = (len(fiction_df_filtered) + BATCH_SIZE - 1) // BATCH_SIZE

    # 全バッチの結果を蓄積するリスト
    all_responses = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        # フィルタリング済みのfiction_dfから該当範囲を取得
        fiction_batch = fiction_df_filtered.iloc[start_idx:end_idx]

        if len(fiction_batch) == 0:
            print(f"Batch {batch_idx + 1}: データがありません。スキップします。")
            continue

        # IDの範囲を取得
        batch_start_id = fiction_batch.iloc[0]["id"]
        batch_end_id = fiction_batch.iloc[-1]["id"]
        id_range = f"{batch_start_id}-{batch_end_id}"

        print("=" * 80)
        print(f"Batch {batch_idx + 1}/{num_batches}: Fiction {id_range} ({len(fiction_batch)}件)")
        print("=" * 80)

        # プロンプト作成
        prompt = create_prompt(base_df, fiction_batch)
        print(f"プロンプトの文字数: {len(prompt)} 文字")

        # Gemini APIにリクエスト
        print(f"Gemini API ({MODEL_NAME}) に問い合わせ中...")
        response = query_gemini(prompt, MODEL_NAME)

        # デバッグ出力
        print()
        print("-" * 40)
        print("【Gemini APIの出力（デバッグ）】")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print()

        # 結果を蓄積
        all_responses.append({
            "batch_idx": batch_idx + 1,
            "id_range": id_range,
            "count": len(fiction_batch),
            "response": response,
        })

        print(f"Batch {batch_idx + 1}/{num_batches} 完了")
        print()

        # 次のリクエストまで5秒待機（最後のバッチ以外）
        if batch_idx < num_batches - 1:
            print("5秒待機中...")
            time.sleep(5)
            print()

    # すべてのバッチ結果を1つのファイルに保存
    # 形式: {DATASET_TYPE}_{MODEL_NAME}_{MIN_ID}_{MAX_ID}.txt
    output_file = output_dir / f"{DATASET_TYPE}_{MODEL_NAME}_{MIN_ID}_{MAX_ID}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DATASET_TYPE: {DATASET_TYPE}\n")
        f.write(f"MODEL_NAME: {MODEL_NAME}\n")
        f.write(f"Fiction IDs: {MIN_ID}-{MAX_ID}\n")
        f.write(f"総件数: {len(fiction_df_filtered)}\n")
        f.write(f"バッチ数: {num_batches}\n")
        f.write("=" * 80 + "\n\n")

        for resp in all_responses:
            f.write(f"【Batch {resp['batch_idx']}/{num_batches}: Fiction {resp['id_range']} ({resp['count']}件)】\n")
            f.write("-" * 40 + "\n")
            f.write(resp["response"])
            f.write("\n" + "-" * 40 + "\n\n")

    print("=" * 80)
    print(f"全バッチ処理完了")
    print(f"結果を保存しました: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
