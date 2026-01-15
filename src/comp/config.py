from pathlib import Path
from dotenv import load_dotenv
import os

# プロジェクトルート(今回はStudentCup2025)
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 環境変数を取得
# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "bge-m3")
MODEL_PATH = os.getenv("MODEL_PATH", "BAAI/bge-m3")

# Inference Configuration
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
INFER_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "32"))

# Path Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
INPUT_DIR = os.getenv("INPUT_DIR", "inputs")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")