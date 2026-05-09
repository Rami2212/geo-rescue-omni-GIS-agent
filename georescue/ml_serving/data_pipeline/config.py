from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

SENTINEL_CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID")
SENTINEL_CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET")

COLOMBO_BBOX = [80.65, 6.80, 81.05, 7.10]
COLOMBO_CENTER_LAT = 6.9271
COLOMBO_CENTER_LON = 79.8612

BASE_DIR = Path(__file__).parent.parent
TRAINING_DATA_DIR = BASE_DIR / "data"
RAW_DIR = TRAINING_DATA_DIR / "raw"
PROCESSED_DIR = TRAINING_DATA_DIR / "processed"
LABELS_DIR = TRAINING_DATA_DIR / "labels"
