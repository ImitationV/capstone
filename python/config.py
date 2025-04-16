import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration
MODEL_SAVE_PATH = 'models/saved_models'
DATA_SAVE_PATH = 'data/raw'
FEATURES_SAVE_PATH = 'data/processed'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_SAVE_PATH, exist_ok=True)
os.makedirs(FEATURES_SAVE_PATH, exist_ok=True) 