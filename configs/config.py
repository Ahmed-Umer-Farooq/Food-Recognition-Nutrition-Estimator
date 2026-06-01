# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

# Paths to the dataset and metadata
DATASET_PATH  = Path(os.environ.get("DATASET_PATH", "/content/drive/MyDrive/Datasets/Nutrition5k/nutrition5k_dataset"))
METADATA_PATH = DATASET_PATH / "metadata"
IMAGERY_PATH  = DATASET_PATH / "imagery" / "realsense_overhead"

# Constants
IMG_SIZE = 224  # Standard for EfficientNetB0
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 20

# Set Seeds for Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
