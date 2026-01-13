"""
Configuration file for UNet baseline training
"""
import os
from datetime import datetime

# Data paths
DATA_ROOT = "/home/frank/Dropbox/Project/kaggle_severstal/input/"
# DATA_ROOT = "/media/frank/ext_ssd2/kaggle_steel/"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")

# Output directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration
MODEL_ARCH = "unet"
ENCODER = "resnet18"  # can be changed to resnet34, resnet50, efficientnet-b0, etc.
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 4  # 4 defect classes
ACTIVATION = None  # None for logits

# Training configuration
SEED = 69
NUM_FOLDS = 5
TRAIN_FOLDS = [0, 1, 2, 3, 4]  # which folds to train
NUM_EPOCHS = 25
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL = 4
ACCUMULATION_STEPS = 1  # Set to 1 to disable gradient accumulation (saves RAM)
NUM_WORKERS = 2

# Optimizer configuration
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
OPTIMIZER = "adam"  # adam, adamw, sgd

# Scheduler configuration
SCHEDULER = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-7

# Loss configuration
LOSS = "bce_with_logits"  # bce_with_logits, dice, focal, combo

# Augmentation configuration
IMG_HEIGHT = 256
IMG_WIDTH = 1600
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
AUG_HORIZONTAL_FLIP_PROB = 0.5

# Metrics configuration
THRESHOLD = 0.5
MIN_SIZE = 3500  # minimum component size for post-processing

# Inference configuration
TTA = False  # test-time augmentation
TTA_HORIZONTAL_FLIP = True

# Logging
LOG_INTERVAL = 50  # log every N batches
SAVE_BEST_ONLY = False  # save all epoch checkpoints if False

# Environment
CONDA_ENV = "compute_gpu_3_10"

