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
# Valid MODEL_ARCH options in config.py:
# MODEL_ARCH = "unet"           # Standard UNet
# MODEL_ARCH = "unetplusplus"   # UNet++ (dense skip connections)
# MODEL_ARCH = "fpn"            # Feature Pyramid Network
# MODEL_ARCH = "linknet"        # LinkNet
# MODEL_ARCH = "pspnet"         # PSPNet (Pyramid Scene Parsing)
# MODEL_ARCH = "deeplabv3"      # DeepLabV3
# MODEL_ARCH = "deeplabv3plus"  # DeepLabV3+
MODEL_ARCH = "unet"
ENCODER = "efficientnet-b3"  # can be changed to resnet34, resnet50, efficientnet-b0, etc.
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 4  # 4 defect classes
ACTIVATION = None  # None for logits

# Training configuration
SEED = 69
NUM_FOLDS = 5
TRAIN_FOLDS = [0, 1, 2, 3, 4]  # which folds to train
NUM_EPOCHS = 100
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 8
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
# Options: 
#   - "bce_with_logits": Standard BCE loss
#   - "dice": Dice loss only
#   - "combo": 0.5*BCE + 0.5*Dice
#   - "bce_pos_weight": BCE with pos_weight=(2.0, 2.0, 1.0, 1.5)
#   - "bce_dice_pos_weight": 0.75*BCE + 0.25*Dice with pos_weight=(2.0, 2.0, 1.0, 1.5)
LOSS = "bce_dice_pos_weight"

# Crop vs Full Image Mode
# Set USE_CROPS = True to train on 256x512 crops instead of full 256x1600 images
# Benefits of crops: 6x more training samples, larger batch sizes, better generalization
USE_CROPS = True           # Toggle between crop mode (True) and full image mode (False)
CROP_HEIGHT = 256           # Height of crops (same as image height)
CROP_WIDTH = 512            # Width of crops (1/3 of full width)
CROP_STRIDE = 256           # Stride for sliding window crops
                            # stride=256: 50% overlap, 6 crops per image (recommended)
                            # stride=512: no overlap, 4 crops per image

# Pre-cropped Images Mode (Memory Optimization)
# Set USE_PRECROPPED = True to load pre-cropped images from disk instead of cropping on-the-fly
# Benefits: Reduces memory usage by 3x, faster data loading, eliminates repeated full-image loads
# Setup: Run "bash precrop_setup.sh" or "python precrop_images.py --output_dir <path>" once before training
# Note: When USE_PRECROPPED=True, USE_CROPS setting is ignored (pre-cropped always uses crops)
USE_PRECROPPED = False     # Toggle pre-cropped mode (requires pre-processing step)
PRECROPPED_DIR = "/media/frank/ext_ssd2/kaggle_steel_old_format/precropped_data"  # Directory containing pre-cropped images

# Augmentation configuration
IMG_HEIGHT = 256
IMG_WIDTH = 512 if USE_CROPS else 1600  # Automatically adjust based on mode
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Data augmentation probabilities and parameters
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_VERTICAL_FLIP_PROB = 0.5
AUG_RANDOM_BRIGHTNESS_CONTRAST_PROB = 0.5
AUG_BRIGHTNESS_LIMIT = 0.2  # ±20% brightness
AUG_CONTRAST_LIMIT = 0.2    # ±20% contrast

# Metrics configuration
THRESHOLD = 0.5
MIN_SIZE = 3500  # minimum component size for post-processing

# Inference configuration
TTA = False  # test-time augmentation
TTA_HORIZONTAL_FLIP = True

# Logging
LOG_INTERVAL = 50  # log every N batches
SAVE_BEST_ONLY = True  # save all epoch checkpoints if False (consumes ~38 GB per fold)

# OOF Predictions Memory Optimization
# Store OOF predictions as binary masks (uint8) instead of probabilities (float32)
# Saves 75% memory (2.1 MB -> 0.52 MB per crop) but loses probability information
OOF_STORE_BINARY = False  # Set to True to save memory, False to keep probabilities

# Environment
CONDA_ENV = "compute_gpu_3_10"

