"""
Test script to verify setup and data paths
"""
import os
import sys

print("Testing setup...")
print("="*80)

# Test imports
print("\n1. Testing imports...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"   ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"   ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"   ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"   ✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"   ✗ Pandas import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import albumentations as A
    print(f"   ✓ Albumentations {A.__version__}")
except ImportError as e:
    print(f"   ✗ Albumentations import failed: {e}")
    sys.exit(1)

try:
    import segmentation_models_pytorch as smp
    print(f"   ✓ Segmentation Models PyTorch {smp.__version__}")
except ImportError as e:
    print(f"   ✗ Segmentation Models PyTorch import failed: {e}")
    sys.exit(1)

# Test local imports
print("\n2. Testing local imports...")
try:
    import config
    print("   ✓ config.py")
except ImportError as e:
    print(f"   ✗ config.py import failed: {e}")
    sys.exit(1)

try:
    from dataset import prepare_trainval_dataframe, SteelDataset
    print("   ✓ dataset.py")
except ImportError as e:
    print(f"   ✗ dataset.py import failed: {e}")
    sys.exit(1)

try:
    from model import get_model
    print("   ✓ model.py")
except ImportError as e:
    print(f"   ✗ model.py import failed: {e}")
    sys.exit(1)

try:
    from losses import MetricTracker, BCEDiceLoss
    print("   ✓ losses.py")
except ImportError as e:
    print(f"   ✗ losses.py import failed: {e}")
    sys.exit(1)

# Test data paths
print("\n3. Testing data paths...")
data_root = config.DATA_ROOT
print(f"   Data root: {data_root}")

if os.path.exists(data_root):
    print(f"   ✓ Data root exists")
else:
    print(f"   ✗ Data root does not exist: {data_root}")
    print("   Please check the DATA_ROOT path in config.py")
    sys.exit(1)

train_csv = config.TRAIN_CSV
if os.path.exists(train_csv):
    print(f"   ✓ train.csv exists")
    df = pd.read_csv(train_csv)
    print(f"   ✓ train.csv loaded: {len(df)} rows")
else:
    print(f"   ✗ train.csv does not exist: {train_csv}")
    sys.exit(1)

train_images_dir = config.TRAIN_IMAGES_DIR
if os.path.exists(train_images_dir):
    print(f"   ✓ train_images directory exists")
    num_images = len([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])
    print(f"   ✓ Found {num_images} training images")
else:
    print(f"   ✗ train_images directory does not exist: {train_images_dir}")
    sys.exit(1)

test_images_dir = config.TEST_IMAGES_DIR
if os.path.exists(test_images_dir):
    print(f"   ✓ test_images directory exists")
    num_test_images = len([f for f in os.listdir(test_images_dir) if f.endswith('.jpg')])
    print(f"   ✓ Found {num_test_images} test images")
else:
    print(f"   ✗ test_images directory does not exist: {test_images_dir}")

# Test model creation
print("\n4. Testing model creation...")
try:
    model = get_model(
        arch=config.MODEL_ARCH,
        encoder=config.ENCODER,
        encoder_weights=None,  # Don't download weights for testing
        num_classes=config.NUM_CLASSES,
        activation=config.ACTIVATION
    )
    print(f"   ✓ Model created: {config.MODEL_ARCH} with {config.ENCODER}")
    
    # Test forward pass
    x = torch.randn(1, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    with torch.no_grad():
        y = model(x)
    print(f"   ✓ Forward pass successful: {x.shape} -> {y.shape}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test dataframe preparation
print("\n5. Testing dataframe preparation...")
try:
    df = prepare_trainval_dataframe(config.TRAIN_CSV, n_folds=config.NUM_FOLDS, 
                                   seed=config.SEED)
    print(f"   ✓ Dataframe prepared: {len(df)} samples")
    print(f"   ✓ Fold distribution:")
    for fold in range(config.NUM_FOLDS):
        fold_count = (df['fold'] == fold).sum()
        print(f"      Fold {fold}: {fold_count} samples")
except Exception as e:
    print(f"   ✗ Dataframe preparation failed: {e}")
    sys.exit(1)

# Test output directory
print("\n6. Testing output directory...")
output_dir = config.OUTPUT_DIR
if os.path.exists(output_dir):
    print(f"   ✓ Output directory exists: {output_dir}")
else:
    os.makedirs(output_dir, exist_ok=True)
    print(f"   ✓ Output directory created: {output_dir}")

print("\n" + "="*80)
print("All tests passed! ✓")
print("="*80)
print("\nYou can now run training with:")
print("  bash train_single_fold.sh 0")
print("  or")
print("  bash train_all_folds.sh")




