# Severstal Steel Defect Detection - UNet Baseline

A comprehensive deep learning solution for the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) competition, implementing semantic segmentation using UNet and related architectures with various encoder backbones.

## Competition Overview

The Severstal Steel Defect Detection challenge requires identifying and localizing defects in steel manufacturing images. Each image can contain multiple types of defects, and the goal is to accurately segment defect regions using run-length encoding (RLE).

### Defect Classes

The competition involves detecting **4 distinct types of defects**:

1. **Class 1**: Defect type 1 (specific characteristics)
2. **Class 2**: Defect type 2 (specific characteristics)
3. **Class 3**: Defect type 3 (specific characteristics)
4. **Class 4**: Defect type 4 (specific characteristics)

Each image (256×1600 pixels) may contain **zero, one, or multiple defect classes**, making this a multi-label semantic segmentation problem.

### Task

- **Input**: Steel manufacturing images (256×1600 pixels, RGB)
- **Output**: Run-length encoded (RLE) masks for each defect class
- **Evaluation**: Dice coefficient (F1 score for segmentation)

---

## Features

### Core Capabilities

- ✅ **Multi-class semantic segmentation** for 4 defect types
- ✅ **5-fold stratified cross-validation** (stratified by number of defects per image)
- ✅ **Out-of-fold (OOF) predictions** for ensemble modeling
- ✅ **Comprehensive metrics**: Dice, IoU, Precision, Recall (per-class and overall)
- ✅ **Model checkpointing** with best model saving
- ✅ **Resume training** from checkpoints
- ✅ **Test-time augmentation (TTA)** support
- ✅ **Model ensemble** for inference

### Training Modes

1. **Full Images** (256×1600): Baseline mode
2. **On-the-fly Crops** (256×512): 6x more training samples
3. **Pre-cropped Images** (256×512): Memory-optimized production mode ⭐

### Model Architectures

Supports multiple segmentation architectures via `segmentation-models-pytorch`:

- **UNet**: Standard U-shaped encoder-decoder
- **UNet++**: Dense skip connections
- **FPN**: Feature Pyramid Network
- **LinkNet**: Efficient encoder-decoder
- **PSPNet**: Pyramid Scene Parsing Network
- **DeepLabV3**: Atrous spatial pyramid pooling
- **DeepLabV3+**: Enhanced DeepLabV3

### Encoder Backbones

Compatible with any encoder from `segmentation-models-pytorch`:

- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **EfficientNet**: efficientnet-b0 through efficientnet-b7
- **SE-ResNeXt**: se_resnext50_32x4d, se_resnext101_32x4d
- **MobileNet**: mobilenet_v2
- **And many more...**

### Loss Functions

- `bce_with_logits`: Standard Binary Cross-Entropy
- `dice`: Dice loss for class imbalance
- `combo`: Combined BCE + Dice (0.5 each)
- `bce_pos_weight`: BCE with positive class weighting
- `bce_dice_pos_weight`: Combined 0.75*BCE + 0.25*Dice with pos_weight ⭐

---

## Installation

### Requirements

```bash
pip install torch torchvision opencv-python pandas numpy scikit-learn \
    albumentations segmentation-models-pytorch tqdm
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conda Environment

```bash
conda activate compute_gpu_3_10
```

---

## Data Structure

Organize your data as follows:

```
input/
├── train_images/
│   ├── 0002cc93b.jpg
│   ├── 000361c98.jpg
│   └── ...
├── test_images/
│   ├── 00a3ab3ee.jpg
│   └── ...
├── train.csv              # Format: ImageId, ClassId, EncodedPixels
└── sample_submission.csv  # Format: ImageId_ClassId, EncodedPixels
```

### Data Format

- **Images**: 256×1600 pixels, RGB, JPG format
- **train.csv**: Contains RLE-encoded masks for each defect class
- **sample_submission.csv**: Template for submission format

---

## Codebase Structure

```
code_unet_baseline_20260110/
├── config.py              # Main configuration file
├── model.py               # Model architecture definitions
├── dataset.py             # Dataset classes and data loaders
├── losses.py              # Loss functions and metrics
├── train.py               # Training script
├── inference.py           # Inference script
│
├── train_single_fold.sh   # Shell script for single fold training
├── train_all_folds.sh     # Shell script for all folds
├── inference.sh            # Shell script for inference
│
├── precrop_images.py      # Pre-cropping utility
├── precrop_setup.sh       # Pre-cropping setup script
│
├── test_crops.py          # Test crop mode
├── test_precropped.py     # Test pre-cropped mode
├── test_setup.py          # Test environment setup
│
├── outputs_*/             # Training outputs (timestamped)
│   ├── fold_0/
│   │   ├── best_model.pth
│   │   ├── oof_predictions.npy
│   │   └── history.csv
│   └── ...
│
└── Documentation/
    ├── README.md                    # This file
    ├── SETUP_GUIDE.md               # Detailed setup instructions
    ├── TRAINING_MODES.md            # Training mode comparison
    ├── PRECROPPED_MODE_GUIDE.md     # Pre-cropped mode guide
    ├── LOSS_FUNCTIONS_GUIDE.md      # Loss function details
    ├── CHECKPOINT_GUIDE.md          # Checkpoint management
    └── ...
```

### Key Files

- **`config.py`**: Central configuration for all hyperparameters
- **`model.py`**: Model factory function supporting multiple architectures
- **`dataset.py`**: Dataset classes for full images, crops, and pre-cropped data
- **`losses.py`**: Loss functions, metrics, and post-processing utilities
- **`train.py`**: Main training loop with validation and OOF generation
- **`inference.py`**: Inference script with ensemble support

---

## Quick Start

### 1. Configure Settings

Edit `config.py` to set your data path and preferences:

```python
DATA_ROOT = "/path/to/your/data/"
MODEL_ARCH = "unet"
ENCODER = "efficientnet-b3"
NUM_EPOCHS = 100
BATCH_SIZE_TRAIN = 8
```

### 2. Train a Single Fold

```bash
bash train_single_fold.sh 0
```

Or using Python:

```bash
python train.py --fold 0
```

### 3. Train All Folds

```bash
bash train_all_folds.sh
```

Or:

```bash
python train.py --all_folds
```

### 4. Generate Predictions

```bash
bash inference.sh outputs_YYYYMMDD_HHMMSS \
    outputs_YYYYMMDD_HHMMSS/fold_0/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_1/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_2/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_3/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_4/best_model.pth
```

---

## Configuration

### Model Configuration

```python
MODEL_ARCH = "unet"              # Architecture: unet, unetplusplus, fpn, linknet, etc.
ENCODER = "efficientnet-b3"      # Encoder: resnet18, resnet34, efficientnet-b0, etc.
ENCODER_WEIGHTS = "imagenet"     # Pretrained weights
NUM_CLASSES = 4                  # 4 defect classes
```

### Training Configuration

```python
NUM_FOLDS = 5                    # Cross-validation folds
NUM_EPOCHS = 100                 # Training epochs
BATCH_SIZE_TRAIN = 8             # Training batch size
BATCH_SIZE_VAL = 8               # Validation batch size
LEARNING_RATE = 5e-4             # Initial learning rate
OPTIMIZER = "adam"               # adam, adamw, sgd
SCHEDULER = "reduce_on_plateau"  # Learning rate scheduler
LOSS = "bce_dice_pos_weight"     # Loss function
```

### Training Modes

#### Mode 1: Full Images (Baseline)

```python
USE_CROPS = False
USE_PRECROPPED = False
IMG_HEIGHT = 256
IMG_WIDTH = 1600
```

- **Training samples**: ~2,800 per fold
- **Batch size**: 4-8 (limited by GPU memory)
- **Use case**: Quick baseline experiments

#### Mode 2: On-the-fly Crops

```python
USE_CROPS = True
USE_PRECROPPED = False
CROP_HEIGHT = 256
CROP_WIDTH = 512
CROP_STRIDE = 256  # 50% overlap → 6 crops per image
```

- **Training samples**: ~16,800 per fold (6x increase)
- **Batch size**: 8-12 (smaller images)
- **Use case**: Better performance, more training data

#### Mode 3: Pre-cropped Images ⭐ Recommended

```python
USE_CROPS = True
USE_PRECROPPED = True
PRECROPPED_DIR = "precropped_data"
```

**Setup** (one-time):

```bash
bash precrop_setup.sh
```

- **Training samples**: ~16,800 per fold
- **Memory**: 60% reduction vs on-the-fly crops
- **Speed**: 35% faster data loading
- **Use case**: Production training, memory-constrained systems

See `TRAINING_MODES.md` and `PRECROPPED_MODE_GUIDE.md` for detailed comparisons.

### Post-Processing Configuration

```python
THRESHOLD = 0.5                  # Probability threshold for binary mask
MIN_SIZE = 3500                  # Minimum component size (pixels)
```

### Inference Configuration

```python
TTA = False                      # Test-time augmentation
TTA_HORIZONTAL_FLIP = True       # TTA with horizontal flip
```

---

## Training

### Basic Training

Train a single fold:

```bash
python train.py --fold 0
```

Train all folds sequentially:

```bash
python train.py --all_folds
```

### Resume Training

Resume from a checkpoint:

```bash
python train.py --fold 0 --checkpoint outputs_YYYYMMDD_HHMMSS/fold_0/checkpoint_epoch_10.pth
```

This will:
- Load model weights, optimizer state, and training history
- Resume from the next epoch
- Preserve best Dice score and loss

### Training Output

Each training run creates a timestamped directory:

```
outputs_YYYYMMDD_HHMMSS/
├── training_log.txt              # Complete training log
├── fold_0/
│   ├── best_model.pth            # Best model checkpoint (by Dice score)
│   ├── checkpoint_epoch_N.pth     # Epoch checkpoints (if SAVE_BEST_ONLY=False)
│   ├── oof_predictions.npy       # Out-of-fold predictions (from best model)
│   └── history.csv               # Training history (loss, metrics per epoch)
├── fold_1/
│   └── ...
└── ...
```

### Metrics Tracked

**Overall Metrics:**
- Loss (BCE with logits)
- Dice score (overall, positive samples, negative samples)
- IoU (Intersection over Union)

**Per-Class Metrics** (for each of 4 defect classes):

*Segmentation Metrics:*
- Dice coefficient
- IoU
- Precision
- Recall

*Classification Metrics* (defect presence):
- Accuracy
- F1 score

All metrics are logged to console and saved to `training_log.txt`.

---

## Inference

### Single Model Inference

```bash
python inference.py \
    --checkpoints outputs_YYYYMMDD_HHMMSS/fold_0/best_model.pth \
    --output_dir outputs_YYYYMMDD_HHMMSS
```

### Ensemble Inference (Recommended)

Average predictions from multiple folds:

```bash
python inference.py \
    --checkpoints \
        outputs_YYYYMMDD_HHMMSS/fold_0/best_model.pth \
        outputs_YYYYMMDD_HHMMSS/fold_1/best_model.pth \
        outputs_YYYYMMDD_HHMMSS/fold_2/best_model.pth \
        outputs_YYYYMMDD_HHMMSS/fold_3/best_model.pth \
        outputs_YYYYMMDD_HHMMSS/fold_4/best_model.pth \
    --output_dir outputs_YYYYMMDD_HHMMSS
```

This creates `submission.csv` in the output directory.

### Inference Features

- **Automatic crop stitching**: When `USE_CROPS=True`, predictions are stitched back to full-width images
- **Overlap averaging**: Overlapping crop regions are averaged for smooth predictions
- **TTA support**: Test-time augmentation with horizontal flip
- **Post-processing**: Thresholding and small component removal

---

## Available Models

### Architectures

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| **UNet** | Standard U-shaped encoder-decoder | Baseline, general purpose |
| **UNet++** | Dense skip connections | Better feature fusion |
| **FPN** | Feature Pyramid Network | Multi-scale features |
| **LinkNet** | Efficient encoder-decoder | Fast inference |
| **PSPNet** | Pyramid Scene Parsing | Context aggregation |
| **DeepLabV3** | Atrous spatial pyramid pooling | Large receptive field |
| **DeepLabV3+** | Enhanced DeepLabV3 | Best accuracy |

### Encoders

| Encoder Family | Examples | Characteristics |
|---------------|----------|----------------|
| **ResNet** | resnet18, resnet34, resnet50 | Balanced, widely used |
| **EfficientNet** | efficientnet-b0 to b7 | Efficient, scalable |
| **SE-ResNeXt** | se_resnext50_32x4d | Attention mechanisms |
| **MobileNet** | mobilenet_v2 | Lightweight, fast |

### Recommended Configurations

**For Best Accuracy:**
```python
MODEL_ARCH = "unetplusplus"
ENCODER = "efficientnet-b3"
```

**For Fast Training:**
```python
MODEL_ARCH = "unet"
ENCODER = "resnet18"
```

**For Memory-Constrained Systems:**
```python
MODEL_ARCH = "linknet"
ENCODER = "mobilenet_v2"
```

---

## Loss Functions

### Available Losses

1. **`bce_with_logits`**: Standard Binary Cross-Entropy
   - Balanced loss for positive/negative examples
   - Good baseline

2. **`dice`**: Dice loss
   - Handles class imbalance well
   - Good for segmentation tasks

3. **`combo`**: Combined BCE + Dice (0.5 each)
   - Balances pixel-wise and region-overlap objectives
   - General purpose

4. **`bce_pos_weight`**: BCE with positive class weighting
   - `pos_weight = (2.0, 2.0, 1.0, 1.5)` for classes 1-4
   - Emphasizes defect detection

5. **`bce_dice_pos_weight`**: Combined 0.75*BCE + 0.25*Dice with pos_weight ⭐
   - Best for imbalanced defect detection
   - Recommended for production

See `LOSS_FUNCTIONS_GUIDE.md` for detailed explanations.

---

## Memory Optimization

This codebase includes several memory optimizations:

### Training Memory

- **OOF predictions**: Only generated from best model (not every epoch)
- **Garbage collection**: Aggressive cleanup after each epoch
- **Best-only checkpoints**: Saves only best model (not all epochs)
- **Pre-cropped mode**: 60% memory reduction vs on-the-fly crops

### Inference Memory

- **Batch-by-batch processing**: Never accumulates all predictions
- **Immediate RLE conversion**: Converts to strings, frees numpy arrays
- **Sequential model loading**: Loads one model at a time (optional)

See `MEMORY_USAGE_ANALYSIS.md` and `TRAINING_MEMORY_ISSUES.md` for details.

---

## Testing and Validation

### Test Environment Setup

```bash
python test_setup.py
```

Verifies:
- Data paths and structure
- Model creation
- Dataset loading
- Loss functions

### Test Crop Mode

```bash
python test_crops.py
```

Verifies crop extraction and stitching work correctly.

### Test Pre-cropped Mode

```bash
python test_precropped.py
```

Verifies pre-cropped data loading.

---

## Customization

### Experiment with Different Models

Edit `config.py`:

```python
MODEL_ARCH = "unetplusplus"      # Try different architecture
ENCODER = "resnet50"             # Try different encoder
```

### Add Custom Augmentations

Edit `dataset.py` in `get_transforms()`:

```python
def get_transforms(phase, mean, std):
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Add your custom augmentations here
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
```

### Try Different Loss Functions

Edit `config.py`:

```python
LOSS = "bce_dice_pos_weight"     # Try different loss
```

---

## Documentation

Comprehensive documentation is available:

- **`SETUP_GUIDE.md`**: Detailed setup and installation
- **`TRAINING_MODES.md`**: Comparison of training modes
- **`PRECROPPED_MODE_GUIDE.md`**: Pre-cropped mode setup and usage
- **`LOSS_FUNCTIONS_GUIDE.md`**: Loss function details
- **`CHECKPOINT_GUIDE.md`**: Checkpoint management
- **`MEMORY_USAGE_ANALYSIS.md`**: Memory optimization details
- **`TRAINING_MEMORY_ISSUES.md`**: Training memory troubleshooting
- **`OOF_MEMORY_FIX_SUMMARY.md`**: OOF prediction memory fixes

---

## Performance Tips

1. **Use pre-cropped mode** for production training (60% memory savings)
2. **Enable crop mode** for better performance (+0.7-1.5% Dice)
3. **Use ensemble** of 5 folds for best results
4. **Tune post-processing** (`THRESHOLD`, `MIN_SIZE`) on validation set
5. **Try different encoders**: EfficientNet-B3, ResNet50, SE-ResNeXt
6. **Use `bce_dice_pos_weight`** loss for imbalanced data

---

## Troubleshooting

### Out of Memory (OOM)

- Enable `USE_PRECROPPED = True`
- Reduce `BATCH_SIZE_TRAIN`
- Set `SAVE_BEST_ONLY = True`
- Reduce `NUM_WORKERS`

### Slow Training

- Use pre-cropped mode (35% faster)
- Reduce `NUM_WORKERS` if I/O bound
- Use smaller encoder (resnet18 vs resnet50)

### Poor Performance

- Increase training epochs
- Try crop mode (more training samples)
- Experiment with different encoders
- Tune post-processing parameters

---

## References

- **Competition**: [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
- **Segmentation Models**: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- **Albumentations**: [albumentations.ai](https://albumentations.ai/)

---

## License

This code is provided for educational and research purposes.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Last Updated**: 2026-01-15
