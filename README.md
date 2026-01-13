# UNet Baseline for Steel Defect Detection

This is a baseline implementation for the Severstal Steel Defect Detection challenge using UNet with various encoder backbones.

## Features

- **5-fold cross-validation** with stratified splitting based on number of defects
- **Out-of-fold (OOF) predictions** for ensemble modeling
- **Per-class metrics** including:
  - Segmentation: Dice, IoU, Precision, Recall
  - Classification: Accuracy, F1 score (whether defect class is present)
- **Comprehensive logging** with timestamped output directories
- **Flexible configuration** via `config.py`
- **Model checkpointing** with best model saving
- **Test-time augmentation (TTA)** support
- **Model ensemble** support for inference

## Requirements

Install required packages:

```bash
pip install torch torchvision opencv-python pandas numpy scikit-learn albumentations segmentation-models-pytorch tqdm
```

Or use the conda environment:

```bash
conda activate compute_gpu_3_10
```

## Data Structure

The data should be organized as follows:

```
/media/frank/ext_ssd2/kaggle_steel/
├── train_images/
│   ├── 0002cc93b.jpg
│   ├── 000361c98.jpg
│   └── ...
├── test_images/
│   ├── 00a3ab3ee.jpg
│   └── ...
├── train.csv
└── sample_submission.csv
```

## Configuration

Edit `config.py` to change parameters:

- **Model**: Architecture (unet, unetplusplus, fpn), encoder (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
- **Training**: Epochs, batch size, learning rate, optimizer, scheduler, loss function
- **Data**: Image size, augmentations, normalization
- **Inference**: Threshold, minimum component size, TTA

## Training

### Train a single fold:

```bash
bash train_single_fold.sh 0
```

### Resume training from a checkpoint:

```bash
# Using shell script
bash train_single_fold.sh 0 outputs_20260110_220612/fold_0/checkpoint_epoch_10.pth

# Or using Python directly
python train.py --fold 0 --checkpoint outputs_20260110_220612/fold_0/checkpoint_epoch_10.pth
```

This will:
- Load the model weights, optimizer state, and training history
- Resume training from the next epoch
- Preserve the best Dice score and loss from the previous run

### Train all folds at once:

```bash
bash train_all_folds.sh
```

### Train all folds sequentially:

```bash
bash train_sequential.sh
```

### Train with Python directly:

```bash
python train.py --fold 0
python train.py --all_folds
```

## Output Structure

Each training run creates a timestamped output directory:

```
outputs_YYYYMMDD_HHMMSS/
├── training_log.txt              # Complete training log
├── fold_0/
│   ├── best_model.pth            # Best model checkpoint
│   ├── checkpoint_epoch_N.pth    # Epoch checkpoints (if SAVE_BEST_ONLY=False)
│   ├── oof_predictions.npy       # Out-of-fold predictions
│   └── history.csv               # Training history (loss, metrics per epoch)
├── fold_1/
│   └── ...
└── ...
```

## Inference

Generate test predictions using trained models:

### Single model:

```bash
bash inference.sh outputs_YYYYMMDD_HHMMSS outputs_YYYYMMDD_HHMMSS/fold_0/best_model.pth
```

### Ensemble (average predictions from multiple folds):

```bash
bash inference.sh outputs_YYYYMMDD_HHMMSS \
    outputs_YYYYMMDD_HHMMSS/fold_0/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_1/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_2/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_3/best_model.pth \
    outputs_YYYYMMDD_HHMMSS/fold_4/best_model.pth
```

This will create `submission.csv` in the output directory.

## Metrics Tracked

During training and validation, the following metrics are computed:

### Overall Metrics:
- Loss (BCE with logits)
- Dice score (overall, positive samples, negative samples)
- IoU score

### Per-Class Metrics (for each of the 4 defect classes):
- **Segmentation metrics**:
  - Dice coefficient
  - IoU (Intersection over Union)
  - Precision
  - Recall

- **Classification metrics** (whether the defect class is present):
  - Accuracy
  - F1 score

All metrics are logged to console and saved to `training_log.txt`.

## Notes

- The code handles 4 defect classes as specified in the competition
- Images are 256x1600 pixels
- Data augmentation includes horizontal flip (vertical flip is not used as defects have spatial orientation)
- The validation set is stratified by number of defects per image
- OOF predictions are saved for each fold and can be used for stacking/blending ensembles
- GPU memory is cleared after each epoch to prevent OOM errors

## Customization

To experiment with different models or settings:

1. Edit `config.py`:
   - Change `ENCODER` to try different backbones (resnet34, resnet50, efficientnet-b0, etc.)
   - Change `MODEL_ARCH` to try different architectures (unetplusplus, fpn, linknet, etc.)
   - Adjust `NUM_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE_TRAIN`, etc.

2. Add custom augmentations in `dataset.py` in the `get_transforms()` function

3. Try different loss functions by modifying the `LOSS` parameter in `config.py`

## References

Based on the following Kaggle kernels:
- UNet starter kernel by Rishabh Mishra
- Segmentation models pytorch by Pavel Yakubovskiy

## License

This code is provided for educational and research purposes.



# kaggle_severstal
