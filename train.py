"""
Training script for Steel Defect Detection with 5-fold CV and OOF generation
"""
import os
import sys
import time
import random
import argparse
import warnings
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import config
from dataset import prepare_trainval_dataframe, get_train_val_loaders
from model import get_model
from losses import MetricTracker, BCEDiceLoss

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """
    Logger to write both to console and file
    """
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
    
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()


class Trainer:
    """
    Trainer class for training and validation
    """
    def __init__(self, fold, output_dir, log_file, checkpoint_path=None):
        self.fold = fold
        self.output_dir = output_dir
        self.log_file = log_file
        self.checkpoint_path = checkpoint_path
        
        # Create fold directory
        self.fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(self.fold_dir, exist_ok=True)
        
        # Training parameters
        self.num_epochs = config.NUM_EPOCHS
        self.batch_size_train = config.BATCH_SIZE_TRAIN
        self.batch_size_val = config.BATCH_SIZE_VAL
        self.accumulation_steps = config.ACCUMULATION_STEPS
        self.lr = config.LEARNING_RATE
        self.num_workers = config.NUM_WORKERS
        
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model
        self.model = get_model(
            arch=config.MODEL_ARCH,
            encoder=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            num_classes=config.NUM_CLASSES,
            activation=config.ACTIVATION
        )
        self.model = self.model.to(self.device)
        
        # Loss function
        if config.LOSS == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config.LOSS == 'dice':
            from losses import DiceLoss
            self.criterion = DiceLoss()
        elif config.LOSS == 'combo':
            self.criterion = BCEDiceLoss()
        elif config.LOSS == 'bce_pos_weight':
            from losses import BCEWithPosWeightLoss
            self.criterion = BCEWithPosWeightLoss(pos_weight=(2.0, 2.0, 1.0, 1.5))
        elif config.LOSS == 'bce_dice_pos_weight':
            from losses import BCEDiceWithPosWeightLoss
            self.criterion = BCEDiceWithPosWeightLoss(pos_weight=(2.0, 2.0, 1.0, 1.5), 
                                                      bce_weight=0.75, dice_weight=0.25)
        else:
            raise ValueError(f"Unknown loss: {config.LOSS}")
        
        # Optimizer
        if config.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, 
                                       weight_decay=config.WEIGHT_DECAY)
        elif config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr,
                                        weight_decay=config.WEIGHT_DECAY)
        elif config.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                      momentum=0.9, weight_decay=config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        # Scheduler
        if config.SCHEDULER == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", 
                patience=config.SCHEDULER_PATIENCE,
                factor=config.SCHEDULER_FACTOR,
                min_lr=config.SCHEDULER_MIN_LR
            )
        elif config.SCHEDULER == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs,
                eta_min=config.SCHEDULER_MIN_LR
            )
        else:
            self.scheduler = None
        
        # Data loaders
        df = prepare_trainval_dataframe(config.TRAIN_CSV, n_folds=config.NUM_FOLDS, 
                                       seed=config.SEED, train_images_dir=config.TRAIN_IMAGES_DIR)
        self.train_loader, self.val_loader, self.val_df = get_train_val_loaders(
            df, fold, config.TRAIN_IMAGES_DIR, config.MEAN, config.STD,
            self.batch_size_train, self.batch_size_val, self.num_workers,
            use_crops=config.USE_CROPS,
            crop_height=config.CROP_HEIGHT,
            crop_width=config.CROP_WIDTH,
            crop_stride=config.CROP_STRIDE,
            use_precropped=config.USE_PRECROPPED,
            precropped_dir=config.PRECROPPED_DIR
        )
        
        # Store val_df for OOF predictions
        self.val_df = self.val_df
        
        # Best loss tracking
        self.best_loss = float('inf')
        self.best_dice = 0.0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': [],
        }
        
        # OOF predictions
        self.oof_predictions = {}
        
        # Starting epoch (for resuming training)
        self.start_epoch = 0
        
        # Load checkpoint if provided
        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)
        
        cudnn.benchmark = True
    
    def forward(self, images, masks):
        """
        Forward pass
        """
        images = images.to(self.device)
        masks = masks.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs
    
    def train_one_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        metric_tracker = MetricTracker('train', threshold=config.THRESHOLD, 
                                      num_classes=config.NUM_CLASSES)
        
        running_loss = 0.0
        self.optimizer.zero_grad()
        
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Fold {self.fold} | Epoch {epoch+1}/{self.num_epochs} | Phase: TRAIN")
        print(f"{'='*80}")
        
        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            loss, outputs = self.forward(images, masks)
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.accumulation_steps
            
            # Update metrics
            outputs = outputs.detach().cpu()
            masks = masks.cpu()
            metric_tracker.update(outputs, masks)
            
            # Log progress
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] | Loss: {avg_loss:.4f}")
        
        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        metrics = metric_tracker.get_metrics()
        
        elapsed = time.time() - start_time
        print(f"\nTrain Results:")
        print(f"  Loss: {epoch_loss:.4f} | Dice: {metrics['dice']:.4f} | "
              f"Dice_pos: {metrics['dice_pos']:.4f} | Dice_neg: {metrics['dice_neg']:.4f} | "
              f"IoU: {metrics['iou']:.4f}")
        print(f"  Per-Class Metrics:")
        for cls in range(config.NUM_CLASSES):
            print(f"    Class {cls+1}: Dice={metrics[f'class_{cls+1}_dice']:.4f}, "
                  f"IoU={metrics[f'class_{cls+1}_iou']:.4f}, "
                  f"Prec={metrics[f'class_{cls+1}_precision']:.4f}, "
                  f"Recall={metrics[f'class_{cls+1}_recall']:.4f}, "
                  f"ClsAcc={metrics[f'class_{cls+1}_cls_accuracy']:.4f}, "
                  f"ClsF1={metrics[f'class_{cls+1}_cls_f1']:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_dice'].append(metrics['dice'])
        self.history['train_iou'].append(metrics['iou'])
        
        return epoch_loss, metrics
    
    def validate_one_epoch(self, epoch, store_oof=False):
        """
        Validate for one epoch
        
        Args:
            epoch: Current epoch number
            store_oof: Whether to store OOF predictions (only when generating final OOF)
        """
        self.model.eval()
        metric_tracker = MetricTracker('val', threshold=config.THRESHOLD,
                                      num_classes=config.NUM_CLASSES)
        
        running_loss = 0.0
        oof_preds_epoch = {}
        
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Fold {self.fold} | Epoch {epoch+1}/{self.num_epochs} | Phase: VALIDATION")
        print(f"{'='*80}")
        if store_oof:
            print(f"  NOTE: Generating OOF predictions from best model")
        
        with torch.no_grad():
            for batch_idx, (images, masks, image_ids) in enumerate(self.val_loader):
                loss, outputs = self.forward(images, masks)
                running_loss += loss.item()
                
                # Update metrics
                outputs_cpu = outputs.detach().cpu()
                masks_cpu = masks.cpu()
                metric_tracker.update(outputs_cpu, masks_cpu)
                
                # Store OOF predictions only when requested
                if store_oof:
                    probs = torch.sigmoid(outputs_cpu).numpy()
                    for i, image_id in enumerate(image_ids):
                        # Store as binary masks (uint8) or probabilities (float32)
                        if config.OOF_STORE_BINARY:
                            # Binary masks: 4x smaller memory (0.52 MB vs 2.1 MB per crop)
                            oof_preds_epoch[image_id] = (probs[i] > config.THRESHOLD).astype(np.uint8)
                        else:
                            # Full probabilities: keeps all prediction information
                            oof_preds_epoch[image_id] = probs[i]
                
                # Log progress
                if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    print(f"  Batch [{batch_idx+1}/{len(self.val_loader)}] | Loss: {avg_loss:.4f}")
        
        # Epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        metrics = metric_tracker.get_metrics()
        
        elapsed = time.time() - start_time
        print(f"\nValidation Results:")
        print(f"  Loss: {epoch_loss:.4f} | Dice: {metrics['dice']:.4f} | "
              f"Dice_pos: {metrics['dice_pos']:.4f} | Dice_neg: {metrics['dice_neg']:.4f} | "
              f"IoU: {metrics['iou']:.4f}")
        print(f"  Per-Class Metrics:")
        for cls in range(config.NUM_CLASSES):
            print(f"    Class {cls+1}: Dice={metrics[f'class_{cls+1}_dice']:.4f}, "
                  f"IoU={metrics[f'class_{cls+1}_iou']:.4f}, "
                  f"Prec={metrics[f'class_{cls+1}_precision']:.4f}, "
                  f"Recall={metrics[f'class_{cls+1}_recall']:.4f}, "
                  f"ClsAcc={metrics[f'class_{cls+1}_cls_accuracy']:.4f}, "
                  f"ClsF1={metrics[f'class_{cls+1}_cls_f1']:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        
        self.history['val_loss'].append(epoch_loss)
        self.history['val_dice'].append(metrics['dice'])
        self.history['val_iou'].append(metrics['iou'])
        
        # Return OOF predictions if generated
        if store_oof:
            print(f"  OOF predictions generated: {len(oof_preds_epoch)} samples")
            return epoch_loss, metrics, oof_preds_epoch
        
        return epoch_loss, metrics
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint for resuming training
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*80}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*80}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Model state loaded")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  ✓ Optimizer state loaded")
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_dice = checkpoint.get('best_dice', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"  ✓ Resuming from epoch {self.start_epoch}")
        print(f"  ✓ Best Dice: {self.best_dice:.4f}")
        print(f"  ✓ Best Loss: {self.best_loss:.4f}")
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_dice': self.best_dice,
            'history': self.history,
        }
        
        # Save latest checkpoint
        if not config.SAVE_BEST_ONLY:
            checkpoint_path = os.path.join(self.fold_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(state, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.fold_dir, 'best_model.pth')
            torch.save(state, best_path)
            print(f"  *** New best model saved: {best_path} ***")
    
    def generate_oof_predictions(self):
        """
        Generate OOF predictions from the best model
        This is called after training completes to ensure OOF is from the best epoch
        """
        print(f"\n{'='*80}")
        print(f"Generating OOF predictions from best model")
        print(f"{'='*80}")
        
        # Load best model
        best_model_path = os.path.join(self.fold_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            print(f"  WARNING: Best model not found at {best_model_path}")
            print(f"  Skipping OOF generation")
            return
        
        print(f"  Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        best_dice = checkpoint.get('best_dice', 0.0)
        best_epoch = checkpoint.get('epoch', -1)
        print(f"  Best model: Epoch {best_epoch + 1}, Dice: {best_dice:.4f}")
        
        # Run validation with OOF storage enabled
        self.model.eval()
        oof_preds = {}
        
        with torch.no_grad():
            for batch_idx, (images, masks, image_ids) in enumerate(self.val_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs_cpu = outputs.detach().cpu()
                
                probs = torch.sigmoid(outputs_cpu).numpy()
                for i, image_id in enumerate(image_ids):
                    # Store as binary masks (uint8) or probabilities (float32)
                    if config.OOF_STORE_BINARY:
                        oof_preds[image_id] = (probs[i] > config.THRESHOLD).astype(np.uint8)
                    else:
                        oof_preds[image_id] = probs[i]
                
                if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                    print(f"  Batch [{batch_idx+1}/{len(self.val_loader)}]")
        
        self.oof_predictions = oof_preds
        print(f"  OOF predictions generated: {len(oof_preds)} samples")
        print(f"{'='*80}\n")
    
    def save_oof_predictions(self):
        """
        Save out-of-fold predictions
        """
        oof_path = os.path.join(self.fold_dir, 'oof_predictions.npy')
        np.save(oof_path, self.oof_predictions)
        print(f"  Saved OOF predictions: {oof_path}")
    
    def save_history(self):
        """
        Save training history
        """
        history_path = os.path.join(self.fold_dir, 'history.csv')
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(history_path, index=False)
        print(f"  Saved training history: {history_path}")
    
    def train(self):
        """
        Main training loop
        """
        print(f"\n{'#'*80}")
        print(f"# Starting training for Fold {self.fold}")
        print(f"# Output directory: {self.fold_dir}")
        if self.start_epoch > 0:
            print(f"# Resuming from epoch {self.start_epoch + 1}")
        print(f"{'#'*80}\n")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_loss, train_metrics = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_one_epoch(epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                if config.SCHEDULER == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check if best model
            is_best = False
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_loss = val_loss
                is_best = True
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            print(f"\n  Current LR: {current_lr:.2e} | Best Dice: {self.best_dice:.4f} | Best Loss: {self.best_loss:.4f}")
            
            # Clear GPU cache and force garbage collection to free memory
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Generate OOF predictions from the best model
        self.generate_oof_predictions()
        
        # Save final outputs
        self.save_oof_predictions()
        self.save_history()
        
        print(f"\n{'#'*80}")
        print(f"# Finished training for Fold {self.fold}")
        print(f"# Best Dice: {self.best_dice:.4f} | Best Loss: {self.best_loss:.4f}")
        print(f"{'#'*80}\n")
        
        return self.best_dice, self.best_loss


def main():
    parser = argparse.ArgumentParser(description='Train Steel Defect Detection model')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train')
    parser.add_argument('--all_folds', action='store_true', help='Train all folds')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint file to resume training from')
    args = parser.parse_args()
    
    # Set seed
    set_seed(config.SEED)
    
    # Create output directory
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy config.py to output directory for reproducibility
    config_source = 'config.py'
    config_dest = os.path.join(output_dir, 'config.py')
    shutil.copy2(config_source, config_dest)
    print(f"Configuration saved to: {config_dest}\n")
    
    # Train folds
    if args.all_folds:
        folds_to_train = config.TRAIN_FOLDS
    else:
        folds_to_train = [args.fold]
    
    # Check if checkpoint is provided and we're training a single fold
    if args.checkpoint and not args.all_folds:
        print(f"\nNote: Checkpoint will be loaded for fold {args.fold}")
    elif args.checkpoint and args.all_folds:
        print("\nWarning: Checkpoint loading is only supported for single fold training.")
        print("         The --checkpoint flag will be ignored when using --all_folds.")
        args.checkpoint = None
    
    results = {}
    for fold in folds_to_train:
        # Setup logging for this fold
        log_file = os.path.join(output_dir, f'training_log_fold_{fold}.txt')
        sys.stdout = Logger(log_file)
        
        # Print configuration
        print("="*80)
        print("CONFIGURATION")
        print("="*80)
        print(f"Data root: {config.DATA_ROOT}")
        print(f"Output dir: {output_dir}")
        print(f"Model: {config.MODEL_ARCH} with {config.ENCODER} encoder")
        print(f"Num classes: {config.NUM_CLASSES}")
        if config.USE_PRECROPPED:
            print(f"Image mode: PRE-CROPPED")
            print(f"  Pre-cropped dir: {config.PRECROPPED_DIR}")
        elif config.USE_CROPS:
            print(f"Image mode: CROPS")
            print(f"  Crop size: {config.CROP_HEIGHT}x{config.CROP_WIDTH}")
            print(f"  Crop stride: {config.CROP_STRIDE}")
        else:
            print(f"Image mode: FULL IMAGES")
            print(f"  Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
        print(f"Num folds: {config.NUM_FOLDS}")
        print(f"Num epochs: {config.NUM_EPOCHS}")
        print(f"Batch size: {config.BATCH_SIZE_TRAIN} (effective: {config.BATCH_SIZE_TRAIN * config.ACCUMULATION_STEPS})")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Optimizer: {config.OPTIMIZER}")
        print(f"Loss: {config.LOSS}")
        print(f"Scheduler: {config.SCHEDULER}")
        print(f"Seed: {config.SEED}")
        if args.checkpoint:
            print(f"Checkpoint: {args.checkpoint}")
        print("="*80)
        print()
        
        # Only use checkpoint for the specified fold when not training all folds
        checkpoint_to_use = args.checkpoint if fold == args.fold else None
        trainer = Trainer(fold, output_dir, log_file, checkpoint_path=checkpoint_to_use)
        best_dice, best_loss = trainer.train()
        results[fold] = {'best_dice': best_dice, 'best_loss': best_loss}
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for fold, metrics in results.items():
        print(f"Fold {fold}: Best Dice = {metrics['best_dice']:.4f}, Best Loss = {metrics['best_loss']:.4f}")
    
    if len(results) > 1:
        avg_dice = np.mean([m['best_dice'] for m in results.values()])
        avg_loss = np.mean([m['best_loss'] for m in results.values()])
        print(f"\nAverage: Dice = {avg_dice:.4f}, Loss = {avg_loss:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()

