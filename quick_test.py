"""
Quick test script to verify training pipeline works
Runs 1 epoch on a small subset of data
"""
import os
import sys
import warnings
import random
import numpy as np
import torch

# Modify config for quick test
import config
config.NUM_EPOCHS = 2
config.BATCH_SIZE_TRAIN = 2
config.BATCH_SIZE_VAL = 2
config.ACCUMULATION_STEPS = 2
config.NUM_WORKERS = 2
config.LOG_INTERVAL = 10
config.OUTPUT_DIR = "outputs_quick_test"

from train import Trainer, set_seed, Logger

warnings.filterwarnings("ignore")

def main():
    print("="*80)
    print("QUICK TEST OF TRAINING PIPELINE")
    print("="*80)
    print(f"Running {config.NUM_EPOCHS} epochs on fold 0")
    print(f"Batch size: {config.BATCH_SIZE_TRAIN}")
    print(f"Output dir: {config.OUTPUT_DIR}")
    print("="*80)
    print()
    
    # Set seed
    set_seed(config.SEED)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config.OUTPUT_DIR, 'test_log.txt')
    sys.stdout = Logger(log_file)
    
    # Train fold 0
    fold = 0
    trainer = Trainer(fold, config.OUTPUT_DIR, log_file)
    
    try:
        best_dice, best_loss = trainer.train()
        
        print("\n" + "="*80)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best Dice: {best_dice:.4f}")
        print(f"Best Loss: {best_loss:.4f}")
        print("\nThe training pipeline is working correctly.")
        print("You can now run full training with:")
        print("  bash train_single_fold.sh 0")
        print("  or")
        print("  bash train_all_folds.sh")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Training failed!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)




