"""
Inference script for Steel Defect Detection
Generates predictions on test set
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
from dataset import get_test_loader, mask2rle
from model import get_model
from losses import post_process

warnings.filterwarnings("ignore")


class Inferencer:
    """
    Inferencer class for generating predictions
    """
    def __init__(self, checkpoint_paths, output_dir):
        self.checkpoint_paths = checkpoint_paths
        self.output_dir = output_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        print(f"Loading {len(checkpoint_paths)} model(s)")
        
        # Load models
        self.models = []
        for i, ckpt_path in enumerate(checkpoint_paths):
            print(f"  Loading model {i+1}: {ckpt_path}")
            model = get_model(
                arch=config.MODEL_ARCH,
                encoder=config.ENCODER,
                encoder_weights=None,
                num_classes=config.NUM_CLASSES,
                activation=None
            )
            
            # Load checkpoint
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.models.append(model)
        
        # Get test loader
        self.test_loader = get_test_loader(
            config.TEST_IMAGES_DIR,
            config.SAMPLE_SUBMISSION,
            config.MEAN,
            config.STD,
            config.BATCH_SIZE_VAL,
            config.NUM_WORKERS
        )
        
        cudnn.benchmark = True
    
    def predict_single_model(self, model, images):
        """
        Predict using a single model
        """
        with torch.no_grad():
            images = images.to(self.device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            return probs.cpu().numpy()
    
    def predict_batch(self, images):
        """
        Predict a batch using all models (ensemble)
        """
        predictions = []
        
        # Predict with each model
        for model in self.models:
            preds = self.predict_single_model(model, images)
            predictions.append(preds)
        
        # Average predictions
        predictions = np.mean(predictions, axis=0)
        
        # TTA: Horizontal flip
        if config.TTA and config.TTA_HORIZONTAL_FLIP:
            images_flipped = torch.flip(images, dims=[3])  # flip width
            predictions_flipped = []
            
            for model in self.models:
                preds = self.predict_single_model(model, images_flipped)
                predictions_flipped.append(preds)
            
            predictions_flipped = np.mean(predictions_flipped, axis=0)
            predictions_flipped = np.flip(predictions_flipped, axis=3)  # flip back
            
            # Average original and flipped
            predictions = (predictions + predictions_flipped) / 2.0
        
        return predictions
    
    def generate_submission(self):
        """
        Generate submission file
        """
        print("\nGenerating predictions...")
        
        predictions_list = []
        
        for batch_idx, (fnames, images) in enumerate(tqdm(self.test_loader)):
            batch_preds = self.predict_batch(images)
            
            # Post-process and encode predictions
            for fname, preds in zip(fnames, batch_preds):
                for cls in range(config.NUM_CLASSES):
                    pred = preds[cls]
                    
                    # Post-process
                    pred, num_components = post_process(
                        pred, 
                        config.THRESHOLD, 
                        config.MIN_SIZE
                    )
                    
                    # Encode to RLE
                    rle = mask2rle(pred)
                    
                    # Add to predictions
                    name = fname + f"_{cls+1}"
                    predictions_list.append([name, rle])
        
        # Create submission dataframe
        submission_df = pd.DataFrame(predictions_list, 
                                     columns=['ImageId_ClassId', 'EncodedPixels'])
        
        # Save submission
        submission_path = os.path.join(self.output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        print(f"\nSubmission saved to: {submission_path}")
        print(f"Submission shape: {submission_df.shape}")
        print("\nFirst few rows:")
        print(submission_df.head(10))
        
        return submission_df


def main():
    parser = argparse.ArgumentParser(description='Generate test predictions')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Path(s) to model checkpoint(s)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for submission file')
    args = parser.parse_args()
    
    # Output directory
    if args.output_dir is None:
        output_dir = config.OUTPUT_DIR
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create inferencer
    inferencer = Inferencer(args.checkpoints, output_dir)
    
    # Generate submission
    submission_df = inferencer.generate_submission()
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()




