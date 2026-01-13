"""
Dataset and data loading utilities for Steel Defect Detection
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
)
from albumentations.pytorch import ToTensorV2


def mask2rle(img):
    """
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, shape=(256, 1600)):
    """
    rle: run-length encoded string
    shape: (height, width) of the mask
    Returns numpy array, 1 -> mask, 0 -> background
    """
    if pd.isna(rle) or rle == '' or rle == ' ':
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape, order='F')


def make_mask(row_id, df):
    """
    Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`
    """
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)
    
    for idx, label in enumerate(labels.values):
        if label is not np.nan and label != '' and label != ' ':
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    
    return fname, masks


def prepare_trainval_dataframe(csv_path, n_folds=5, seed=69, train_images_dir=None):
    """
    Prepare train dataframe with fold information for cross-validation
    
    Args:
        csv_path: path to train.csv
        n_folds: number of cross-validation folds
        seed: random seed for reproducibility
        train_images_dir: path to train_images directory (to include all images)
    """
    df = pd.read_csv(csv_path)
    
    # Handle different CSV formats
    if 'ImageId_ClassId' in df.columns:
        # Format 1: ImageId_ClassId combined (Dropbox format)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
    elif 'ImageId' in df.columns and 'ClassId' in df.columns:
        # Format 2: ImageId and ClassId already separated (External SSD format)
        # This format only contains rows for images WITH defects!
        df['ClassId'] = df['ClassId'].astype(int)
    else:
        raise ValueError(f"Unexpected CSV format. Columns: {df.columns.tolist()}")
    
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    
    # IMPORTANT: Include ALL images from train_images directory
    # The External SSD CSV format only contains images WITH defects,
    # so we need to add images WITHOUT defects (empty masks)
    if train_images_dir is not None:
        import os
        all_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
        
        # Find images not in the CSV (i.e., no defects)
        existing_images = set(df.index)
        missing_images = [img for img in all_images if img not in existing_images]
        
        if len(missing_images) > 0:
            print(f"Found {len(missing_images)} images without defects (not in CSV)")
            
            # Create empty rows for missing images
            missing_df = pd.DataFrame(index=missing_images, columns=df.columns)
            missing_df.index.name = 'ImageId'
            
            # Concatenate
            df = pd.concat([df, missing_df])
            print(f"Total images after adding defect-free images: {len(df)}")
    
    # Count number of defects per image for stratification
    df['defects'] = df.count(axis=1)
    
    # Create folds using StratifiedKFold
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['defects'])):
        df.iloc[val_idx, df.columns.get_loc('fold')] = fold
    
    return df


class SteelDataset(Dataset):
    """
    Dataset for Steel Defect Detection
    """
    def __init__(self, df, data_folder, mean, std, phase='train'):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()
    
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # ToTensorV2 returns (H, W, C) -> need to permute to (C, H, W)
        
        # Convert mask to tensor and permute to (C, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        
        # Permute from (H, W, C) to (C, H, W)
        if mask.dim() == 3:
            mask = mask.permute(2, 0, 1)  # (4, 256, 1600)
        
        return img, mask, image_id
    
    def __len__(self):
        return len(self.fnames)


class TestDataset(Dataset):
    """
    Dataset for test prediction
    """
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose([
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ])
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        
        images = self.transform(image=image)["image"]
        return fname, images
    
    def __len__(self):
        return self.num_samples


def get_transforms(phase, mean, std):
    """
    Get augmentation transforms for training/validation
    """
    list_transforms = []
    
    if phase == "train":
        list_transforms.extend([
            HorizontalFlip(p=0.5),
        ])
    
    list_transforms.extend([
        Normalize(mean=mean, std=std, p=1),
        ToTensorV2(),
    ])
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_train_val_loaders(df, fold, data_folder, mean, std, batch_size_train, 
                          batch_size_val, num_workers):
    """
    Get train and validation data loaders for a specific fold
    """
    train_df = df[df['fold'] != fold].copy()
    val_df = df[df['fold'] == fold].copy()
    
    print(f"Fold {fold}: Train samples = {len(train_df)}, Val samples = {len(val_df)}")
    
    train_dataset = SteelDataset(train_df, data_folder, mean, std, phase='train')
    val_dataset = SteelDataset(val_df, data_folder, mean, std, phase='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    
    return train_loader, val_loader, val_df


def get_test_loader(test_folder, sample_submission_path, mean, std, 
                   batch_size, num_workers):
    """
    Get test data loader
    """
    df = pd.read_csv(sample_submission_path)
    test_dataset = TestDataset(test_folder, df, mean, std)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

