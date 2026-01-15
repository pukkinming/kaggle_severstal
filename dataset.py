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
    HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
    ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
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
    Dataset for Steel Defect Detection - Full Images
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


class SteelCropsDataset(Dataset):
    """
    Dataset for Steel Defect Detection - Crops
    Generates multiple crops from each steel image to increase training samples
    """
    def __init__(self, df, data_folder, mean, std, phase='train', 
                 crop_height=256, crop_width=512, stride=256):
        """
        Args:
            df: DataFrame with image annotations
            data_folder: Path to image folder
            mean: Normalization mean
            std: Normalization std
            phase: 'train' or 'val'
            crop_height: Height of crops (256)
            crop_width: Width of crops (512)
            stride: Horizontal stride for crops
                    stride=256: 50% overlap, 5 crops per image
                    stride=512: no overlap, 3 crops per image
        """
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.stride = stride
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()
        
        # Calculate number of crops per image
        self.image_width = 1600  # Full image width
        self.crops_per_image = self._calculate_crops_per_image()
        
        # Total samples = num_images × crops_per_image
        self.total_samples = len(self.fnames) * self.crops_per_image
        
        print(f"  {phase.upper()} - Crops per image: {self.crops_per_image}, "
              f"Total samples: {self.total_samples} (from {len(self.fnames)} images)")
    
    def _calculate_crops_per_image(self):
        """
        Calculate number of crops per image based on stride
        For 1600 width with 512 crop and stride 256:
        - Crop 0: [0:512]
        - Crop 1: [256:768]
        - Crop 2: [512:1024]
        - Crop 3: [768:1280]
        - Crop 4: [1088:1600] (adjusted to cover the end)
        """
        # Start with crops at regular stride intervals
        num_crops = (self.image_width - self.crop_width) // self.stride + 1
        
        # Check if we need an additional crop to cover the remaining pixels
        last_crop_end = (num_crops - 1) * self.stride + self.crop_width
        if last_crop_end < self.image_width:
            num_crops += 1  # Add one more crop to cover the end
        
        return num_crops
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Determine which image and which crop
        image_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image
        
        # Get full image and mask
        image_id, full_mask = make_mask(image_idx, self.df)
        image_path = os.path.join(self.root, image_id)
        full_image = cv2.imread(image_path)
        
        if full_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Calculate crop coordinates
        x_start = crop_idx * self.stride
        x_end = min(x_start + self.crop_width, self.image_width)
        
        # If crop extends beyond image, shift it back to fit
        if x_end - x_start < self.crop_width:
            x_end = self.image_width
            x_start = self.image_width - self.crop_width
        
        # Extract crop from full image and mask
        crop_image = full_image[:, x_start:x_end, :]  # (256, 512, 3)
        crop_mask = full_mask[:, x_start:x_end, :]    # (256, 512, 4)
        
        # Apply augmentations
        augmented = self.transforms(image=crop_image, mask=crop_mask)
        crop_image = augmented['image']
        crop_mask = augmented['mask']
        
        # Convert mask to tensor and permute to (C, H, W)
        if not isinstance(crop_mask, torch.Tensor):
            crop_mask = torch.from_numpy(crop_mask)
        
        if crop_mask.dim() == 3:
            crop_mask = crop_mask.permute(2, 0, 1)  # (4, 256, 512)
        
        return crop_image, crop_mask, image_id


class SteelPrecroppedDataset(Dataset):
    """
    Dataset for Steel Defect Detection - Pre-cropped Images
    
    Loads images that have been pre-cropped to disk using precrop_images.py
    This eliminates the need to load full images and extract crops on-the-fly,
    reducing memory usage and improving data loading speed.
    """
    def __init__(self, crop_df, precropped_dir, mean, std, phase='train'):
        """
        Args:
            crop_df: DataFrame with crop information (from crop_mapping.csv)
            precropped_dir: Path to directory containing pre-cropped images
            mean: Normalization mean
            std: Normalization std
            phase: 'train' or 'val'
        """
        self.crop_df = crop_df.reset_index(drop=True)
        self.precropped_dir = precropped_dir
        self.images_dir = os.path.join(precropped_dir, 'images')
        self.masks_dir = os.path.join(precropped_dir, 'masks')
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        
        print(f"  {phase.upper()} - Pre-cropped samples: {len(self.crop_df)}")
    
    def __len__(self):
        return len(self.crop_df)
    
    def __getitem__(self, idx):
        # Get crop information
        crop_info = self.crop_df.iloc[idx]
        crop_id = crop_info['crop_id']
        original_image_id = crop_info['original_image_id']
        
        # Load pre-cropped image
        crop_image_path = os.path.join(self.images_dir, f"{crop_id}.jpg")
        crop_image = cv2.imread(crop_image_path)
        
        if crop_image is None:
            raise ValueError(f"Could not load crop image: {crop_image_path}")
        
        # Load pre-cropped mask
        crop_mask_path = os.path.join(self.masks_dir, f"{crop_id}.npy")
        crop_mask = np.load(crop_mask_path)
        
        # Apply augmentations
        augmented = self.transforms(image=crop_image, mask=crop_mask)
        crop_image = augmented['image']
        crop_mask = augmented['mask']
        
        # Convert mask to tensor and permute to (C, H, W)
        if not isinstance(crop_mask, torch.Tensor):
            crop_mask = torch.from_numpy(crop_mask)
        
        if crop_mask.dim() == 3:
            crop_mask = crop_mask.permute(2, 0, 1)  # (4, 256, 512)
        
        return crop_image, crop_mask, original_image_id


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
    # Import config to access augmentation parameters
    import config
    
    list_transforms = []
    
    if phase == "train":
        # Add augmentations for training
        list_transforms.extend([
            HorizontalFlip(p=config.AUG_HORIZONTAL_FLIP_PROB),
            VerticalFlip(p=config.AUG_VERTICAL_FLIP_PROB),
            RandomBrightnessContrast(
                brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
                contrast_limit=config.AUG_CONTRAST_LIMIT,
                p=config.AUG_RANDOM_BRIGHTNESS_CONTRAST_PROB
            ),
        ])
    
    # Normalization and conversion (applied to both train and val)
    list_transforms.extend([
        Normalize(mean=mean, std=std, p=1),
        ToTensorV2(),
    ])
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_train_val_loaders(df, fold, data_folder, mean, std, batch_size_train, 
                          batch_size_val, num_workers, use_crops=False,
                          crop_height=256, crop_width=512, crop_stride=256,
                          use_precropped=False, precropped_dir=None):
    """
    Get train and validation data loaders for a specific fold
    
    Args:
        df: DataFrame with fold information
        fold: Fold number for validation
        data_folder: Path to image folder
        mean: Normalization mean
        std: Normalization std
        batch_size_train: Training batch size
        batch_size_val: Validation batch size
        num_workers: Number of workers for data loading
        use_crops: If True, use crop-based dataset; if False, use full images
        crop_height: Height of crops (only used if use_crops=True)
        crop_width: Width of crops (only used if use_crops=True)
        crop_stride: Stride for crops (only used if use_crops=True)
        use_precropped: If True, load pre-cropped images from disk
        precropped_dir: Path to pre-cropped images directory
    
    Returns:
        train_loader, val_loader, val_df
    """
    train_df = df[df['fold'] != fold].copy()
    val_df = df[df['fold'] == fold].copy()
    
    print(f"\nFold {fold}: Train images = {len(train_df)}, Val images = {len(val_df)}")
    
    if use_precropped:
        # Load pre-cropped images from disk
        print(f"Using PRE-CROPPED mode from: {precropped_dir}")
        
        # Load crop mapping CSV
        crop_mapping_path = os.path.join(precropped_dir, 'crop_mapping.csv')
        if not os.path.exists(crop_mapping_path):
            raise FileNotFoundError(
                f"Crop mapping file not found: {crop_mapping_path}\n"
                f"Please run: python precrop_images.py --output_dir {precropped_dir}"
            )
        
        crop_df = pd.read_csv(crop_mapping_path)
        
        # Filter crops by fold
        train_crop_df = crop_df[crop_df['fold'] != fold].copy()
        val_crop_df = crop_df[crop_df['fold'] == fold].copy()
        
        print(f"  Train crops: {len(train_crop_df)} (from {len(train_df)} images)")
        print(f"  Val crops: {len(val_crop_df)} (from {len(val_df)} images)")
        
        train_dataset = SteelPrecroppedDataset(
            train_crop_df, precropped_dir, mean, std, phase='train'
        )
        val_dataset = SteelPrecroppedDataset(
            val_crop_df, precropped_dir, mean, std, phase='val'
        )
        
    elif use_crops:
        print(f"Using CROP mode: {crop_height}x{crop_width} with stride {crop_stride}")
        train_dataset = SteelCropsDataset(
            train_df, data_folder, mean, std, 
            phase='train',
            crop_height=crop_height,
            crop_width=crop_width,
            stride=crop_stride
        )
        val_dataset = SteelCropsDataset(
            val_df, data_folder, mean, std,
            phase='val',
            crop_height=crop_height,
            crop_width=crop_width,
            stride=crop_stride
        )
    else:
        print(f"Using FULL IMAGE mode: 256x1600")
        train_dataset = SteelDataset(train_df, data_folder, mean, std, phase='train')
        val_dataset = SteelDataset(val_df, data_folder, mean, std, phase='val')
        print(f"  TRAIN - Total samples: {len(train_dataset)}")
        print(f"  VAL - Total samples: {len(val_dataset)}")
    
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
        pin_memory=False,
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

