"""
Loss functions and evaluation metrics
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCEWithPosWeightLoss(nn.Module):
    """
    BCE Loss with custom pos_weight for each class
    pos_weight: weight of positive examples (defects) for each class
    Higher pos_weight gives more importance to detecting defects
    
    pos_weight=(2.0, 2.0, 1.0, 1.5) means:
    - Class 1: 2x weight for positive examples
    - Class 2: 2x weight for positive examples  
    - Class 3: 1x weight (balanced)
    - Class 4: 1.5x weight for positive examples
    """
    def __init__(self, pos_weight=(2.0, 2.0, 1.0, 1.5)):
        super(BCEWithPosWeightLoss, self).__init__()
        # Register as buffer so it moves with model to GPU
        self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
    
    def forward(self, predictions, targets):
        # predictions shape: [batch, num_classes, height, width]
        # targets shape: [batch, num_classes, height, width]
        # pos_weight shape: [num_classes] -> reshape to [1, num_classes, 1, 1] for broadcasting
        
        # Ensure pos_weight is on the same device as predictions
        pos_weight = self.pos_weight.to(predictions.device).view(1, -1, 1, 1)
        
        # Use PyTorch's functional API
        # F.binary_cross_entropy_with_logits applies pos_weight correctly
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, pos_weight=pos_weight, reduction='mean'
        )
        
        return loss


class BCEDiceWithPosWeightLoss(nn.Module):
    """
    Combined BCE (with pos_weight) and Dice Loss
    Default: 0.75*BCE + 0.25*Dice
    """
    def __init__(self, pos_weight=(2.0, 2.0, 1.0, 1.5), bce_weight=0.75, dice_weight=0.25):
        super(BCEDiceWithPosWeightLoss, self).__init__()
        self.bce = BCEWithPosWeightLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def predict(X, threshold):
    """
    X is sigmoid output of the model
    """
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def compute_dice_class(predictions, targets, threshold=0.5):
    """
    Compute Dice score for positive and negative samples separately
    Returns:
        dice: overall dice score
        dice_neg: dice for negative samples (no defect)
        dice_pos: dice for positive samples (has defect)
        num_neg: number of negative samples
        num_pos: number of positive samples
    """
    batch_size = len(targets)
    
    with torch.no_grad():
        predictions = predictions.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        p = (predictions > threshold).float()
        t = (targets > 0.5).float()
        
        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        
        neg_index = torch.nonzero(t_sum == 0, as_tuple=False)
        pos_index = torch.nonzero(t_sum >= 1, as_tuple=False)
        
        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-6)
        
        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        
        num_neg = len(neg_index)
        num_pos = len(pos_index)
    
    return dice, dice_neg, dice_pos, num_neg, num_pos


def compute_iou(pred, label, classes, ignore_index=255, only_present=True):
    """
    Computes IoU for one ground truth mask and predicted mask
    """
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    """
    Computes mean IoU for a batch of ground truth masks and predicted masks
    """
    ious = []
    preds = np.copy(outputs)
    labels = np.array(labels)
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_iou(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def compute_per_class_metrics(predictions, targets, threshold=0.5, num_classes=4):
    """
    Compute metrics for each class separately
    
    Returns:
        per_class_dice: dice score for each class
        per_class_precision: precision for each class
        per_class_recall: recall for each class
        per_class_iou: IoU for each class
    """
    batch_size = predictions.shape[0]
    
    per_class_dice = []
    per_class_precision = []
    per_class_recall = []
    per_class_iou = []
    
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        
        for cls in range(num_classes):
            pred_cls = predictions[:, cls, :, :].contiguous().view(batch_size, -1)
            target_cls = targets[:, cls, :, :].contiguous().view(batch_size, -1)
            
            p = (pred_cls > threshold).float()
            t = (target_cls > 0.5).float()
            
            # Dice
            intersection = (p * t).sum(-1)
            dice = (2. * intersection + 1e-6) / (p.sum(-1) + t.sum(-1) + 1e-6)
            
            # Precision and Recall
            tp = (p * t).sum(-1)
            fp = (p * (1 - t)).sum(-1)
            fn = ((1 - p) * t).sum(-1)
            
            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            
            # IoU
            iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            
            per_class_dice.append(dice.mean().item())
            per_class_precision.append(precision.mean().item())
            per_class_recall.append(recall.mean().item())
            per_class_iou.append(iou.mean().item())
    
    return per_class_dice, per_class_precision, per_class_recall, per_class_iou


def compute_classification_metrics(predictions, targets, threshold=0.5, num_classes=4):
    """
    Compute classification metrics (whether each class is present or not)
    
    Returns:
        per_class_accuracy: classification accuracy for each class
        per_class_f1: classification F1 score for each class
    """
    batch_size = predictions.shape[0]
    
    per_class_accuracy = []
    per_class_f1 = []
    
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        
        for cls in range(num_classes):
            pred_cls = predictions[:, cls, :, :].contiguous().view(batch_size, -1)
            target_cls = targets[:, cls, :, :].contiguous().view(batch_size, -1)
            
            # Classification: does this class exist in the image?
            pred_exists = (pred_cls.max(dim=1)[0] > threshold).float()
            target_exists = (target_cls.max(dim=1)[0] > 0.5).float()
            
            # Accuracy
            accuracy = (pred_exists == target_exists).float().mean()
            
            # F1 score
            tp = ((pred_exists == 1) & (target_exists == 1)).float().sum()
            fp = ((pred_exists == 1) & (target_exists == 0)).float().sum()
            fn = ((pred_exists == 0) & (target_exists == 1)).float().sum()
            
            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            per_class_accuracy.append(accuracy.item())
            per_class_f1.append(f1.item())
    
    return per_class_accuracy, per_class_f1


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


class MetricTracker:
    """
    Track metrics during training and validation
    """
    def __init__(self, phase, threshold=0.5, num_classes=4):
        self.phase = phase
        self.threshold = threshold
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.per_class_dice = [[] for _ in range(self.num_classes)]
        self.per_class_precision = [[] for _ in range(self.num_classes)]
        self.per_class_recall = [[] for _ in range(self.num_classes)]
        self.per_class_iou = [[] for _ in range(self.num_classes)]
        self.per_class_cls_accuracy = [[] for _ in range(self.num_classes)]
        self.per_class_cls_f1 = [[] for _ in range(self.num_classes)]
    
    def update(self, predictions, targets):
        """
        Update metrics with a batch of predictions and targets
        """
        probs = torch.sigmoid(predictions)
        
        # Overall dice
        dice, dice_neg, dice_pos, _, _ = compute_dice_class(probs, targets, self.threshold)
        self.dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        
        # IoU
        preds = predict(probs.cpu().numpy(), self.threshold)
        iou = compute_iou_batch(preds, targets.cpu().numpy(), classes=[1])
        self.iou_scores.append(iou)
        
        # Per-class segmentation metrics
        pc_dice, pc_prec, pc_recall, pc_iou = compute_per_class_metrics(
            predictions, targets, self.threshold, self.num_classes
        )
        for cls in range(self.num_classes):
            self.per_class_dice[cls].append(pc_dice[cls])
            self.per_class_precision[cls].append(pc_prec[cls])
            self.per_class_recall[cls].append(pc_recall[cls])
            self.per_class_iou[cls].append(pc_iou[cls])
        
        # Per-class classification metrics
        pc_cls_acc, pc_cls_f1 = compute_classification_metrics(
            predictions, targets, self.threshold, self.num_classes
        )
        for cls in range(self.num_classes):
            self.per_class_cls_accuracy[cls].append(pc_cls_acc[cls])
            self.per_class_cls_f1[cls].append(pc_cls_f1[cls])
    
    def get_metrics(self):
        """
        Get averaged metrics
        """
        metrics = {
            'dice': np.nanmean(self.dice_scores),
            'dice_neg': np.nanmean(self.dice_neg_scores),
            'dice_pos': np.nanmean(self.dice_pos_scores),
            'iou': np.nanmean(self.iou_scores),
        }
        
        # Per-class metrics
        for cls in range(self.num_classes):
            metrics[f'class_{cls+1}_dice'] = np.nanmean(self.per_class_dice[cls])
            metrics[f'class_{cls+1}_precision'] = np.nanmean(self.per_class_precision[cls])
            metrics[f'class_{cls+1}_recall'] = np.nanmean(self.per_class_recall[cls])
            metrics[f'class_{cls+1}_iou'] = np.nanmean(self.per_class_iou[cls])
            metrics[f'class_{cls+1}_cls_accuracy'] = np.nanmean(self.per_class_cls_accuracy[cls])
            metrics[f'class_{cls+1}_cls_f1'] = np.nanmean(self.per_class_cls_f1[cls])
        
        return metrics




