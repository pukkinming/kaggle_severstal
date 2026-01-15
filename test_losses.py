"""
Test script to verify new loss functions work correctly
"""
import torch
import torch.nn.functional as F
from losses import BCEWithPosWeightLoss, BCEDiceWithPosWeightLoss, DiceLoss

def test_bce_pos_weight():
    """Test BCEWithPosWeightLoss"""
    print("Testing BCEWithPosWeightLoss...")
    
    # Create sample data
    batch_size = 2
    num_classes = 4
    height, width = 256, 1600
    
    predictions = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()
    
    # Test loss
    criterion = BCEWithPosWeightLoss(pos_weight=(2.0, 2.0, 1.0, 1.5))
    loss = criterion(predictions, targets)
    
    print(f"  Input shape: {predictions.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    
    # Test backward
    loss.backward()
    print(f"  ✓ Backward pass successful")
    
    # Compare with manual calculation
    pos_weight = torch.tensor([2.0, 2.0, 1.0, 1.5]).view(1, -1, 1, 1)
    manual_loss = F.binary_cross_entropy_with_logits(
        predictions.detach(), targets, pos_weight=pos_weight, reduction='mean'
    )
    print(f"  Manual loss: {manual_loss.item():.4f}")
    print(f"  Difference: {abs(loss.item() - manual_loss.item()):.6f}")
    
    assert abs(loss.item() - manual_loss.item()) < 1e-5, "Loss calculation mismatch!"
    print("  ✓ BCEWithPosWeightLoss test passed!\n")


def test_bce_dice_pos_weight():
    """Test BCEDiceWithPosWeightLoss"""
    print("Testing BCEDiceWithPosWeightLoss...")
    
    # Create sample data
    batch_size = 2
    num_classes = 4
    height, width = 256, 1600
    
    predictions = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()
    
    # Test loss
    criterion = BCEDiceWithPosWeightLoss(
        pos_weight=(2.0, 2.0, 1.0, 1.5), 
        bce_weight=0.75, 
        dice_weight=0.25
    )
    loss = criterion(predictions, targets)
    
    print(f"  Input shape: {predictions.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    
    # Test backward
    loss.backward()
    print(f"  ✓ Backward pass successful")
    
    # Verify it's a combination of BCE and Dice
    bce_criterion = BCEWithPosWeightLoss(pos_weight=(2.0, 2.0, 1.0, 1.5))
    dice_criterion = DiceLoss()
    
    bce_loss = bce_criterion(predictions.detach(), targets)
    dice_loss = dice_criterion(predictions.detach(), targets)
    expected_loss = 0.75 * bce_loss + 0.25 * dice_loss
    
    print(f"  BCE component: {bce_loss.item():.4f}")
    print(f"  Dice component: {dice_loss.item():.4f}")
    print(f"  Expected combined: {expected_loss.item():.4f}")
    print(f"  Actual combined: {loss.item():.4f}")
    print(f"  Difference: {abs(loss.item() - expected_loss.item()):.6f}")
    
    assert abs(loss.item() - expected_loss.item()) < 1e-5, "Combined loss calculation mismatch!"
    print("  ✓ BCEDiceWithPosWeightLoss test passed!\n")


def test_gpu_compatibility():
    """Test that losses work on GPU if available"""
    if not torch.cuda.is_available():
        print("GPU not available, skipping GPU test\n")
        return
    
    print("Testing GPU compatibility...")
    device = torch.device("cuda:0")
    
    # Create sample data on GPU
    batch_size = 2
    num_classes = 4
    height, width = 256, 1600
    
    predictions = torch.randn(batch_size, num_classes, height, width, requires_grad=True).to(device)
    targets = torch.randint(0, 2, (batch_size, num_classes, height, width)).float().to(device)
    
    # Test BCE with pos_weight
    criterion1 = BCEWithPosWeightLoss(pos_weight=(2.0, 2.0, 1.0, 1.5))
    loss1 = criterion1(predictions, targets)
    loss1.backward()
    print(f"  ✓ BCEWithPosWeightLoss works on GPU (loss: {loss1.item():.4f})")
    
    # Test combined loss
    criterion2 = BCEDiceWithPosWeightLoss(pos_weight=(2.0, 2.0, 1.0, 1.5))
    predictions2 = torch.randn(batch_size, num_classes, height, width, requires_grad=True).to(device)
    loss2 = criterion2(predictions2, targets)
    loss2.backward()
    print(f"  ✓ BCEDiceWithPosWeightLoss works on GPU (loss: {loss2.item():.4f})")
    
    print("  ✓ GPU compatibility test passed!\n")


if __name__ == "__main__":
    print("="*70)
    print("Testing New Loss Functions")
    print("="*70 + "\n")
    
    test_bce_pos_weight()
    test_bce_dice_pos_weight()
    test_gpu_compatibility()
    
    print("="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nYou can now use these loss functions in training:")
    print("  - Set LOSS='bce_pos_weight' in config.py for BCE with pos_weight")
    print("  - Set LOSS='bce_dice_pos_weight' in config.py for 0.75*BCE + 0.25*Dice")

