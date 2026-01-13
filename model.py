"""
UNet model with various encoder backbones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def get_model(arch='unet', encoder='resnet18', encoder_weights='imagenet', 
              num_classes=4, activation=None):
    """
    Get segmentation model
    
    Args:
        arch: architecture name (unet, unetplusplus, fpn, linknet, etc.)
        encoder: encoder backbone (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: pretrained weights (imagenet, None)
        num_classes: number of output classes
        activation: activation function (sigmoid, softmax, None)
    
    Returns:
        model: segmentation model
    """
    if arch == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    elif arch == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    elif arch == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    elif arch == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    elif arch == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    elif arch == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    elif arch == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = get_model(arch='unet', encoder='resnet18', encoder_weights='imagenet', 
                     num_classes=4, activation=None)
    print(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 1600)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")




