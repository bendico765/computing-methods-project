import torch

def dice_loss(preds: torch.Tensor, targets: torch.Tensor, smooth=1e-6):
    """
    Dice = 2 x | A & B | / |A| + |B|
    
    :param preds: tensor of predictions
    :param targets: tensor of targets 
    """
    # sigmoid maps unbounded numbers to continous probability range [0,1]
    sigmoid_preds = torch.sigmoid(preds)

    sigmoid_preds = sigmoid_preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (sigmoid_preds * targets).sum()
    union = sigmoid_preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice