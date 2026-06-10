import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

def center_crop(enc_feat: torch.Tensor, target_feat: torch.Tensor):
    """
    Crop encoder feature map to match target feature map size.
    """
    _, _, H, W = target_feat.shape
    _, _, H_enc, W_enc = enc_feat.shape
    
    delta_h = H_enc - H
    delta_w = W_enc - W
    
    top = delta_h // 2
    left = delta_w // 2

    return enc_feat.narrow(2, top, H).narrow(3, left, W)

class DoubleConv(nn.Module):
    """
    """
    def __init__(self, in_ch: int, out_ch:int, kernel_size:int = 3, padding:int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    
class UNet(nn.Module):
    """
    For convolutional layer:
    Output size O of a single dimension: O = (I - K + 2P)/S + 1
    with I: input size, K: kernel size, P: padding, S: stride
    
    For pooling layer:
    Output size O of a single dimension: O = (I - K)/S + 1
    """
    def __init__(self, n_class):
        super().__init__()

        # ---ENCODER---
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # TODO: verify stride

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv(512, 1024)

        # ---DECODER---
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # ---------------- Encoder ----------------
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        b = self.bottleneck(p4)

        # ---------------- Decoder ----------------
        d4 = self.up4(b)
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.outconv(d1) 
    
def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device="cpu"):
    model.train() # set the model to training mode
    
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device) # (B, 1, W, H)
        y = y.to(device) # (B, 2, W, H) 
        
        # compute prediction and loss
        logits = model(X)
        pred_probs = torch.softmax(logits, dim=1)
        loss = loss_fn(y.float(), pred_probs)
        
        # backpropagation
        loss.backward() # backpropagate the prediction loss
        optimizer.step() # adjust the parameters by the gradients collected in the backward pass
        optimizer.zero_grad() # reset the gradients of model parameters

        loss, current = loss.item(), batch * batch_size + len(X)
        total_loss += loss
        
        print(f"loss: {loss:>6f}  [{current:>5d}/{size:>5d}]")

    return total_loss/len(dataloader)

def test_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval() # set model to evaluation mode
    
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            pred_probs = torch.softmax(logits, dim=1)
            test_loss += loss_fn(y.float(), pred_probs).item()
    
    return test_loss / len(dataloader) # return average on the batches