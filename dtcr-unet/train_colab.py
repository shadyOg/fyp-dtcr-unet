"""Minimal training script intended for use in Google Colab.

Run after downloading MosMed and installing requirements.
Example:
  python train_colab.py --data /content/mosmeddata --epochs 1 --batch-size 2
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.mosmed_dataset import MosMedDataset
from models.unet.unet_model import UNet


def get_loader(data_dir, split='train', batch_size=2, num_workers=2):
    ds = MosMedDataset(root_dir=data_dir, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for i, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        # assume binary segmentation
        if outputs.shape[1] == 1:
            loss = criterion(outputs, masks)
        else:
            loss = criterion(outputs, masks.long().squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'batch {i} loss {loss.item():.4f}')
    return total_loss / (i + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset root')
    parser.add_argument('--checkpoint', action='store_true', help='Enable activation checkpointing')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
    if args.checkpoint:
        model.use_checkpointing()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = get_loader(args.data, batch_size=args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch} avg loss {loss:.4f}')
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                   os.path.join(args.out, f'model_epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()
