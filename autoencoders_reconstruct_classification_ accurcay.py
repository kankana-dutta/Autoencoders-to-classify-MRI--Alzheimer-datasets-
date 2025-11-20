#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-Supervised Autoencoder (4-class classification)
----------------------------------------------------
Dataset: Falah/Alzheimer_MRI
Classes:
  0 - NonDemented
  1 - VeryMildDemented
  2 - MildDemented
  3 - ModerateDemented

Architecture:
  Encoder -> latent(64)
           -> Decoder (MSE reconstruction)
           -> Classifier (4-class softmax)

Loss = MSE + λ * CrossEntropy
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for easier debugging on CUDA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Define Autoencoder + Classifier
# -----------------------------
class AE_with_Classifier(nn.Module):
    def __init__(self, latent_dim=64, img_size=128, num_classes=4):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        flat_dim = 64 * (img_size // 8) * (img_size // 8)

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Tanh()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_mu(h)

        h_dec = self.fc_dec(z).view(-1, 64, x.shape[-2] // 8, x.shape[-1] // 8)
        x_hat = self.decoder(h_dec)

        logits = self.classifier(z)
        return x_hat, logits, z


# -----------------------------
# 2. Prepare Dataset
# -----------------------------
def get_dataloaders(img_size=128, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def tfm(batch):
        batch["image"] = [transform(i) for i in batch["image"]]
        return batch

    ds = load_dataset("Falah/Alzheimer_MRI").with_transform(tfm)
    print("\nDataset label names:", ds["train"].features["label"].names)
    print("Checking label distribution...")

    labels_train = ds["train"]["label"]
    labels_test = ds["test"]["label"]
    print("Train label counts:", {i: labels_train.count(i) for i in set(labels_train)})
    print("Test label counts:", {i: labels_test.count(i) for i in set(labels_test)})

    combo = ConcatDataset([ds["train"], ds["test"]])
    loader = DataLoader(combo, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


# -----------------------------
# 3. Training loop
# -----------------------------
def train(model, loader, device, epochs=40, lambda_cls=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    model.train()
    losses_recon, losses_cls = [], []

    for ep in range(epochs):
        total_recon, total_cls = 0, 0

        for b in loader:
            x = b["image"].to(device)
            y = b["label"].to(device).long()

            opt.zero_grad()
            x_hat, logits, _ = model(x)

            loss_recon = recon_loss_fn(x_hat, x)
            loss_cls = cls_loss_fn(logits, y)
            loss = loss_recon + lambda_cls * loss_cls

            loss.backward()
            opt.step()

            total_recon += loss_recon.item()
            total_cls += loss_cls.item()

        losses_recon.append(total_recon / len(loader))
        losses_cls.append(total_cls / len(loader))
        print(f"Epoch [{ep+1}/{epochs}] | Recon={losses_recon[-1]:.4f} | Cls={losses_cls[-1]:.4f}")

    # Plot losses
    plt.figure(figsize=(7, 4))
    plt.plot(losses_recon, label="Reconstruction Loss (MSE)")
    plt.plot(losses_cls, label="Classification Loss (CrossEntropy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training Losses (λ={lambda_cls})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model


# -----------------------------
# 4. Visualize Reconstructions (one per class)
# -----------------------------
def visualize_reconstructions(model, loader, device):
    model.eval()

    originals = {}
    reconstructions = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            x_hat, _, _ = model(x)

            for img, rec, label in zip(x, x_hat, y):
                label = int(label.item())
                if label not in originals:
                    originals[label] = img.cpu().numpy()
                    reconstructions[label] = rec.cpu().numpy()

                if len(originals) == 4:
                    break
            if len(originals) == 4:
                break

    plt.figure(figsize=(10, 8))
    for idx, label in enumerate([0, 1, 2, 3]):
        orig = originals[label][0]
        rec = reconstructions[label][0]

        plt.subplot(4, 2, idx*2 + 1)
        plt.imshow(orig, cmap='gray')
        plt.title(f"Original (Class {label})")
        plt.axis("off")

        plt.subplot(4, 2, idx*2 + 2)
        plt.imshow(rec, cmap='gray')
        plt.title(f"Reconstructed (Class {label})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Evaluate Classification Accuracy + Latent Extraction
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_latent, all_labels = [], []

    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            y = b["label"].to(device)
            _, logits, z = model(x)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_latent.append(z.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    acc = 100 * correct / total
    print(f"\nOverall classification accuracy: {acc:.2f}%")

    Z = np.concatenate(all_latent)
    labels = np.concatenate(all_labels)

    plot_latent_distribution(Z, labels)
    return acc


# -----------------------------
# 6. Latent Space Plot
# -----------------------------
def plot_latent_distribution(Z, labels):
    non_mask = labels == 0
    dem_mask = np.isin(labels, [1, 2, 3])

    Z_non = Z[non_mask]
    Z_dem = Z[dem_mask]

    dims = np.arange(Z.shape[1])
    mean_non, std_non = Z_non.mean(axis=0), Z_non.std(axis=0)
    mean_dem, std_dem = Z_dem.mean(axis=0), Z_dem.std(axis=0)

    plt.figure(figsize=(10, 5))
    plt.errorbar(dims, mean_non, yerr=std_non, fmt='-o', capsize=3,
                 label="Non-Demented", alpha=0.7, color='tab:blue')
    plt.errorbar(dims, mean_dem, yerr=std_dem, fmt='-o', capsize=3,
                 label="Demented", alpha=0.7, color='tab:red')
    plt.xlabel("Latent Dimension Index (0–63)")
    plt.ylabel("Mean ± 1σ")
    plt.title("Latent-Space Differences per Dimension")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 7. Main Function
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    latent_dim = 64
    epochs = 40
    lambda_cls = 0.1

    loader = get_dataloaders()
    model = AE_with_Classifier(latent_dim=latent_dim, num_classes=4).to(device)

    print(f"\nTraining semi-supervised 4-class AE for {epochs} epochs...\n")
    model = train(model, loader, device, epochs, lambda_cls)

    print("\nEvaluating model classification accuracy...\n")
    evaluate(model, loader, device)

    print("\nVisualizing original vs reconstructed MRIs...")
    visualize_reconstructions(model, loader, device)


# -----------------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

