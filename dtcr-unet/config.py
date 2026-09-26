"""Configuration file containing all hyperparameters for DTCR-U-Net training and evaluation.

Hyperparameters match the experimental protocol from Section 4 of the DTCR-U-Net paper.
"""

from dataclasses import dataclass
import torch


@dataclass
class DTCRConfig:
    # Data parameters
    data_dir: str = "data/processed"
    image_size: int = 256
    context_slices: int = 1       # [-1, 0, +1] adjacent slice context
    num_workers: int = 2
    labeled_ratio: float = 0.4    # 40% labeled, 60% unlabeled for semi-supervised training

    # Training parameters
    epochs: int = 80
    batch_size: int = 16          # 16 or 32 depending on GPU VRAM
    lr: float = 1e-3              # Initial learning rate
    weight_decay: float = 1e-4    # AdamW weight decay
    min_lr: float = 1e-6          # Cosine Annealing minimum learning rate

    # Loss hyperparameters (Section 4.6)
    beta: float = 0.5             # Gradient constraint weight for Supervised LSF loss
    lambda1: float = 1.0          # Gradient consistency weight for Dual-Task Consistency loss
    lambda2: float = 1.0          # Interaction enhancement weight for Dual-Task Consistency loss
    lambda_smooth: float = 0.3    # Level Set smoothing regularization parameter

    # Model architecture
    in_channels: int = 1
    n_classes: int = 1
    base_channels: int = 64
    bilinear: bool = False

    # Output & Checkpoint paths
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    seed: int = 42

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"