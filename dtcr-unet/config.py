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
    context_slices: int = 2       # 2 adjacent slices above and below (5 slices total) per Section 4.1 line 849
    num_workers: int = 2
    labeled_ratio: float = 0.4    # 40% labeled, 60% unlabeled for semi-supervised training per Section 4.2

    # Training parameters (Section 4.1)
    epochs: int = 80              # 80 epochs per paper Section 4.1
    batch_size: int = 32          # Batch size 32 per paper Section 4.1
    lr: float = 0.001             # Initial learning rate 0.001 per paper Section 4.1
    weight_decay: float = 1e-4    # AdamW weight decay
    min_lr: float = 1e-6          # Cosine Annealing minimum learning rate

    # Loss hyperparameters (Section 3.1, Section 4.6 & Fig. 7)
    beta: float = 0.5             # Gradient constraint weight for Supervised LSF loss (Section 4.6)
    lambda1: float = 0.3          # Gradient consistency weight for Dual-Task Consistency loss (optimal per Fig. 7)
    lambda2: float = 1.0          # Interaction enhancement weight for Dual-Task Consistency loss (Section 4.6)
    lambda_context: float = 0.0   # Context-guided loss weight (0.0 matching Eq. 4 & Section 4.6)
    lambda_smooth: float = 0.3    # Level Set smoothing regularization parameter (Fig. 7)
    consistency_weight: float = 0.1 # Maximum weight for dynamic consistency lambda_d(t) (default: 0.1 per DTC AAAI 2021)
    bce_weight: float = 0.0       # BCE loss weight (0.0 for pure Soft Dice per Eq. 8; >0 if compound loss)
    dice_weight: float = 1.0      # Soft Dice weight (Eq. 8 primary objective)
    pos_weight: float = 20.0      # Positive lesion pixel weighting if BCE is used (balances 99.76% background)
    use_focal: bool = False       # Use Focal Loss if BCE is enabled
    fuse_val: bool = False        # Primary segmentation head f1 inference (single forward pass per Section 4.1)
    val_threshold: float = 0.50   # Standard binarization threshold (Section 4.3)

    # Model architecture (Table 1: 18.65M parameters)
    in_channels: int = 1
    n_classes: int = 1
    base_channels: int = 64
    num_stages: int = 4           # 4-stage encoder / 3-stage decoder (18.65M params per Table 1)
    bilinear: bool = False

    # Output & Checkpoint paths
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    seed: int = 42

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"