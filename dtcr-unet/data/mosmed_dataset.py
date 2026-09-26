"""MosMedDataset wrapper for backwards compatibility."""

from .datasets import DualTaskDataset, get_dataloaders

MosMedDataset = DualTaskDataset

__all__ = ["MosMedDataset", "DualTaskDataset", "get_dataloaders"]
