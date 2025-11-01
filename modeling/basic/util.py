import torch

def compute_coverage(ref: torch.Tensor, pred: torch.Tensor) -> float:
		"""Compute coverage metric between reference and predicted attention masks.

		Args:
				ref (torch.Tensor): Reference attention mask of shape (Lq, Lk), with binary values (0 or 1).
				pred (torch.Tensor): Predicted attention mask of shape (Lq, Lk), with binary values (0 or 1).

		Returns:
				float: Coverage score between reference and predicted masks.
		"""
		assert(ref.shape == pred.shape), "Reference and predicted masks must have the same shape."
		# Compute intersection and union
		intersection = (ref & pred).float().sum()
		union = (ref | pred).float().sum()

		# Compute coverage
		coverage = intersection / (union + 1e-8)
		return coverage.item()