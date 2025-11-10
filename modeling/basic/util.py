import os
import torch
from typing import Optional
from enum import Enum
import numpy as np

class CfgType(Enum):
	NORMAL = "normal"
	CFG_TEXT = "cfg_text"
	CFG_IMAGE = "cfg_img"

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

def save_map(is_save: bool, save_dir: str, map: torch.Tensor, mode: str,
    timestep: Optional[int] = None, layer_idx: Optional[int] = None, 
    cfg_type: Optional[str] = None
):
	should_save = layer_idx is not None and is_save and timestep is not None and timestep >= 0
	if not should_save:
		return
	assert (mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None) or (mode == "und" and cfg_type is None), "Invalid save conditions."
	
	entry_to_save = {
		"mlp": map.cpu()
	}
	filename = f"{mode}_mlp_layer_{layer_idx}_ts_{timestep}.pt" if mode == "und" else f"{mode}_mlp_{cfg_type}_layer_{layer_idx}_ts_{timestep}.pt"
	os.makedirs(save_dir, exist_ok=True)
	save_path = os.path.join(save_dir, filename)
	if not os.path.exists(save_path):
		torch.save([entry_to_save], save_path)

def save_swift(is_save: bool, save_dir: str, name: str, data: torch.Tensor):
	if not is_save:
		return
	os.makedirs(save_dir, exist_ok=True)
	save_path = os.path.join(save_dir, f"{name}.pt")
	if not os.path.exists(save_path):
		torch.save(data.cpu(), save_path)

class MLPArgs:
	def __init__(
		self,
		use_custom_mlp: bool = False,
		use_quantized_w: bool = False,
		use_similarity: bool = False,
		save_mlp: Optional[str]= None,
		save_dir: Optional[str]= None,
	):
		if use_custom_mlp:
			print("Using custom MLP with CFG sparsity tracking.")
		
		self.use_custom_mlp = use_custom_mlp
		self.use_quantized_w = use_quantized_w
		self.use_similarity = use_similarity
		self.save_mlp = save_mlp
		self.save_dir = save_dir

		self.sparsity = {
			"cfg_text": np.zeros((49 ,28), dtype=np.float32),
			"cfg_img": np.zeros((49, 28), dtype=np.float32)
		}

		self.mot_sparsity = {
			"cfg_text": np.zeros((49 ,28), dtype=np.float32),
			"cfg_img": np.zeros((49, 28), dtype=np.float32)
		}

	def clear_sparsity(self):
		self.sparsity = {
			"cfg_text": np.zeros((49 ,28), dtype=np.float32),
			"cfg_img": np.zeros((49, 28), dtype=np.float32)
		}

		self.mot_sparsity = {
			"cfg_text": np.zeros((49 ,28), dtype=np.float32),
			"cfg_img": np.zeros((49, 28), dtype=np.float32)
		}