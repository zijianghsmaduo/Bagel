import torch
from typing import Optional, Tuple
import os

from torch.nn import functional as F

import math

import numpy as np


class BlockSparsify:
	def __init__(self, group_size: int = 128, top_k: float = 0.1, threshold: float = 4e-5): 
		self.group_size = group_size
		self.top_k = top_k
		self.threshold = threshold
		self.threshold_max = False

	def sparsify_kv_cache_threshold(
			self, q: torch.Tensor, k: torch.Tensor, 
			q_range: torch.Tensor, k_range: torch.Tensor,
			vae_range: torch.Tensor, vit_range: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			q: (H, L, D)
			k: (H, N, D)
		"""
		# print("sparsify_kv_cache called")
		# q = q[:, q_range[0]:q_range[1], :]
		# k = k[:, k_range[0]:k_range[1], :]
		
		H, L, D = q.shape
		_, N, _ = k.shape
		group_size = self.group_size

		L_NEW = L
		N_NEW = N

		num_groups_q = L // group_size
		padding_value = float('-inf') if self.threshold_max else 0.0
		padding_q = (group_size - (L % group_size)) % group_size
		if padding_q > 0:
			q = F.pad(q, (0, 0, 0, padding_q), value=padding_value)
			num_groups_q += 1
			L_NEW += padding_q
		num_groups_k = N // group_size
		padding_k = (group_size - (N % group_size)) % group_size
		if padding_k > 0:
			k = F.pad(k, (0, 0, 0, padding_k), value=padding_value)
			num_groups_k += 1
			N_NEW += padding_k
			# pad后，k的长度变了
		assert(q.size(1) == L_NEW and k.size(1) == N_NEW)
		q = q.view(H, num_groups_q, group_size, D)
		k = k.view(H, num_groups_k, group_size, D)
		representative_q = torch.zeros(H, num_groups_q, D, device=q.device, dtype=q.dtype)
		representative_k = torch.zeros(H, num_groups_k, D, device=k.device, dtype=k.dtype)
		if self.threshold_max:
			representative_q = q.max(dim=2).values  # (H, num_groups_q, D)
			representative_k = k.max(dim=2).values  # (H, num_groups_k, D)
		else:
			representative_q = q.mean(dim=2)  # (H, num_groups_q, D)
			representative_k = k.mean(dim=2)  # (H, num_groups_k, D)
		assert(representative_q.size(0) == H and representative_q.size(1) == num_groups_q and representative_q.size(2) == D)
		assert(representative_k.size(0) == H and representative_k.size(1) == num_groups_k and representative_k.size(2) == D)
		representative_attn_scores = torch.bmm(representative_q, representative_k.transpose(-2, -1) / math.sqrt(D))  # (H, num_groups_q, num_groups_k)
		representative_attn_scores = torch.softmax(representative_attn_scores, dim=-1)
		# print(f"max = {representative_attn_scores.max()}")
		# print(f"threshold = {self.threshold}")
		# nr_maintain = max(1, int(num_groups_k * num_groups_q * top_k))
		# representative_attn_scores = representative_attn_scores.view(H, -1)
		block_mask = representative_attn_scores < self.threshold

		full_mask = torch.repeat_interleave(block_mask, repeats=group_size, dim=1)
		full_mask = torch.repeat_interleave(full_mask, repeats=group_size, dim=2)
		# print(f"sparsity {torch.mean(full_mask.float())}")
		final_attn_mask = full_mask
		# True 代表“屏蔽”
		final_attn_mask = final_attn_mask[:, :L, :N]
		assert(final_attn_mask.size(0) == H and final_attn_mask.size(1) == L and final_attn_mask.size(2) == N)
		return final_attn_mask[:, q_range[0]:q_range[1], k_range[0]:k_range[1]], representative_attn_scores

	def padding(self, x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
		H, L, D = x.shape
		padding_len = (group_size - (L % group_size)) % group_size
		if padding_len > 0:
			if self.threshold_max:
				pad_value = float('-inf')
				x = F.pad(x, (0, 0, 0, padding_len), value=pad_value)  # (H, L + padding_len, D)
			else:
				remain_len = L % group_size
				last_group = x[:, -remain_len:, :]
				pad_value = last_group.mean(dim=1, keepdim=True)  # (H, 1, D)
				pad_tensor = pad_value.repeat(1, padding_len, 1)  # (H, padding_len, D)
				x = torch.cat([x, pad_tensor], dim=1)  # (H, L + padding_len, D)
		return x, padding_len

	def sparsify_kv_cache_threshold_fine(
			self, q: torch.Tensor, k: torch.Tensor, 
			q_range: torch.Tensor, k_range: torch.Tensor,
			vae_range: torch.Tensor, vit_range: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			q: (H, L, D)
			k: (H, N, D)
		"""
		# print("sparsify_kv_cache called")
		# q = q[:, q_range[0]:q_range[1], :]
		# k = k[:, k_range[0]:k_range[1], :]
		
		H, L, D = q.shape
		_, N, _ = k.shape
		group_size = self.group_size

		L_NEW = L
		N_NEW = N

		pre_vae_vit_range = torch.tensor((0, vae_range[0]), device=k.device, dtype=torch.long)
		post_vae_vit_range = torch.tensor((vit_range[1], N), device=k.device, dtype=torch.long)

		pre_vae_vit_k = k[:, pre_vae_vit_range[0]:pre_vae_vit_range[1], :]
		vae = k[:, vae_range[0]:vae_range[1], :]
		vit = k[:, vit_range[0]:vit_range[1], :]
		post_vae_vit_k = k[:, post_vae_vit_range[0]:post_vae_vit_range[1], :]

		pre_vae_vit_k_padded, pre_vae_vit_k_padded_len = self.padding(pre_vae_vit_k, group_size)
		vae_padded, vae_padded_len = self.padding(vae, group_size)
		vit_padded, vit_padded_len = self.padding(vit, group_size)
		post_vae_vit_k_padded, post_vae_vit_k_padded_len = self.padding(post_vae_vit_k, group_size)

		k_padded = torch.cat([pre_vae_vit_k_padded, vae_padded, vit_padded, post_vae_vit_k_padded], dim=1)
		pre_vae_vit_padded_range = torch.tensor((
			0, pre_vae_vit_k_padded.size(1)
		), device=k.device, dtype=torch.long)
		vae_padded_range = torch.tensor((
			pre_vae_vit_k_padded.size(1),
			pre_vae_vit_k_padded.size(1) + vae_padded.size(1) - vae_padded_len
		), device=k.device, dtype=torch.long)
		vit_padded_range = torch.tensor((
			pre_vae_vit_k_padded.size(1) + vae_padded.size(1),
			pre_vae_vit_k_padded.size(1) + vae_padded.size(1) + vit_padded.size(1) - vit_padded_len
		), device=k.device, dtype=torch.long)

		q_padded, q_padded_len = self.padding(q, group_size)

		num_groups_q = q_padded.size(1) // group_size
		num_groups_k = k_padded.size(1) // group_size
		q_padded = q_padded.view(H, num_groups_q, group_size, D)
		k_padded = k_padded.view(H, num_groups_k, group_size, D)
		representative_q = torch.zeros(H, num_groups_q, D, device=q.device, dtype=q.dtype)
		representative_k = torch.zeros(H, num_groups_k, D, device=k.device, dtype=k.dtype)
		if self.threshold_max:
			representative_q = q_padded.max(dim=2).values  # (H, num_groups_q, D)
			representative_k = k_padded.max(dim=2).values  # (H, num_groups_k, D)
		else:
			representative_q = q_padded.mean(dim=2)  # (H, num_groups_q, D)
			representative_k = k_padded.mean(dim=2)  # (H, num_groups_k, D)
		
		assert(representative_q.size(0) == H and representative_q.size(1) == num_groups_q and representative_q.size(2) == D)
		assert(representative_k.size(0) == H and representative_k.size(1) == num_groups_k and representative_k.size(2) == D)
		representative_attn_scores = torch.bmm(representative_q, representative_k.transpose(-2, -1) / math.sqrt(D))  # (H, num_groups_q, num_groups_k)
		representative_attn_scores = torch.softmax(representative_attn_scores, dim=-1)
		block_mask = representative_attn_scores < self.threshold

		full_mask = torch.repeat_interleave(block_mask, repeats=group_size, dim=1)
		full_mask = torch.repeat_interleave(full_mask, repeats=group_size, dim=2)
		# print(f"sparsity {torch.mean(full_mask.float())}")
		final_attn_mask = full_mask

		vae_mask = final_attn_mask[:, 0:L, vae_padded_range[0]:vae_padded_range[1]]
		vit_mask = final_attn_mask[:, 0:L, vit_padded_range[0]:vit_padded_range[1]]
		vae_vit_mask = torch.cat([vae_mask, vit_mask], dim=2)
		assert(vae_vit_mask.size(0) == H and vae_vit_mask.size(1) == L and vae_vit_mask.size(2) == (vae_range[1] - vae_range[0] + vit_range[1] - vit_range[0]))
		return vae_vit_mask, representative_attn_scores

	def sparsify_kv_cache_threshold_self_fine(
		self,
		q: torch.Tensor,
		k: torch.Tensor,
		self_range: torch.Tensor,
	):
		H, L, D = q.shape
		_, N, _ = k.shape
		group_size = self.group_size

		pre_self_k = k[:, 0:self_range[0], :]
		self_k = k[:, self_range[0]:self_range[1], :]
		assert(self_range[1] == N), f"self_range end {self_range[1]} must equal to N {N}"

		pre_self_k_padded, pre_self_k_padded_len = self.padding(pre_self_k, group_size)
		self_k_padded, self_k_padded_len = self.padding(self_k, group_size)

		k_padded = torch.cat([pre_self_k_padded, self_k_padded], dim=1)
		pre_self_padded_range = torch.tensor((
			0, pre_self_k_padded.size(1)
		), device=k.device, dtype=torch.long)
		self_padded_range = torch.tensor((
			pre_self_k_padded.size(1),
			pre_self_k_padded.size(1) + self_k_padded.size(1) - self_k_padded_len
		), device=k.device, dtype=torch.long)

		q_padded, q_padded_len = self.padding(q, group_size)
		num_groups_q = q_padded.size(1) // group_size
		num_groups_k = k_padded.size(1) // group_size
		q_padded = q_padded.view(H, num_groups_q, group_size, D)
		k_padded = k_padded.view(H, num_groups_k, group_size, D)
		representative_q = torch.zeros(H, num_groups_q, D, device=q.device, dtype=q.dtype)
		representative_k = torch.zeros(H, num_groups_k, D, device=k.device, dtype=k.dtype)
		if self.threshold_max:
			representative_q = q_padded.max(dim=2).values  # (H, num_groups_q, D)
			representative_k = k_padded.max(dim=2).values  # (H, num_groups_k, D)
		else:
			representative_q = q_padded.mean(dim=2)  # (H, num_groups_q, D)
			representative_k = k_padded.mean(dim=2)  # (H, num_groups_k, D)
		assert(representative_q.size(0) == H and representative_q.size(1) == num_groups_q and representative_q.size(2) == D)
		assert(representative_k.size(0) == H and representative_k.size(1) == num_groups_k and representative_k.size(2) == D)
		representative_attn_scores = torch.bmm(representative_q, representative_k.transpose(-2, -1) / math.sqrt(D))  # (H, num_groups_q, num_groups_k)
		representative_attn_scores = torch.softmax(representative_attn_scores, dim=-1)
		block_mask = representative_attn_scores < self.threshold
		full_mask = torch.repeat_interleave(block_mask, repeats=group_size, dim=1)
		full_mask = torch.repeat_interleave(full_mask, repeats=group_size, dim=2)
		final_attn_mask = full_mask
		
		self_mask = final_attn_mask[:, 0:L, self_padded_range[0]:self_padded_range[1]]
		assert(self_mask.size(0) == H and self_mask.size(1) == L and self_mask.size(2) == (self_range[1] - self_range[0]))
		return self_mask, representative_attn_scores

	def sparsify_kv_cache_threshold_vae_vit_self_fine(
		self,
		q: torch.Tensor,
		k: torch.Tensor,
		vae_range: torch.Tensor,
		vit_range: torch.Tensor,
		self_range: torch.Tensor
	):
		H, L, D = q.shape
		_, N, _ = k.shape
		group_size = self.group_size

		# pre_vit_range = torch.tensor((0, vit_range[0]), device=k.device, dtype=torch.long)
		pre_vae_vit_range = torch.tensor((0, vae_range[0]), device=k.device, dtype=torch.long)
		middle_vit_self_range = torch.tensor((vit_range[1], self_range[0]), device=k.device, dtype=torch.long)

		pre_vae_vit_k = k[:, pre_vae_vit_range[0]:pre_vae_vit_range[1], :]
		vae_k = k[:, vae_range[0]:vae_range[1], :]
		vit_k = k[:, vit_range[0]:vit_range[1], :]
		middle_vit_self_k = k[:, middle_vit_self_range[0]:middle_vit_self_range[1], :]
		self_k = k[:, self_range[0]:self_range[1], :]
		assert(self_range[1] == N)

		pre_vae_vit_k_padded, pre_vae_vit_k_padded_len = self.padding(pre_vae_vit_k, group_size)
		vae_k_padded, vae_k_padded_len = self.padding(vae_k, group_size)
		vit_k_padded, vit_k_padded_len = self.padding(vit_k, group_size)
		middle_vit_self_k_padded, middle_vit_self_k_padded_len = self.padding(middle_vit_self_k, group_size)
		self_k_padded, self_k_padded_len = self.padding(self_k, group_size)

		k_padded = torch.cat([pre_vae_vit_k_padded, vae_k_padded, vit_k_padded, middle_vit_self_k_padded, self_k_padded], dim=1)
		pre_vae_vit_padded_range = torch.tensor((
			0, pre_vae_vit_k_padded.size(1)
		), device=k.device, dtype=torch.long)
		vae_padded_range = torch.tensor((
			pre_vae_vit_k_padded.size(1),
			pre_vae_vit_k_padded.size(1) + vae_k_padded.size(1) - vae_k_padded_len
		), device=k.device, dtype=torch.long)
		vit_padded_range = torch.tensor((
			pre_vae_vit_k_padded.size(1),
			pre_vae_vit_k_padded.size(1) + vit_k_padded.size(1) - vit_k_padded_len
		), device=k.device, dtype=torch.long)
		middle_vit_self_padded_range = torch.tensor((
			pre_vae_vit_k_padded.size(1) + vit_k_padded.size(1),
			pre_vae_vit_k_padded.size(1) + vit_k_padded.size(1) + middle_vit_self_k_padded.size(1) - middle_vit_self_k_padded_len
		), device=k.device, dtype=torch.long)
		self_padded_range = torch.tensor((
			pre_vae_vit_k_padded.size(1) + vit_k_padded.size(1) + middle_vit_self_k_padded.size(1),
			pre_vae_vit_k_padded.size(1) + vit_k_padded.size(1) + middle_vit_self_k_padded.size(1) + self_k_padded.size(1) - self_k_padded_len
		), device=k.device, dtype=torch.long)

		q_padded, q_padded_len = self.padding(q, group_size)

		num_groups_q = q_padded.size(1) // group_size
		num_groups_k = k_padded.size(1) // group_size
		q_padded = q_padded.view(H, num_groups_q, group_size, D)
		k_padded = k_padded.view(H, num_groups_k, group_size, D)
		representative_q = torch.zeros(H, num_groups_q, D, device=q.device, dtype=q.dtype)
		representative_k = torch.zeros(H, num_groups_k, D, device=k.device, dtype=k.dtype)
		if self.threshold_max:
			representative_q = q_padded.max(dim=2).values  # (H, num_groups_q, D)
			representative_k = k_padded.max(dim=2).values  # (H, num_groups_k, D)
		else:
			representative_q = q_padded.mean(dim=2)  # (H, num_groups_q, D)
			representative_k = k_padded.mean(dim=2)  # (H, num_groups_k, D)
		
		assert(representative_q.size(0) == H and representative_q.size(1) == num_groups_q and representative_q.size(2) == D)
		assert(representative_k.size(0) == H and representative_k.size(1) == num_groups_k and representative_k.size(2) == D)
		representative_attn_scores = torch.bmm(representative_q, representative_k.transpose(-2, -1) / math.sqrt(D))  # (H, num_groups_q, num_groups_k)
		representative_attn_scores = torch.softmax(representative_attn_scores, dim=-1)
		block_mask = representative_attn_scores < self.threshold
		full_mask = torch.repeat_interleave(block_mask, repeats=group_size, dim=1)
		full_mask = torch.repeat_interleave(full_mask, repeats=group_size, dim=2)
		# print(f"sparsity {torch.mean(full_mask.float())}")
		final_attn_mask = full_mask

		vae_mask = final_attn_mask[:, 0:L, vae_padded_range[0]:vae_padded_range[1]]
		vit_mask = final_attn_mask[:, 0:L, vit_padded_range[0]:vit_padded_range[1]]
		self_mask = final_attn_mask[:, 0:L, self_padded_range[0]:self_padded_range[1]]

		assert(vae_mask.size(0) == H and vae_mask.size(1) == L and vae_mask.size(2) == (vae_range[1] - vae_range[0]))
		assert(vit_mask.size(0) == H and vit_mask.size(1) == L and vit_mask.size(2) == (vit_range[1] - vit_range[0]))
		assert(self_mask.size(0) == H and self_mask.size(1) == L and self_mask.size(2) == (self_range[1] - self_range[0]))

		return vae_mask, vit_mask, self_mask, representative_attn_scores
	
	def sparsify_kv_cache_topk(self, q: torch.Tensor, k: torch.Tensor, q_range: torch.Tensor, k_range: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			q: (H, L, D)
			k: (H, N, D)
		"""
		# print("sparsify_kv_cache called")
		q = q[:, q_range[0]:q_range[1], :]
		k = k[:, k_range[0]:k_range[1], :]
		if self.top_k == 0:
			return torch.zeros(q.size(0), q.size(1), k.size(1), device=q.device, dtype=torch.bool)
		
		H, L, D = q.shape
		_, N, _ = k.shape
		group_size = self.group_size
		top_k = self.top_k

		L_NEW = L
		N_NEW = N

		num_groups_q = L // group_size
		padding_q = (group_size - (L % group_size)) % group_size
		if padding_q > 0:
			q = F.pad(q, (0, 0, 0, padding_q), value=float('-inf'))
			num_groups_q += 1
			L_NEW += padding_q
		num_groups_k = N // group_size
		padding_k = (group_size - (N % group_size)) % group_size
		if padding_k > 0:
			k = F.pad(k, (0, 0, 0, padding_k), value=float('-inf'))
			num_groups_k += 1
			N_NEW += padding_k
			# pad后，k的长度变了
		assert(q.size(1) == L_NEW and k.size(1) == N_NEW)
		q = q.view(H, num_groups_q, group_size, D)
		k = k.view(H, num_groups_k, group_size, D)
		# representative_q = q.max(dim=2).values  # (H, num_groups_q, D)
		# representative_k = k.max(dim=2).values  # (H, num_groups_k, D)
		representative_q = q.mean(dim=2)  # (H, num_groups_q, D)
		representative_k = k.mean(dim=2)  # (H, num_groups_k, D)
		assert(representative_q.size(0) == H and representative_q.size(1) == num_groups_q and representative_q.size(2) == D)
		assert(representative_k.size(0) == H and representative_k.size(1) == num_groups_k and representative_k.size(2) == D)
		representative_attn_scores = torch.matmul(representative_q, representative_k.transpose(-2, -1) / math.sqrt(D))  # (H, num_groups_q, num_groups_k)

		nr_maintain = max(1, int(num_groups_k * num_groups_q * top_k))
		representative_attn_scores = representative_attn_scores.view(H, -1)

		top_k_values, top_k_indices = torch.topk(representative_attn_scores, k=nr_maintain, dim=-1) # (H, nr_maintain)
		top_k_row_indices = top_k_indices // num_groups_k
		top_k_col_indices = top_k_indices % num_groups_k

		block_mask = torch.zeros(H, num_groups_q, num_groups_k, device=q.device, dtype=torch.bool)
		head_indices = torch.arange(H, device=q.device).view(H, 1).expand(-1, nr_maintain) # (H, nr_maintain)
		block_mask[head_indices, top_k_row_indices, top_k_col_indices] = True
		full_mask = torch.repeat_interleave(block_mask, repeats=group_size, dim=1)
		full_mask = torch.repeat_interleave(full_mask, repeats=group_size, dim=2)
		# print(f"sparsity {torch.mean(full_mask.float())}")
		final_attn_mask = full_mask
		# True 代表“屏蔽”
		final_attn_mask = final_attn_mask[:, :L, :N]
		assert(final_attn_mask.size(0) == H and final_attn_mask.size(1) == L and final_attn_mask.size(2) == N)
		return final_attn_mask

	def sparsify_self_attn(self, q: torch.Tensor, k: torch.Tensor, q_range: torch.Tensor, k_range: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			q: (B, H, L, D)
			k: (B, H, L, D)
		"""
		q = q[:, q_range[0]:q_range[1], :]
		k = k[:, k_range[0]:k_range[1], :]
		H, L, D = q.shape
		_, N, _ = k.shape
		return torch.ones(H, L, N, device=q.device, dtype=q.dtype)

class BlockQuantize:
	"""
	一个专门用于对 3D 权重/激活张量进行组量化的类。
	改编自 QueryGroupQuantizer，用于处理 (H, L, D) 形状的数据。
	支持 INT8, INT4, 和 FP4 模式。
	"""
	def __init__(self, group_size: int = 128, mode: str = 'int8'):
		self.group_size = group_size
		self.mode = mode

		if self.mode == 'int4':
			self.n_bits = 4
			if self.group_size % 2 != 0:
				raise ValueError("对于 INT4 模式, group_size 必须是 2 的倍数。")
		elif self.mode == 'int8':
			self.n_bits = 8
		elif self.mode == 'fp4':
			self.n_bits = 4
			# 使用 bfloat16 以匹配常见的模型数据类型，节省码本存储
			self.fp4_codebook = torch.tensor([
					-1.0, -0.6667, -0.5, -0.3333, -0.25, -0.1667, -0.0833, 0.0,
					0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.6667, 1.0
			], dtype=torch.bfloat16)
		else:
			raise ValueError(f"Unsupported mode: {mode}")

		if 'int' in self.mode:
			self.q_max = 2 ** (self.n_bits - 1) - 1
			self.q_min = -2 ** (self.n_bits - 1)

	def quantize(self, x: torch.Tensor) -> Tuple:
		if self.mode in ['int8', 'int4']:
			return self._quantize_int(x)
		elif self.mode == 'fp4':
			return self._quantize_fp4(x)
		else:
			raise ValueError(f"Unsupported mode: {self.mode}")

	def dequantize(self, *args, original_shape: Tuple) -> torch.Tensor:
		if self.mode == 'int8':
			return self._dequantize_int(*args, original_shape=original_shape)
		elif self.mode == 'int4':
			q_packed, scales, zero_points = args
			return self._dequantize_int4(q_packed, scales, zero_points, original_shape=original_shape)
		elif self.mode == 'fp4':
			q_indices, scales = args
			return self._dequantize_fp4(q_indices, scales, original_shape=original_shape)
		else:
			raise ValueError(f"Unsupported mode: {self.mode}")

	def _quantize_int(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""INT8 和 INT4 的核心非对称量化逻辑。"""
		assert x.dim() == 3, "输入必须是 3D 张量 (H, L, D)"
		H, L, D = x.shape
		if D % self.group_size != 0:
			raise ValueError(f"特征维度 ({D}) 必须能被 group_size ({self.group_size}) 整除。")

		grouped_x = x.view(H, L, -1, self.group_size)
		grouped_x_fp32 = grouped_x.float()
		
		min_vals = torch.min(grouped_x_fp32, dim=-1).values
		max_vals = torch.max(grouped_x_fp32, dim=-1).values

		scales = (max_vals - min_vals) / (self.q_max - self.q_min)
		scales = scales.clamp(min=1e-6)
		
		zero_points = torch.round(self.q_min - min_vals / scales).to(torch.int8)

		q_x = torch.round(grouped_x_fp32 / scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
		q_x = q_x.clamp(self.q_min, self.q_max).to(torch.int8)

		if self.mode == 'int4':
			q_x_shifted = q_x - self.q_min
			q_packed = q_x_shifted.view(H, L, -1, self.group_size // 2, 2)
			val1 = q_packed[..., 0]
			val2 = q_packed[..., 1]
			q_packed_byte = (val1 << 4) | val2
			return q_packed_byte.to(torch.uint8), scales, zero_points
		else: # INT8
			return q_x, scales, zero_points

	def _dequantize_int(self, q_x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""INT8 的反量化逻辑。"""
		dequantized_groups = (q_x.float() - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)
		return dequantized_groups.reshape(original_shape)

	def _dequantize_int4(self, q_packed: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""INT4 的反量化逻辑，包含解包。"""
		H, L, D = original_shape
		
		val1_shifted = q_packed >> 4
		val2_shifted = q_packed & 0x0F
		
		q_x_shifted = torch.stack([val1_shifted, val2_shifted], dim=-1).view(H, L, -1, self.group_size)
		
		q_x = q_x_shifted.to(torch.int8) + self.q_min

		return self._dequantize_int(q_x, scales, zero_points, original_shape)

	def _quantize_fp4(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""FP4 对称量化逻辑。"""
		assert x.dim() == 3, "输入必须是 3D 张量 (H, L, D)"
		H, L, D = x.shape
		if D % self.group_size != 0:
			raise ValueError(f"特征维度 ({D}) 必须能被 group_size ({self.group_size}) 整除。")

		grouped_x = x.view(H, L, -1, self.group_size)
		
		scales = grouped_x.abs().max(dim=-1, keepdim=True).values
		scales = scales.clamp(min=1e-6)

		normalized_x = grouped_x / scales

		codebook = self.fp4_codebook.to(x.device, dtype=x.dtype)
		abs_diff = torch.abs(normalized_x.unsqueeze(-1) - codebook)
		q_indices = torch.argmin(abs_diff, dim=-1).to(torch.int8)

		return q_indices, scales.squeeze(-1)

	def _dequantize_fp4(self, q_indices: torch.Tensor, scales: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""FP4 反量化逻辑。"""
		codebook = self.fp4_codebook.to(scales.device, dtype=scales.dtype)
		
		quantized_normalized = codebook[q_indices]
		
		dequantized_groups = quantized_normalized * scales.unsqueeze(-1)
		
		return dequantized_groups.reshape(original_shape)
	
	def simulate_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		执行量化和立即反量化的往返过程，以模拟精度损失。

		Args:
				x (torch.Tensor): 原始的浮点张量。

		Returns:
				Tuple[torch.Tensor, torch.Tensor]:
				- dequantized_x: 模拟量化后的浮点张量。
				- scales: 计算出的量化尺度，可用于分析。
		"""
		original_shape = x.shape
		
		quantized_data = self.quantize(x)
		
		dequantized_x = self.dequantize(*quantized_data, original_shape=original_shape)
		
		scales = quantized_data[1]
		
		return dequantized_x, scales

# class BlockQuantize:
# 	def __init__(self, group_size: int = 128):
# 		self.group_size = group_size
# 		# NVFP4 (E2M1) data type values.
# 		# These are derived from the 1-sign, 2-exponent, 1-mantissa format.
# 		# Positive values: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# 		self.nvfp4_values = torch.tensor([
# 				-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5,
# 					0.0,
# 					0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
# 		], dtype=torch.bfloat16)
# 		# We have 17 values here, but FP4 can only represent 16.
# 		# A common practice in simulation is to merge two values or drop one.
# 		# For this implementation, we will use a slightly simplified 15-value set
# 		# that is symmetric and includes zero, which is common for quantization simulation.
# 		self.fp4_codebook = torch.tensor([
# 				-1.0, -0.6667, -0.5, -0.3333, -0.25, -0.1667, -0.0833, 0.0,
# 				0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.6667, 1.0
# 		], dtype=torch.bfloat16)

# 	def quantize_int4(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
# 		"""
# 		Quantize the input tensor x in blocks that is unmasked by x_mask.

# 		Use Group Quantization along the D dimension (D // group_size). The input tensor x is stored in the FP16 format.
# 		This function should quantize the unmasked blocks of x to FP4 format which should be represented by
# 		a FP4 data tensor (fake format, actually constructed in a FP16 format) and a FP16 scale tensor.
# 		Args:
# 			x: (H, L, D)
# 			x_mask: (H, L) True means masked position
# 		"""
# 		# print("here in block INT4 quantize")
# 		H, L, D = x.shape
# 		assert D % self.group_size == 0, "D must be divisible by group_size"
# 		num_groups = D // self.group_size

# 		x_grouped = x.view(H, L, num_groups, self.group_size)  # (H, L, num_groups, group_size)
# 		scales = x_grouped.abs().max(dim=-1).values  # (H, L, num_groups)
# 		scales = scales + 1e-6  # avoid division by zero

# 		x_normalized = x_grouped / scales.unsqueeze(-1)  # (H, L, num_groups, group_size)

# 		fp4_levels = 7
# 		quantized_to_int = torch.round(x_normalized * fp4_levels)

# 		x_dequant_apx = (quantized_to_int / fp4_levels) * scales.unsqueeze(-1)  # (H, L, num_groups, group_size)

# 		assert x_dequant_apx.dtype == x.dtype
# 		return x_dequant_apx.view(H, L, D), scales

# 	def quantize_nvfp4(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
# 		"""
# 		Quantize the input tensor x in blocks that is unmasked by x_mask.

# 		Use Group Quantization along the D dimension (D // group_size). The input tensor x is stored in the FP16 format.
# 		This function should quantize the unmasked blocks of x to NVidia FP4 format which should be represented by
# 		a NVFP4 data tensor (fake format, actually constructed in a FP16 format) and a FP16 scale tensor.
# 		Args:
# 			x: (H, L, D)
# 			x_mask: (H, L) True means masked position
# 		"""
# 		H, L, D = x.shape
# 		assert D % self.group_size == 0, "D must be divisible by group_size"
# 		num_groups = D // self.group_size

# 		x_grouped = x.view(H, L, num_groups, self.group_size)

# 		scales = x_grouped.abs().max(dim=-1, keepdim=True).values
# 		scales = scales + 1e-6

# 		# 2. Normalize the data to [-1, 1]
# 		x_normalized = x_grouped / scales

# 		# 3. Find the closest value in our FP4 codebook for each normalized value
# 		codebook = self.fp4_codebook.to(x.device)
		
# 		x_expanded = x_normalized.unsqueeze(-1)
# 		codebook_expanded = codebook.view(1, 1, 1, 1, -1)

# 		abs_diff = torch.abs(x_expanded - codebook_expanded)
# 		closest_indices = torch.argmin(abs_diff, dim=-1)

# 		quantized_normalized = codebook[closest_indices]

# 		# 4. De-quantize by applying the scales back
# 		x_dequant_apx = quantized_normalized * scales

# 		# Reshape and handle mask
# 		x_dequant_apx = x_dequant_apx.view(H, L, D)

# 		if x_mask is not None:
# 				mask_expanded = x_mask.unsqueeze(-1).expand_as(x)
# 				final_x = torch.where(mask_expanded, x, x_dequant_apx)
# 		else:
# 				final_x = x_dequant_apx

# 		assert final_x.dtype == x.dtype
# 		return final_x, scales.squeeze(-1)
	
# 	@staticmethod
# 	def eval_snr(x: torch.Tensor, x_dequant: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> float:
# 		"""
# 		Evaluate the SNR (Signal-to-Noise Ratio) between the original tensor x and the dequantized tensor x_dequant.
# 		Args:
# 			x: (H, L, D)
# 			x_dequant: (H, L, D)
# 			x_mask: (H, L) True means masked position
# 		"""
# 		if x_mask is not None:
# 			x = x.masked_fill(x_mask, 0)
# 			x_dequant = x_dequant.masked_fill(x_mask, 0)

# 		signal_power = (x ** 2).mean()
# 		reconstruction_error = (x - x_dequant) ** 2
# 		noise_power = reconstruction_error.mean()

# 		# Convert tensor scalars to Python floats to satisfy the declared return type
# 		noise_power_val = float(noise_power.item()) if isinstance(noise_power, torch.Tensor) and noise_power.numel() == 1 else float(noise_power)
# 		if noise_power_val <= 0.0:
# 			return float("inf")
# 		signal_power_val = float(signal_power.item()) if isinstance(signal_power, torch.Tensor) and signal_power.numel() == 1 else float(signal_power)
# 		snr_val = 10.0 * math.log10(signal_power_val / noise_power_val)
# 		return snr_val
	
# 	def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None, mode: str = 'nvfp4') -> Tuple[torch.Tensor, torch.Tensor]:
# 		if mode == 'nvfp4':
# 			return self.quantize_nvfp4(x, x_mask)
# 		elif mode == 'int4':
# 			return self.quantize_int4(x, x_mask)
# 		else:
# 			raise ValueError(f"Unsupported quantization mode: {mode}")


if __name__ == "__main__":
	# simple test
	block_quantizer = BlockQuantize(group_size=32)
	# x = torch.randn(2, 8, 128).to(torch.float16)
	x = torch.randn(2, 8, 128).to(torch.bfloat16)
	x_dequant, scales = block_quantizer.quantize_nvfp4(x)
	# x_dequant, scales = block_quantizer.quantize_int4(x)
	snr = BlockQuantize.eval_snr(x, x_dequant)
	print(f"SNR: {snr} dB")
	print("Original x:", x)
	print("Dequantized x:", x_dequant)
	print("Scales:", scales)