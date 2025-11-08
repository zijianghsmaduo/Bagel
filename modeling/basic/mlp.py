import torch
from typing import List, Optional, Tuple, cast
from modeling.qwen2.modeling_qwen2 import (
		Qwen2Attention, 
		Qwen2MLP, 
		Qwen2PreTrainedModel, 
		Qwen2RMSNorm, 
		Qwen2RotaryEmbedding,
		apply_rotary_pos_emb,
)

import numpy as np

def calculate_and_print_error(
		tensor_a: torch.Tensor, 
		tensor_b: torch.Tensor, 
		implementation_name: str,
		rtol: float = 1e-2,
		atol: float = 1e-3
):
		"""
		计算两个张量之间的误差，并打印详细的统计信息。
		同时，它会执行 allclose 断言。
		"""
		# 确保比较在 float32 下进行，以获得准确的误差度量
		a_fp32 = tensor_a.to(torch.float32)
		b_fp32 = tensor_b.to(torch.float32)

		# 1. 绝对误差
		abs_error = torch.abs(a_fp32 - b_fp32)
		max_abs_error = torch.max(abs_error).item()
		mean_abs_error = torch.mean(abs_error).item()

		# 2. 相对误差 (添加一个小的 epsilon 以避免除以零)
		# 我们只在 b_fp32 不接近于零的地方计算相对误差
		denominator = torch.abs(b_fp32)
		# 将分母中非常小的值替换为1，以避免产生巨大的相对误差
		denominator[denominator < 1e-5] = 1 
		rel_error = abs_error / denominator
		max_rel_error = torch.max(rel_error).item()
		
		print(f"--- Error Report for: {implementation_name} ---")
		print(f"  - Max Absolute Error: {max_abs_error:.6f}")
		print(f"  - Mean Absolute Error: {mean_abs_error:.6f}")
		print(f"  - Max Relative Error: {max_rel_error:.6%}") # 以百分比形式打印
		
		# 3. 执行断言
		is_close = torch.allclose(a_fp32, b_fp32, rtol=rtol, atol=atol)
		if not is_close:
				print("  - !!! torch.allclose FAILED !!!")
		else:
				print("  - torch.allclose PASSED")
		print("-" * (26 + len(implementation_name)))
		
		assert is_close, f"{implementation_name} computation mismatch."

class TilingUpLinear:
	def __init__(self):
		pass

	def forward(self, act: torch.Tensor, weight: torch.Tensor, nr_head: int) -> torch.Tensor:
		N, D = act.shape
		H = nr_head
		d = D // H
		I, _ = weight.shape
		original_dtype = act.dtype

		act_reshaped = act.view(N, H, d)
		weight_reshaped = weight.T.reshape(H, d, I)

		act_bmm = act_reshaped.permute(1, 0, 2)
		weight_bmm = weight_reshaped
		
		tiled_results = torch.bmm(act_bmm, weight_bmm)
		assert tiled_results.shape == (H, N, I), "TilingUpLinear bmm result has incorrect shape."
		
		return tiled_results.to(original_dtype)

class TilingDownLinear:
	def __init__(self):
		pass

	def forward(self, act: torch.Tensor, weight: torch.Tensor, nr_head: int) -> Tuple[torch.Tensor, torch.Tensor]:
		N, I = act.shape
		D, _ = weight.shape
		H = nr_head
		d = D // H
		original_dtype = act.dtype

		weight_t_chunks = torch.chunk(weight.T, chunks=H, dim=1)

		tiled_results = []
		for chunk in weight_t_chunks:
				tiled_results.append(act @ chunk)
		
		result_concat = torch.cat(tiled_results, dim=1)
		assert result_concat.shape == (N, D), "TilingDownLinear final result has incorrect shape."

		tiled_results_tensor = torch.stack(tiled_results, dim=0)

		return result_concat.to(original_dtype), tiled_results_tensor.to(original_dtype)


class ReuseMLP(Qwen2MLP):
	def __init__(self, config):
		super().__init__(config)
		self.tiling_up_proj = TilingUpLinear()
		self.tiling_gate_proj = TilingUpLinear()
		self.tiling_down_proj = TilingDownLinear()

		self.normal_act_cache = None
		self.up_proj_cache = None
		self.gate_proj_cache = None
		self.gate_act_cache = None
		self.nr_head = config.num_attention_heads

		self.cos_threshold = 0.99
		self.rtol = 1e-2
		self.atol = 1e-3

	def compute_cosine_similarity(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
		assert tensor_a.shape == tensor_b.shape, "Input tensors must have the same shape."
		N, D = tensor_a.shape
		H = self.nr_head
		d = D // H
		tensor_a = tensor_a.to(torch.float32)
		tensor_b = tensor_b.to(torch.float32)

		tensor_a_reshaped = tensor_a.view(N, H, d).permute(1, 0, 2)  # (H, N, d)
		tensor_b_reshaped = tensor_b.view(N, H, d).permute(1, 0, 2)  # (H, N, d)

		numerator = (tensor_a_reshaped * tensor_b_reshaped).sum(dim=-1)  # (H, N)
		denominator = torch.norm(tensor_a_reshaped, dim=-1) * torch.norm(tensor_b_reshaped, dim=-1)  # (H, N)
		denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=denominator.device), denominator)
		cosine_similarity = numerator / denominator

		return cosine_similarity

	def gen_similarity_mask(self, hidden_state: torch.Tensor):
		if self.normal_act_cache is None:
			raise ValueError("Normal activation cache is empty.")
		cosine_sim = self.compute_cosine_similarity(hidden_state, self.normal_act_cache)  # (H, N)
		similarity_mask = cosine_sim > self.cos_threshold  # (H, N)
		return similarity_mask

	def forward(self, hidden_state: torch.Tensor, *, sparsity: Optional[np.ndarray] = None, cfg_type: Optional[str] = None, layer_idx: Optional[int] = None, timestep: Optional[int] = None) -> torch.Tensor:
		if cfg_type is None:
			return super().forward(hidden_state)
		elif cfg_type == "normal":
			self.normal_act_cache = hidden_state

			# --- Gate & Up Projections ---
			gated_tiling_o = self.tiling_gate_proj.forward(hidden_state, self.gate_proj.weight, self.nr_head)
			gated_o = torch.sum(gated_tiling_o, dim=0).to(dtype=hidden_state.dtype)

			gated_act_o = self.act_fn(gated_o)
			
			up_tiling_o = self.tiling_up_proj.forward(hidden_state, self.up_proj.weight, self.nr_head)
			up_o = torch.sum(up_tiling_o, dim=0).to(dtype=hidden_state.dtype)

			# calculate_and_print_error(
			#         gated_o, self.gate_proj(hidden_state), "Gate Projection", self.rtol, self.atol
			# )
			# calculate_and_print_error(
			#         up_o, self.up_proj(hidden_state), "Up Projection", self.rtol, self.atol
			# )

			gated_value_standard = (gated_act_o * up_o).to(dtype=hidden_state.dtype)
			
			down_o, _ = self.tiling_down_proj.forward(gated_value_standard, self.down_proj.weight, self.nr_head)

			return down_o
		elif cfg_type == "cfg_text" or cfg_type == "cfg_img":
			assert self.normal_act_cache is not None, "Normal activation cache is empty."
			assert layer_idx is not None, "Layer index must be provided for CFG mode."
			assert timestep is not None, "Timestep must be provided for CFG mode."
			assert sparsity is not None, "Sparsity array must be provided for CFG mode."

			## True means replace with cache
			similarity_mask = self.gen_similarity_mask(hidden_state)  # (H, N)
			sparsity[cfg_type][timestep, layer_idx] = similarity_mask.float().mean().item()

			gated_proj_cache_gpu = self.tiling_gate_proj.forward(self.normal_act_cache, self.gate_proj.weight, self.nr_head)
			gated_tiling_o = self.tiling_gate_proj.forward(hidden_state, self.gate_proj.weight, self.nr_head)
			gated_tiling_o = torch.where(
					similarity_mask.unsqueeze(-1),
					gated_proj_cache_gpu,
					gated_tiling_o
			)
			gated_o = torch.sum(gated_tiling_o, dim=0)
			gated_act_o = self.act_fn(gated_o)

			up_proj_cache_gpu = self.tiling_up_proj.forward(self.normal_act_cache, self.up_proj.weight, self.nr_head)
			up_tiling_o = self.tiling_up_proj.forward(hidden_state, self.up_proj.weight, self.nr_head)
			up_tiling_o = torch.where(
					similarity_mask.unsqueeze(-1),
					up_proj_cache_gpu,
					up_tiling_o
			)
			up_o = torch.sum(up_tiling_o, dim=0)

			gated_value_similarity = (gated_act_o * up_o).to(dtype=hidden_state.dtype)

			down_o, _ = self.tiling_down_proj.forward(gated_value_similarity, self.down_proj.weight, self.nr_head)

			return down_o
		else:
			raise ValueError(f"Unsupported cfg_type: {cfg_type}")