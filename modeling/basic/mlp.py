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

from modeling.basic.sparse import BlockQuantize

import numpy as np

class WeightGroupQuantizer:
	"""
	一个专门用于对 2D 权重矩阵进行组量化的类。
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
			self.fp4_codebook = torch.tensor([
				-1.0, -0.6667, -0.5, -0.3333, -0.25, -0.1667, -0.0833, 0.0,
				0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.6667, 1.0
			], dtype=torch.float16) # 使用 float16 以节省码本存储
		else:
			raise ValueError(f"Unsupported mode: {mode}")

		if 'int' in self.mode:
			self.q_max = 2 ** (self.n_bits - 1) - 1
			self.q_min = -2 ** (self.n_bits - 1)

	def quantize(self, weight: torch.Tensor) -> Tuple:
		if self.mode == 'int8':
			return self._quantize_int(weight)
		elif self.mode == 'int4':
			return self._quantize_int(weight)
		elif self.mode == 'fp4':
			return self._quantize_fp4(weight)
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

	def _quantize_int(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""INT8 和 INT4 的核心非对称量化逻辑。"""
		assert weight.dim() == 2, "权重必须是 2D 张量"
		out_features, in_features = weight.shape
		if in_features % self.group_size != 0:
			raise ValueError(f"输入特征维度 ({in_features}) 必须能被 group_size ({self.group_size}) 整除。")

		grouped_weight = weight.view(out_features, -1, self.group_size)
		grouped_weight_fp32 = grouped_weight.float()
		
		min_vals, _ = torch.min(grouped_weight_fp32, dim=-1)
		max_vals, _ = torch.max(grouped_weight_fp32, dim=-1)

		scales = (max_vals - min_vals) / (self.q_max - self.q_min)
		scales = scales.clamp(min=1e-6)
		
		zero_points = torch.round(self.q_min - min_vals / scales).to(torch.int8)

		q_weight = torch.round(grouped_weight_fp32 / scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
		q_weight = q_weight.clamp(self.q_min, self.q_max).to(torch.int8)

		if self.mode == 'int4':
			# --- INT4 打包逻辑 ---
			# 将范围 [-8, 7] 映射到 [0, 15]
			q_weight_shifted = q_weight - self.q_min
			# 将每两个 int4 值打包成一个 int8
			q_packed = q_weight_shifted.view(out_features, -1, self.group_size // 2, 2)
			val1 = q_packed[..., 0]
			val2 = q_packed[..., 1]
			# val1 存高4位, val2 存低4位
			q_packed_byte = (val1 << 4) | val2
			return q_packed_byte.to(torch.uint8), scales, zero_points
		else: # INT8
			# print("INT8 quantization completed.")
			return q_weight, scales, zero_points

	def _dequantize_int(self, q_weight: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""INT8 的反量化逻辑。"""
		dequantized_groups = (q_weight.float() - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)
		return dequantized_groups.reshape(original_shape)

	def _dequantize_int4(self, q_packed: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""INT4 的反量化逻辑，包含解包。"""
		out_features, _ = original_shape
		
		# --- INT4 解包逻辑 ---
		# 从 uint8 解包回两个 int4 值
		val1_shifted = q_packed >> 4
		val2_shifted = q_packed & 0x0F # 掩码，只取低4位
		
		# 组合回原始的 int8 张量形状
		q_weight_shifted = torch.stack([val1_shifted, val2_shifted], dim=-1).view(out_features, -1, self.group_size)
		
		# 从 [0, 15] 映射回 [-8, 7]
		q_weight = q_weight_shifted.to(torch.int8) + self.q_min

		# 使用通用的反量化公式
		return self._dequantize_int(q_weight, scales, zero_points, original_shape)

	def _quantize_fp4(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""FP4 对称量化逻辑。"""
		assert weight.dim() == 2, "权重必须是 2D 张量"
		out_features, in_features = weight.shape
		if in_features % self.group_size != 0:
			raise ValueError(f"输入特征维度 ({in_features}) 必须能被 group_size ({self.group_size}) 整除。")

		grouped_weight = weight.view(out_features, -1, self.group_size)
		
		# 1. 对称量化，只计算 scale
		scales = grouped_weight.abs().max(dim=-1, keepdim=True).values
		scales = scales.clamp(min=1e-6)

		# 2. 归一化到 [-1, 1]
		normalized_weight = grouped_weight / scales

		# 3. 找到码本中最近的值的索引
		codebook = self.fp4_codebook.to(weight.device, dtype=weight.dtype)
		# 扩展维度以进行广播和距离计算
		# normalized_weight: (O, G_num, G_size, 1)
		# codebook:          (1, 1,     1,      C_size)
		abs_diff = torch.abs(normalized_weight.unsqueeze(-1) - codebook)
		q_indices = torch.argmin(abs_diff, dim=-1).to(torch.int8)

		return q_indices, scales.squeeze(-1)

	def _dequantize_fp4(self, q_indices: torch.Tensor, scales: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""FP4 反量化逻辑。"""
		codebook = self.fp4_codebook.to(scales.device, dtype=scales.dtype)
		
		# 1. 使用索引从码本中恢复归一化的值
		quantized_normalized = codebook[q_indices]
		
		# 2. 乘以 scale 恢复浮点值
		dequantized_groups = quantized_normalized * scales.unsqueeze(-1)
		
		return dequantized_groups.reshape(original_shape)
	
	def simulate_quantization(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		执行量化和立即反量化的往返过程，以模拟精度损失。

		Args:
				weight (torch.Tensor): 原始的浮点权重张量。

		Returns:
				Tuple[torch.Tensor, torch.Tensor]:
				- dequantized_weight: 模拟量化后的浮点权重张量。
				- scales: 计算出的量化尺度，可用于分析。
		"""
		original_shape = weight.shape
		
		# 1. 执行“真实”量化
		quantized_data = self.quantize(weight)
		
		# 2. 立即执行反量化
		dequantized_weight = self.dequantize(*quantized_data, original_shape=original_shape)
		
		# 提取 scales 用于返回
		# scales 通常是元组中的第二个元素
		scales = quantized_data[1]
		
		return dequantized_weight, scales

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
	def __init__(self, config, use_quantized_w: bool = False, use_similarity: bool = False):
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
		
		self.use_quantized_w = use_quantized_w
		self.use_similarity = use_similarity

		# self.quantizer = BlockQuantize(group_size=16)
		self.quant_type = "int4"
		self.quantizer = WeightGroupQuantizer(group_size=32, mode=self.quant_type)

		self.gate_proj_q = None
		self.gate_proj_s = None
		self.up_proj_q = None
		self.up_proj_s = None
		self.down_proj_q = None
		self.down_proj_s = None
		
	@staticmethod
	def quantize_up_weight(weight: torch.Tensor, quantizer: BlockQuantize, nr_head: int, quant_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
		I, D = weight.shape
		H = nr_head
		d = D // H
		weight_reshaped = weight.view(I, H, d)
		weight_reshaped = weight_reshaped.permute(1, 0, 2).contiguous()  # (H, I, d)
		quant, scale = quantizer.forward(weight_reshaped, mode=quant_type)
		# quant = weight_reshaped
		# scale = torch.ones_like(quant)
		quant = quant.permute(1, 0, 2).contiguous()
		return quant.reshape(I, D), scale

	@staticmethod
	def quantize_down_weight(weight: torch.Tensor, quantizer: BlockQuantize, nr_head: int, quant_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
		D, I = weight.shape
		H = nr_head
		d = D // H
		weight_reshaped = weight.view(H, d, I).contiguous()  # (H, d, I)
		quant, scale = quantizer.forward(weight_reshaped, mode=quant_type)
		# quant = weight_reshaped
		# scale = torch.ones_like(quant)
		return quant.reshape(D, I), scale

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

	def forward(self, hidden_state: torch.Tensor, *, use_quantized_w: bool = False, sparsity: Optional[np.ndarray] = None, cfg_type: Optional[str] = None, layer_idx: Optional[int] = None, timestep: Optional[int] = None) -> torch.Tensor:
		if cfg_type is None or self.use_similarity is False:
			if use_quantized_w and self.use_quantized_w:
				if self.gate_proj_q is None:
					self.gate_proj_q, self.gate_proj_s = self.quantizer.simulate_quantization(self.gate_proj.weight.data)
					self.up_proj_q, self.up_proj_s = self.quantizer.simulate_quantization(self.up_proj.weight.data)
					self.down_proj_q, self.down_proj_s = self.quantizer.simulate_quantization(self.down_proj.weight.data)
					# self.gate_proj_q, self.gate_proj_s = self.quantize_up_weight(self.gate_proj.weight.data, self.quantizer, self.nr_head, self.quant_type)
					# self.up_proj_q, self.up_proj_s = self.quantize_up_weight(self.up_proj.weight.data, self.quantizer, self.nr_head, self.quant_type)
					# self.down_proj_q, self.down_proj_s = self.quantize_down_weight(self.down_proj.weight.data, self.quantizer, self.nr_head, self.quant_type)

					# self.gate_proj_q = self.gate_proj.weight
					# self.up_proj_q = self.up_proj.weight
					# self.down_proj_q = self.down_proj.weight
				assert self.gate_proj_q is not None, "Quantized gate projection weights are not initialized."
				assert self.up_proj_q is not None, "Quantized up projection weights are not initialized."
				assert self.down_proj_q is not None, "Quantized down projection weights are not initialized."

				gated_o = torch.nn.functional.linear(hidden_state, self.gate_proj_q)
				up_o = torch.nn.functional.linear(hidden_state, self.up_proj_q)
				gated_value = self.act_fn(gated_o) * up_o
				return torch.nn.functional.linear(gated_value, self.down_proj_q)
			else:
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