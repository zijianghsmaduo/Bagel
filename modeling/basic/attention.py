import torch
from typing import Optional
import os

from modeling.basic.flash_attn import quant_attn_score_triton
from modeling.basic.sparse import (
	BlockSparsify, BlockQuantize
)
from modeling.basic import KVCacheStructure
from torch.nn import functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel

import math

import numpy as np

from modeling.basic.plot import plot_mask_heads
from modeling.basic.util import compute_coverage



class TrickAttention:
  def __init__(
    self, attention_backend: str = "naive", 
    sparse_gsize: int = 10, sparse_topk: float = 0.2, sparse_threshold: float = 4e-5,
    quant_gsize: int = 32,
    posterior_truncate_threshold: float = 4e-5, save_dir: str = "attn_probs_qkv_dump_new",
    is_save: bool = False, is_plot: bool = False, is_truncate: bool = False,
    plot_dir: str = "plot/sparse_attention_scores", heads_to_plot: Optional[list] = None,
    vae_vit: bool = True, self_attn: bool = False
  ):
    print("Initializing TrickAttention with settings: ")
    print(f"  attention_backend: {attention_backend}")
    print(f"  sparse_gsize: {sparse_gsize}")
    print(f"  sparse_topk: {sparse_topk}")
    print(f"  sparse_threshold: {sparse_threshold}")
    print(f"  quant_gsize: {quant_gsize}")
    print(f"  posterior_truncate_threshold: {posterior_truncate_threshold}")
    print(f"  is_save: {is_save}")
    print(f"  save_dir: {save_dir}")

    self.block_sparsifier = BlockSparsify(group_size=sparse_gsize, top_k=sparse_topk, threshold=sparse_threshold)
    self.block_quantizer = BlockQuantize(group_size=quant_gsize)
    
    self.posterior_truncate_threshold = posterior_truncate_threshold
    self.save_dir = save_dir
    self.is_save = is_save
    self.is_plot = is_plot
    self.plot_dir = plot_dir
    self.is_truncate = is_truncate

    self.heads_to_plot = heads_to_plot

    self.attention_backend = attention_backend

    self.sparsity = np.zeros((2, 49, 28), dtype=np.float32)

    self.vae_vit = vae_vit
    self.self_attn = self_attn

  def get_sparsity(self):
    return self.sparsity

  @staticmethod
  def get_vae_vit_range(
      kv_cache: KVCacheStructure,
  ):
      vit_range = kv_cache.vit
      vae_range = kv_cache.vae
      vae_vit_range = None
      if vit_range is not None and vae_range is not None:
        vae_vit_range = (vae_range[0], vit_range[1]+1)
      elif vit_range is None and vae_range is None:
        vae_vit_range = None
      elif vit_range is not None:
        vae_vit_range = (vit_range[0], vit_range[1]+1)
      elif vae_range is not None:
        vae_vit_range = (vae_range[0], vae_range[1]+1)
      return vae_vit_range

  @staticmethod
  def get_self_range(
     kv_cache: KVCacheStructure,
  ):
    self_range = kv_cache.gen_image
    return (self_range[0], self_range[1]+1) if self_range is not None else None
  
  @staticmethod
  def get_vae_range(
    kv_cache: KVCacheStructure,
  ):
    vae_range = kv_cache.vae
    return (vae_range[0], vae_range[1]+1) if vae_range is not None else None

  def save(
    self, entry_to_save: dict, mode: str, batch_idx: int,
    timestep: Optional[int] = None, layer_idx: Optional[int] = None, 
    cfg_type: Optional[str] = None
  ):
    should_save = layer_idx is not None and self.is_save and timestep is not None and timestep >= 0
    if not should_save:
      return
    assert (mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None) or (mode == "und" and cfg_type is None), "Invalid save conditions."
    
    filename = f"{mode}_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_{batch_idx}.pt" if mode == "und" else f"{mode}_qkv_attn_probs_{cfg_type}_layer_{layer_idx}_ts_{timestep}_batch_{batch_idx}.pt"
    os.makedirs(self.save_dir, exist_ok=True)
    save_path = os.path.join(self.save_dir, filename)
    if not os.path.exists(save_path):
      torch.save([entry_to_save], save_path)

  def naive_varlen_attention(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
      # scale = packed_query_states.size(-1) ** -0.5

      num_q_heads = packed_query_states.size(1)
      num_kv_heads = merged_key_states.size(1)
      if num_q_heads != num_kv_heads:
          if num_q_heads % num_kv_heads != 0:
              raise ValueError(
                  f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
              )
          group_size = num_q_heads // num_kv_heads
          merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
          merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)


      for b in range(B):
          q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
          k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

          q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
          k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
          v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

          q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
          k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
          v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

          # attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2)) * scale # (n_heads, Lq, Lk)
          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

          if causal:
              Lq, Lk = attn_scores.size(1), attn_scores.size(2)
              if Lq > 1:
                  q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
                  k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                  
                  causal_mask_shift = Lk - Lq
                  mask = k_indices > (q_indices + causal_mask_shift)
                  
                  attn_scores.masked_fill_(mask, float("-inf"))

          attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)

          should_save = layer_idx is not None and self.is_save and timestep is not None and timestep >= 0
          if should_save:
            entry_to_save = {
                "q": q_bmm.cpu(),
                # "k": k_bmm.cpu(),
                # "v": v_bmm.cpu(),
                # "attn_probs": attn_probs.cpu(),
                # "data": attn_probs.cpu()
            }
            self.save(
              entry_to_save=entry_to_save, mode=mode, 
              timestep=timestep, layer_idx=layer_idx, 
              batch_idx=b, cfg_type=cfg_type
            )

          # should_save = layer_idx is not None and timestep is not None and self.is_save
          # if should_save:
          #     os.makedirs(self.save_dir, exist_ok=True)
          # if should_save:
          #     filename = f"{mode}_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_{b}.pt"
          #     save_path = os.path.join(self.save_dir, filename)

          #     # 2. 将元数据和数据打包成一个字典
          #     entry_to_save = {
          #         "q": q_bmm.cpu(),
          #         "k": k_bmm.cpu(),
          #         "v": v_bmm.cpu(),
          #         "attn_probs": attn_probs.cpu(),
          #         # "data": attn_probs.cpu()
          #     }

          #     # 3. 加载、追加并保存
          #     if not os.path.exists(save_path):
          #         # try:
          #         #     existing_data = torch.load(save_path)
          #         #     if isinstance(existing_data, list):
          #         #         existing_data.append(entry_to_save)
          #         #         torch.save(existing_data, save_path)
          #         #     else: # 兼容旧格式
          #         #         torch.save([existing_data, entry_to_save], save_path)
          #         # except Exception as e:
          #         #     print(f"Could not append to {save_path}: {e}. Overwriting.")
          #         #     torch.save([entry_to_save], save_path)
          #     # else:
          #         torch.save([entry_to_save], save_path)
          
          context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
          context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

          outputs.append(context)

      return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

  def naive_varlen_posterior_truncate_attention(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: KVCacheStructure,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
      min_threshold = self.posterior_truncate_threshold
      vae_vit_range = self.get_vae_vit_range(kv_cache)
      self_range = self.get_self_range(kv_cache)

      num_q_heads = packed_query_states.size(1)
      num_kv_heads = merged_key_states.size(1)
      if num_q_heads != num_kv_heads:
          if num_q_heads % num_kv_heads != 0:
              raise ValueError(
                  f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
              )
          group_size = num_q_heads // num_kv_heads
          merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
          merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)


      for b in range(B):
          q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
          k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

          q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
          k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
          v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

          q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
          k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
          v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

          # attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2)) * scale # (n_heads, Lq, Lk)
          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

          if causal:
              Lq, Lk = attn_scores.size(1), attn_scores.size(2)
              if Lq > 1:
                  q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
                  k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                  
                  causal_mask_shift = Lk - Lq
                  mask = k_indices > (q_indices + causal_mask_shift)
                  
                  attn_scores.masked_fill_(mask, float("-inf"))

          attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)

          # if self.is_truncate:
          if vae_vit_range is not None and self_range is not None:
            attn_probs_vae_vit = attn_probs[:, :, vae_vit_range[0]:vae_vit_range[1]]
            attn_probs_self = attn_probs[:, :, self_range[0]:self_range[1]]
            if mode == "gen" and timestep is not None and timestep >= 0:
              if self.vae_vit:
                to_zero = torch.abs(attn_probs_vae_vit) < min_threshold
                self.sparsity[0][timestep][layer_idx] = torch.mean(to_zero.float()).item()
                attn_probs[:, :, vae_vit_range[0]:vae_vit_range[1]] = attn_probs_vae_vit.masked_fill(to_zero, 0.0)
              if self.self_attn:
                to_zero_self = torch.abs(attn_probs_self) < min_threshold
                self.sparsity[1][timestep][layer_idx] = torch.mean(to_zero_self.float()).item()
                attn_probs[:, :, self_range[0]:self_range[1]] = attn_probs_self.masked_fill(to_zero_self, 0.0)

          should_save = layer_idx is not None and self.is_save and timestep is not None and timestep >= 0
          if should_save:
            entry_to_save = {
                "q": q_bmm.cpu(),
                "k": k_bmm.cpu(),
                "v": v_bmm.cpu(),
                "attn_probs": attn_probs.cpu(),
                # "data": attn_probs.cpu()
            }
            self.save(
              entry_to_save=entry_to_save, mode=mode, 
              timestep=timestep, layer_idx=layer_idx, 
              batch_idx=b, cfg_type=cfg_type
            )

          context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
          context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

          outputs.append(context)

      return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

  def naive_varlen_sparse_attention(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: KVCacheStructure,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
      vae_vit_range = self.get_vae_vit_range(kv_cache)
      min_threshold = self.posterior_truncate_threshold

      num_q_heads = packed_query_states.size(1)
      num_kv_heads = merged_key_states.size(1)
      if num_q_heads != num_kv_heads:
          if num_q_heads % num_kv_heads != 0:
              raise ValueError(
                  f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
              )
          group_size = num_q_heads // num_kv_heads
          merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
          merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)

      for b in range(B):
          q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
          k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

          q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
          k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
          v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

          q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
          k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
          v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

          ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          ref_mask = ref_attn_probs < min_threshold

          mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
          representative_attn_scores = None
          q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
          
          if vae_vit_range is not None:
            if mode == "gen" and timestep is not None and timestep >= 0:
              vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
              vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor)
              vae_vit_mask = vae_vit_mask.to(torch.bool)
              coverage = compute_coverage(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], vae_vit_mask)
              # plot_mask_heads(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], heads=heads_to_plot, out_dir=f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
              #                 filename=f"ref_mask.png",
              #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
              #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
              # plot_mask_heads(vae_vit_mask, heads=heads_to_plot, out_dir=f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
              #                 filename=f"sparse_mask.png",
              #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
              #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
              if not os.path.exists(f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}"):
                os.makedirs(f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}", exist_ok=True)
              with open(f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}/coverage.txt", "w") as f:
                f.write(f"Coverage: {coverage}\n")
              # mask_sparse[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
              # assert(coverage == 1.0)
              mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
              # mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = ref_mask[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
              # sparsity[0][timestep][layer_idx] = torch.mean(ref_mask[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]])
              self.sparsity[0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()


          if causal:
              Lq, Lk = attn_scores.size(1), attn_scores.size(2)
              if Lq > 1:
                  q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
                  k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                  
                  causal_mask_shift = Lk - Lq
                  mask = k_indices > (q_indices + causal_mask_shift)
                  
                  attn_scores.masked_fill_(mask, float("-inf"))

          attn_scores = attn_scores.masked_fill(mask_sparse, float("-inf"))
          attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          
          if self.is_plot and mode == "gen" and timestep is not None and timestep >= 0:
            plot_mask_heads(representative_attn_scores, heads=self.heads_to_plot, out_dir=self.plot_dir,
                            filename=f"sparse_mask_layer_{layer_idx}_ts_{timestep}_batch_{b}.png",
                            ncols=6, cmap="inferno_r", tick_density=100, norm="log",
                            box_coords=None, box_style=None, figsize_per_plot=(5,5))

          should_save = layer_idx is not None and self.is_save and timestep is not None and timestep >= 0
          if should_save:
            entry_to_save = {
                "q": q_bmm.cpu(),
                "k": k_bmm.cpu(),
                "v": v_bmm.cpu(),
                "attn_probs": attn_probs.cpu(),
                # "data": attn_probs.cpu()
            }
            self.save(
              entry_to_save=entry_to_save, mode=mode, 
              timestep=timestep, layer_idx=layer_idx, 
              batch_idx=b, cfg_type=cfg_type
            )
          
          context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
          context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

          outputs.append(context)

      return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

  def naive_varlen_sparse_quant_attention(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: KVCacheStructure,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
      vae_vit_range = self.get_vae_vit_range(kv_cache)
      vae_range = self.get_vae_range(kv_cache)
      self_range = self.get_self_range(kv_cache)
      min_threshold = self.posterior_truncate_threshold

      num_q_heads = packed_query_states.size(1)
      num_kv_heads = merged_key_states.size(1)
      if num_q_heads != num_kv_heads:
          if num_q_heads % num_kv_heads != 0:
              raise ValueError(
                  f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
              )
          group_size = num_q_heads // num_kv_heads
          merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
          merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)

      for b in range(B):
          q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
          k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

          q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
          k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
          v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

          q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
          k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
          v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

          # q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_int4(q_bmm)
          q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_nvfp4(q_bmm)

          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          ref_mask = ref_attn_probs < min_threshold

          attn_scores_q_quant_fp4 = torch.bmm(q_bmm_quant_fp4, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          assert attn_scores_q_quant_fp4.shape == attn_scores.shape, "Quantized attention scores shape mismatch."

          mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
          representative_attn_scores = None
          q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
          
          if vae_vit_range is not None and vae_range is not None and self_range is not None:
            # if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal" or cfg_type == "cfg_text"):
            if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal"):
              vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
              vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor)
              vae_vit_mask = vae_vit_mask.to(torch.bool)
              # coverage = compute_coverage(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], vae_vit_mask)
              # plot_mask_heads(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], heads=heads_to_plot, out_dir=f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
              #                 filename=f"ref_mask.png",
              #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
              #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
              # plot_mask_heads(vae_vit_mask, heads=heads_to_plot, out_dir=f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
              #                 filename=f"sparse_mask.png",
              #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
              #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
              # if not os.path.exists(f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}"):
              #   os.makedirs(f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}", exist_ok=True)
              # with open(f"{self.plot_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}/coverage.txt", "w") as f:
              #   f.write(f"Coverage: {coverage}\n")
              # assert(coverage == 1.0)
              if self.vae_vit:
                mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
                self.sparsity[0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()
                attn_scores[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
              if self.self_attn:
                self_attn_mask = mask_sparse[:, :, vae_range[0]:vae_range[1]]
                self.sparsity[1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                attn_scores[:, :, self_range[0]:self_range[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range[0]:self_range[1]], attn_scores[:, :, self_range[0]:self_range[1]])

          if causal:
              Lq, Lk = attn_scores.size(1), attn_scores.size(2)
              if Lq > 1:
                  q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
                  k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                  
                  causal_mask_shift = Lk - Lq
                  mask = k_indices > (q_indices + causal_mask_shift)
                  
                  attn_scores.masked_fill_(mask, float("-inf"))

          attn_scores_masked = attn_scores.masked_fill(mask_sparse, float("-inf"))
          # attn_scores = attn_scores.masked_fill(mask_sparse, float("-inf"))
          attn_probs = torch.softmax(attn_scores_masked, dim=-1) # (n_heads, Lq, Lk)
          
          if self.is_plot and mode == "gen" and timestep is not None and timestep >= 0:
            plot_mask_heads(representative_attn_scores, heads=self.heads_to_plot, out_dir=self.plot_dir,
                            filename=f"sparse_mask_layer_{layer_idx}_ts_{timestep}_batch_{b}.png",
                            ncols=6, cmap="inferno_r", tick_density=100, norm="log",
                            box_coords=None, box_style=None, figsize_per_plot=(5,5))

          should_save = layer_idx is not None and self.is_save and timestep is not None and timestep >= 0
          if should_save:
            entry_to_save = {
                "q": q_bmm.cpu(),
                # "k": k_bmm.cpu(),
                # "v": v_bmm.cpu(),
                # "attn_probs": attn_probs.cpu(),
                # "data": attn_probs.cpu()
            }
            self.save(
              entry_to_save=entry_to_save, mode=mode, 
              timestep=timestep, layer_idx=layer_idx, 
              batch_idx=b, cfg_type=cfg_type
            )
          
          context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
          context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

          outputs.append(context)

      return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

  def forward(self, **kwargs):
    if self.attention_backend == "naive":
      kwargs.pop("kv_cache", None)  # kv_cache is not used in naive attention
      return self.naive_varlen_attention(**kwargs)
    elif self.attention_backend == "naive_truncate":
      return self.naive_varlen_posterior_truncate_attention(**kwargs)
    elif self.attention_backend == "naive_sparse":
      return self.naive_varlen_sparse_attention(**kwargs)
    elif self.attention_backend == "naive_sparse_quant":
      return self.naive_varlen_sparse_quant_attention(**kwargs)
    else:
      raise ValueError(f"Unsupported attention backend: {self.attention_backend}")


# DEBUG_CORE = False
# DEBUG = False

# min_threshold: float = 3e-6
# vae_vit_start = 48
# vae_vit_end = 3241
# self_start = 3491
# is_save = False
# is_truncate = False
# save_dir = "attn_probs_qkv_dump_new"
# sparsity = np.zeros((2, 49, 28), dtype=np.float32)

# is_plot = False

# is_sparse = True

# heads_to_plot = [0, 2, 4, 10, 16, 23, 27]

# sparsifier_threshold = min_threshold
# block_sparsifier = BlockSparsify(group_size=10, top_k=0.2, threshold=4e-5)

# coverage = np.zeros((2, 49, 28), dtype=np.float32)

# def set_save(flag: bool):
#   global is_save
#   print(f"Setting is_save to {flag}")
#   is_save = flag

# def set_truncate(flag: bool):
#   global is_truncate
#   print(f"Setting is_truncate to {flag}")
#   is_truncate = flag

# def set_dump_dir(new_dir: str):	
#   global save_dir
#   print(f"Setting save_dir to {new_dir}")
#   save_dir = new_dir

# def set_new_threshold(new_value):
#   global min_threshold
#   print(f"Changing min_threshold from {min_threshold} to {new_value}")
#   min_threshold = new_value

# def get_sparsity():
#   return sparsity
# def store_attn_scores(attn_probs, mode: str, timestep: Optional[int], layer_idx: Optional[int], b: int, save_dir: str):
#     if layer_idx is None or timestep is None or not is_save:
#         return
#     os.makedirs(save_dir, exist_ok=True)
#     filename = f"{mode}_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_{b}.pt"
#     save_path = os.path.join(save_dir, filename)
    
#     # 2. 将元数据和数据打包成一个字典
#     entry_to_save = {
#         "attn_probs": attn_probs.cpu()
#     }

#     # 3. 加载、追加并保存
#     if not os.path.exists(save_path):
#         torch.save([entry_to_save], save_path)
#     else:
#         torch.save([entry_to_save], save_path)

# def set_sparseify_threshold(new_value: float):
#   global sparsifier_threshold
#   print(f"Changing sparsifier_threshold from {sparsifier_threshold} to {new_value}")
#   sparsifier_threshold = new_value
#   block_sparsifier.threshold = new_value

# def naive_varlen_attention(
#     packed_query_states,   # (total_q, n_heads, head_dim)
#     merged_key_states,     # (total_k, n_kv_heads, head_dim)
#     merged_value_states,   # (total_k, n_kv_heads, head_dim)
#     cu_seqlens_q,          # (B+1,)
#     cu_seqlens_k,          # (B+1,)
#     causal: bool,
#     mode: str = "und",
#     timestep: Optional[int] = None,
#     layer_idx: Optional[int] = None,
# ):
#     outputs = []
#     B = cu_seqlens_q.numel() - 1
#     D = packed_query_states.shape[-1]
#     # scale = packed_query_states.size(-1) ** -0.5

#     num_q_heads = packed_query_states.size(1)
#     num_kv_heads = merged_key_states.size(1)
#     if num_q_heads != num_kv_heads:
#         if num_q_heads % num_kv_heads != 0:
#             raise ValueError(
#                 f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
#             )
#         group_size = num_q_heads // num_kv_heads
#         merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
#         merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)


#     for b in range(B):
#         q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
#         k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

#         q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
#         k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
#         v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

#         q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
#         k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
#         v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

#         # attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2)) * scale # (n_heads, Lq, Lk)
#         attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

#         if causal:
#             Lq, Lk = attn_scores.size(1), attn_scores.size(2)
#             if Lq > 1:
#                 q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
#                 k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                
#                 causal_mask_shift = Lk - Lq
#                 mask = k_indices > (q_indices + causal_mask_shift)
                
#                 attn_scores.masked_fill_(mask, float("-inf"))

#         attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)

#         if is_truncate:
#           attn_probs_vae_vit = attn_probs[:, :, vae_vit_start:vae_vit_end]
#           attn_probs_self = attn_probs[:, :, self_start:]
#           if mode == "gen" and timestep is not None and timestep >= 0:
#             to_zero = torch.abs(attn_probs_vae_vit) < min_threshold
#             sparsity[0][timestep][layer_idx] = torch.mean(to_zero.float()).item()
#             attn_probs[:, :, vae_vit_start:vae_vit_end] = attn_probs_vae_vit.masked_fill(to_zero, 0.0)

#             to_zero_self = torch.abs(attn_probs_self) < min_threshold
#             sparsity[1][timestep][layer_idx] = torch.mean(to_zero_self.float()).item()
#             attn_probs[:, :, self_start:] = attn_probs_self.masked_fill(to_zero_self, 0.0)


#         should_save = layer_idx is not None and timestep is not None and is_save
#         if should_save:
#             os.makedirs(save_dir, exist_ok=True)
#         if should_save:
#             filename = f"{mode}_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_{b}.pt"
#             save_path = os.path.join(save_dir, filename)
            
#             # 2. 将元数据和数据打包成一个字典
#             entry_to_save = {
#                 "q": q_bmm.cpu(),
#                 "k": k_bmm.cpu(),
#                 "v": v_bmm.cpu(),
#                 "attn_probs": attn_probs.cpu(),
#                 # "data": attn_probs.cpu()
#             }

#             # 3. 加载、追加并保存
#             if not os.path.exists(save_path):
#                 # try:
#                 #     existing_data = torch.load(save_path)
#                 #     if isinstance(existing_data, list):
#                 #         existing_data.append(entry_to_save)
#                 #         torch.save(existing_data, save_path)
#                 #     else: # 兼容旧格式
#                 #         torch.save([existing_data, entry_to_save], save_path)
#                 # except Exception as e:
#                 #     print(f"Could not append to {save_path}: {e}. Overwriting.")
#                 #     torch.save([entry_to_save], save_path)
#             # else:
#                 torch.save([entry_to_save], save_path)
        
#         context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
#         context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

#         outputs.append(context)

#     return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

# def naive_verlen_sparse_attention(
#     packed_query_states,   # (total_q, n_heads, head_dim)
#     merged_key_states,     # (total_k, n_kv_heads, head_dim)
#     merged_value_states,   # (total_k, n_kv_heads, head_dim)
#     cu_seqlens_q,          # (B+1,)
#     cu_seqlens_k,          # (B+1,)
#     causal: bool,
#     kv_cache: KVCacheStructure,
#     mode: str = "und",
#     timestep: Optional[int] = None,
#     layer_idx: Optional[int] = None,
# ):
#     outputs = []
#     B = cu_seqlens_q.numel() - 1
#     D = packed_query_states.shape[-1]
#     # scale = packed_query_states.size(-1) ** -0.5
#     vit_range = kv_cache.vit
#     vae_range = kv_cache.vae
#     vae_vit_range = None
#     if vit_range is not None and vae_range is not None:
#       vae_vit_range = (vae_range[0], vit_range[1]+1)
#     elif vit_range is None and vae_range is None:
#       vae_vit_range = None
#     elif vit_range is not None:
#       vae_vit_range = (vit_range[0], vit_range[1]+1)
#     elif vae_range is not None:
#       vae_vit_range = (vae_range[0], vae_range[1]+1)

#     num_q_heads = packed_query_states.size(1)
#     num_kv_heads = merged_key_states.size(1)
#     if num_q_heads != num_kv_heads:
#         if num_q_heads % num_kv_heads != 0:
#             raise ValueError(
#                 f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
#             )
#         group_size = num_q_heads // num_kv_heads
#         merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
#         merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)

#     out_dir = "plot/sparse_attention_scores_coverage"

#     for b in range(B):
#         q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
#         k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

#         q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
#         k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
#         v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

#         q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
#         k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
#         v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

#         attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

#         ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
#         ref_mask = ref_attn_probs < min_threshold

#         mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
#         representative_attn_scores = None
#         q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
        
#         if is_sparse and vae_vit_range is not None:
#           if mode == "gen" and timestep is not None and timestep >= 0:
#             vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
#             vae_vit_mask, representative_attn_scores = block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor)
#             vae_vit_mask = vae_vit_mask.to(torch.bool)
#             coverage = compute_coverage(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], vae_vit_mask)
#             # plot_mask_heads(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], heads=heads_to_plot, out_dir=f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
#             #                 filename=f"ref_mask.png",
#             #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
#             #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
#             # plot_mask_heads(vae_vit_mask, heads=heads_to_plot, out_dir=f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
#             #                 filename=f"sparse_mask.png",
#             #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
#             #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
#             if not os.path.exists(f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}"):
#               os.makedirs(f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}", exist_ok=True)
#             with open(f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}/coverage.txt", "w") as f:
#               f.write(f"Coverage: {coverage}\n")
#             # mask_sparse[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
#             # assert(coverage == 1.0)
#             mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
#             # mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = ref_mask[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
#             # sparsity[0][timestep][layer_idx] = torch.mean(ref_mask[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]])
#             sparsity[0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()


#         if causal:
#             Lq, Lk = attn_scores.size(1), attn_scores.size(2)
#             if Lq > 1:
#                 q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
#                 k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                
#                 causal_mask_shift = Lk - Lq
#                 mask = k_indices > (q_indices + causal_mask_shift)
                
#                 attn_scores.masked_fill_(mask, float("-inf"))

#         attn_scores = attn_scores.masked_fill(mask_sparse, float("-inf"))
#         attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
#         # attn_probs = attn_probs.masked_fill(mask_sparse, 0.0)
        
#         if is_plot and mode == "gen" and timestep is not None and timestep >= 0:
#           plot_mask_heads(representative_attn_scores, heads=heads_to_plot, out_dir="plot/sparse_attention_scores",
#                           filename=f"sparse_mask_layer_{layer_idx}_ts_{timestep}_batch_{b}.png",
#                           ncols=6, cmap="inferno_r", tick_density=100, norm="log",
#                           box_coords=None, box_style=None, figsize_per_plot=(5,5))

#         should_save = layer_idx is not None and timestep is not None and is_save
#         if should_save:
#             os.makedirs(save_dir, exist_ok=True)
#         if should_save:
#             filename = f"{mode}_qkv_attn_probs_ts_{timestep}_layer_{layer_idx}_batch_{b}.pt"
#             save_path = os.path.join(save_dir, filename)
            
#             # 2. 将元数据和数据打包成一个字典
#             entry_to_save = {
#                 # "q": q_bmm.cpu(),
#                 # "k": k_bmm.cpu(),
#                 # "v": v_bmm.cpu(),
#                 "attn_probs": attn_probs.cpu(),
#                 "mask_sparse": mask_sparse.cpu(),
#             }

#             # 3. 加载、追加并保存
#             if not os.path.exists(save_path):
#                 # try:
#                 #     existing_data = torch.load(save_path)
#                 #     if isinstance(existing_data, list):
#                 #         existing_data.append(entry_to_save)
#                 #         torch.save(existing_data, save_path)
#                 #     else: # 兼容旧格式
#                 #         torch.save([existing_data, entry_to_save], save_path)
#                 # except Exception as e:
#                 #     print(f"Could not append to {save_path}: {e}. Overwriting.")
#                 #     torch.save([entry_to_save], save_path)
#             # else:
#                 torch.save([entry_to_save], save_path)
        
#         context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
#         context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

#         outputs.append(context)

#     return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

# def naive_varlen_sparse_quant_attention(
#     packed_query_states,   # (total_q, n_heads, head_dim)
#     merged_key_states,     # (total_k, n_kv_heads, head_dim)
#     merged_value_states,   # (total_k, n_kv_heads, head_dim)
#     cu_seqlens_q,          # (B+1,)
#     cu_seqlens_k,          # (B+1,)
#     causal: bool,
#     kv_cache: KVCacheStructure,
#     mode: str = "und",
#     timestep: Optional[int] = None,
#     layer_idx: Optional[int] = None,
# ):
#     outputs = []
#     B = cu_seqlens_q.numel() - 1
#     D = packed_query_states.shape[-1]
#     # scale = packed_query_states.size(-1) ** -0.5
#     vit_range = kv_cache.vit
#     vae_range = kv_cache.vae
#     vae_vit_range = None
#     if vit_range is not None and vae_range is not None:
#       vae_vit_range = (vae_range[0], vit_range[1]+1)
#     elif vit_range is None and vae_range is None:
#       vae_vit_range = None
#     elif vit_range is not None:
#       vae_vit_range = (vit_range[0], vit_range[1]+1)
#     elif vae_range is not None:
#       vae_vit_range = (vae_range[0], vae_range[1]+1)

#     num_q_heads = packed_query_states.size(1)
#     num_kv_heads = merged_key_states.size(1)
#     if num_q_heads != num_kv_heads:
#         if num_q_heads % num_kv_heads != 0:
#             raise ValueError(
#                 f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
#             )
#         group_size = num_q_heads // num_kv_heads
#         merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
#         merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)

#     out_dir = "plot/sparse_attention_scores_coverage"

#     for b in range(B):
#         q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
#         k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

#         q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
#         k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
#         v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

#         q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
#         k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
#         v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

#         attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

#         ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
#         ref_mask = ref_attn_probs < min_threshold

#         mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
#         representative_attn_scores = None
#         q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
        
#         if is_sparse and vae_vit_range is not None:
#           if mode == "gen" and timestep is not None and timestep >= 0:
#             vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
#             vae_vit_mask, representative_attn_scores = block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor)
#             vae_vit_mask = vae_vit_mask.to(torch.bool)
#             coverage = compute_coverage(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], vae_vit_mask)
#             # plot_mask_heads(ref_mask[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]], heads=heads_to_plot, out_dir=f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
#             #                 filename=f"ref_mask.png",
#             #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
#             #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
#             # plot_mask_heads(vae_vit_mask, heads=heads_to_plot, out_dir=f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}",
#             #                 filename=f"sparse_mask.png",
#             #                 ncols=6, cmap="inferno_r", tick_density=100, norm="log",
#             #                 box_coords=None, box_style=None, figsize_per_plot=(5,5))
#             if not os.path.exists(f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}"):
#               os.makedirs(f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}", exist_ok=True)
#             with open(f"{out_dir}/ts_{timestep}_layer_{layer_idx}_batch_{b}/coverage.txt", "w") as f:
#               f.write(f"Coverage: {coverage}\n")
#             # mask_sparse[:, q_range_tensor[0]:q_range_tensor[1], vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
#             # assert(coverage == 1.0)
#             mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
#             # mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = ref_mask[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
#             # sparsity[0][timestep][layer_idx] = torch.mean(ref_mask[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]])
#             sparsity[0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()


#         if causal:
#             Lq, Lk = attn_scores.size(1), attn_scores.size(2)
#             if Lq > 1:
#                 q_indices = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
#                 k_indices = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)
                
#                 causal_mask_shift = Lk - Lq
#                 mask = k_indices > (q_indices + causal_mask_shift)
                
#                 attn_scores.masked_fill_(mask, float("-inf"))

#         attn_scores = attn_scores.masked_fill(mask_sparse, float("-inf"))
#         attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
#         # attn_probs = attn_probs.masked_fill(mask_sparse, 0.0)
        
#         if is_plot and mode == "gen" and timestep is not None and timestep >= 0:
#           plot_mask_heads(representative_attn_scores, heads=heads_to_plot, out_dir="plot/sparse_attention_scores",
#                           filename=f"sparse_mask_layer_{layer_idx}_ts_{timestep}_batch_{b}.png",
#                           ncols=6, cmap="inferno_r", tick_density=100, norm="log",
#                           box_coords=None, box_style=None, figsize_per_plot=(5,5))

#         should_save = layer_idx is not None and timestep is not None and is_save
#         if should_save:
#             os.makedirs(save_dir, exist_ok=True)
#         if should_save:
#             filename = f"{mode}_qkv_attn_probs_ts_{timestep}_layer_{layer_idx}_batch_{b}.pt"
#             save_path = os.path.join(save_dir, filename)
            
#             # 2. 将元数据和数据打包成一个字典
#             entry_to_save = {
#                 # "q": q_bmm.cpu(),
#                 # "k": k_bmm.cpu(),
#                 # "v": v_bmm.cpu(),
#                 "attn_probs": attn_probs.cpu(),
#                 "mask_sparse": mask_sparse.cpu(),
#             }

#             # 3. 加载、追加并保存
#             if not os.path.exists(save_path):
#                 # try:
#                 #     existing_data = torch.load(save_path)
#                 #     if isinstance(existing_data, list):
#                 #         existing_data.append(entry_to_save)
#                 #         torch.save(existing_data, save_path)
#                 #     else: # 兼容旧格式
#                 #         torch.save([existing_data, entry_to_save], save_path)
#                 # except Exception as e:
#                 #     print(f"Could not append to {save_path}: {e}. Overwriting.")
#                 #     torch.save([entry_to_save], save_path)
#             # else:
#                 torch.save([entry_to_save], save_path)
        
#         context_bmm = torch.bmm(attn_probs, v_bmm) # (n_heads, Lq, d)
#         context = context_bmm.transpose(0, 1) # (Lq, n_heads, d)

#         outputs.append(context)

#     return torch.cat(outputs, dim=0) # (total_q, n_heads, d)

# def flash_attn_varlen_attention(
#     packed_query_states: torch.Tensor,   # (total_q, n_heads, head_dim)
#     merged_key_states,     # (total_k, n_kv_heads, head_dim)
#     merged_value_states,   # (total_k, n_kv_heads, head_dim)
#     cu_seqlens_q,          # (B+1,)
#     cu_seqlens_k,          # (B+1,)
#     causal: bool,
#     mode: str = "und",
#     timestep: Optional[int] = None,
#     layer_idx: Optional[int] = None,
# ):
#   b = 0

#   num_q_heads = packed_query_states.size(1)
#   num_kv_heads = merged_key_states.size(1)
#   if num_q_heads != num_kv_heads:
#     if num_q_heads % num_kv_heads != 0:
#         raise ValueError(
#             f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
#         )
#     group_size = num_q_heads // num_kv_heads
#     merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
#     merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)

#   q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
#   k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

#   q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
#   k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
#   v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

#   q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
#   k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
#   v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

#   q_bmm = q_bmm.unsqueeze(0) # (1, n_heads, Lq, d)
#   k_bmm = k_bmm.unsqueeze(0) # (1, n_heads, Lk, d)
#   v_bmm = v_bmm.unsqueeze(0) # (1, n_heads, Lk, d)

#   output = quant_attn_score_triton(query=q_bmm, key=k_bmm, value=v_bmm,
#     args=type('args', (object,), {'score_bitwidth': 8, 'group_size': 32})(),
#     BLOCK_N = 128,
#     BLOCK_M = 128,
#     BLOCK_d = 64,
#   )

#   return output.squeeze(0).transpose(0, 1)  # (Lq, n_heads, d)

# def scaled_dot_attn_varlen_attention(
#     packed_query_states: torch.Tensor,   # (total_q, n_heads, head_dim)
#     merged_key_states,     # (total_k, n_kv_heads, head_dim)
#     merged_value_states,   # (total_k, n_kv_heads, head_dim)
#     cu_seqlens_q,          # (B+1,)
#     cu_seqlens_k,          # (B+1,)
#     causal: bool,
#     mode: str = "und",
#     timestep: Optional[int] = None,
#     layer_idx: Optional[int] = None,
# ):
#   b = 0

#   num_q_heads = packed_query_states.size(1)
#   num_kv_heads = merged_key_states.size(1)
#   if num_q_heads != num_kv_heads:
#     if num_q_heads % num_kv_heads != 0:
#         raise ValueError(
#             f"Grouped-query attention mismatch: {num_q_heads=} is not a multiple of {num_kv_heads=}."
#         )
#     group_size = num_q_heads // num_kv_heads
#     merged_key_states = merged_key_states.repeat_interleave(group_size, dim=1)
#     merged_value_states = merged_value_states.repeat_interleave(group_size, dim=1)

#   q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
#   k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()

#   q = packed_query_states[q_start:q_end]          # (Lq, n_heads, d)
#   k = merged_key_states[k_start:k_end]            # (Lk, n_kv_heads, d)
#   v = merged_value_states[k_start:k_end]          # (Lk, n_kv_heads, d)

#   q_bmm = q.transpose(0, 1)  # (n_heads, Lq, d)
#   k_bmm = k.transpose(0, 1)  # (n_heads, Lk, d)
#   v_bmm = v.transpose(0, 1)  # (n_heads, Lk, d)

#   q_bmm = q_bmm.unsqueeze(0) # (1, n_heads, Lq, d)
#   k_bmm = k_bmm.unsqueeze(0) # (1, n_heads, Lk, d)
#   v_bmm = v_bmm.unsqueeze(0) # (1, n_heads, Lk, d)

#   is_causal = causal
#   # attn_mask = None
#   # is_causal = False
#   # if causal:
#   #     Lq, Lk = q_bmm.size(2), k_bmm.size(2)
#   #     if Lq > 1:
#   #         q_indices = torch.arange(Lq, device=q_bmm.device).unsqueeze(1)
#   #         k_indices = torch.arange(Lk, device=k_bmm.device).unsqueeze(0)
          
#   #         causal_mask_shift = Lk - Lq
#   #         attn_mask = k_indices > (q_indices + causal_mask_shift)
#   #     else:
#   #         is_causal = True
#   with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
#     output = F.scaled_dot_product_attention(
#         query=q_bmm,
#         key=k_bmm,
#         value=v_bmm,
#         # attn_mask=attn_mask,
#         dropout_p=0.0,
#         is_causal=is_causal
#     )

#   output = output.squeeze(0).transpose(0, 1)

#   return output  # (Lq, n_heads, d)

# def comp_mse(flash: torch.Tensor, naive: torch.Tensor, mode: str, layer_idx: Optional[int], timestep: Optional[int]):
#   if layer_idx is None or timestep is None:
#       return
#   packed_attn_output_ref_float = flash.to(torch.float32)
#   packed_attn_output_float = naive.to(torch.float32)

#   mse_error = torch.nn.functional.mse_loss(packed_attn_output_ref_float, packed_attn_output_float)

#   save_dir = "attn_mse_error_dump"
#   os.makedirs(save_dir, exist_ok=True)
  
#   # 1. 定义要写入的文件路径
#   log_file_path = os.path.join(save_dir, "mse_log.txt")
  
#   # 2. 准备要写入的文本内容
#   log_entry = f"Mode: {mode}, Timestep: {timestep}, Layer: {layer_idx}, MSE: {mse_error.item():.8f}\n"

#   # 3. 使用 'a' 模式（追加）打开文件并写入
#   with open(log_file_path, 'a') as f:
#       f.write(log_entry)

# def attention_forward_core_ref_impl(q, k, v, sm_scale, causal, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2):
#     if DEBUG_CORE:
#         print()
#         print("attention_forward_core_ref_impl")
#         print("q:", q, q.shape)
#         print("k:", k, k.shape)
#         print("v:", v, v.shape)
#         print("sm_scale:", sm_scale)
#         print("causal:", causal)
#         print("dropout_p:", dropout_p)
#         print("philox_seed:", philox_seed)
#         print("philox_offset:", philox_offset)
#         print("use_exp2:", use_exp2)

#     # cast to float32
#     q = q.to(torch.float32)
#     k = k.to(torch.float32)
#     v = v.to(torch.float32)
    
#     # Compute attention scores
#     attention_scores = torch.matmul(q, k.transpose(-2, -1))
#     if DEBUG_CORE:
#         print("attention_scores:", attention_scores, attention_scores.shape)

#     # Scale scores
#     attention_scaled_scores = sm_scale * attention_scores
#     if DEBUG_CORE:
#         print("attention_scaled_scores:", attention_scaled_scores, attention_scaled_scores.shape)

#     # Apply ALiBi if slopes are provided
#     if alibi_slopes is not None:
#         L_q, L_k = q.shape[1], k.shape[1]
#         if DEBUG_CORE:
#             print("alibi_slopes:", alibi_slopes, alibi_slopes.shape)
#         # alibi_bias = compute_alibi_tensor_ref(alibi_slopes, L_q, L_k)
#         if DEBUG_CORE:
#             print("alibi_bias:", alibi_bias, alibi_bias.shape)
#         alibi_bias = alibi_bias.reshape(-1, L_q, L_k)
#         if DEBUG_CORE:
#             print("alibi_bias_flat:", alibi_bias, alibi_bias.shape)
#         attention_scaled_scores = attention_scaled_scores + alibi_bias
#         if DEBUG_CORE:
#             print("attention_scaled_scores after alibi:", attention_scaled_scores, attention_scaled_scores.shape)


#     # Apply causal mask if necessary
#     if causal:
#         L_q, L_k = q.shape[1], k.shape[1]
#         row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
#         col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
#         col_offset = L_q-L_k
#         causal_mask = row_idx >= (col_offset + col_idx)
#         if DEBUG_CORE:
#             print("causal_mask:", causal_mask)
#         # set -inf to places the causal mask is false
#         attention_scaled_scores = attention_scaled_scores.masked_fill(
#              torch.logical_not(causal_mask.unsqueeze(0)), float('-inf')
#         )
#         if DEBUG_CORE:
#             print("attention_scaled_scores after causal:", attention_scaled_scores, attention_scaled_scores.shape)

#     # Compute max for numerical stability
#     max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
#     if DEBUG_CORE:
#         print("max_scores:", max_scores, max_scores.shape)
#     if causal:
#         # Replace -inf in max_scores with zeros to avoid NaN in subtraction
#         max_scores = torch.where(
#             torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores
#         )
#         if DEBUG:
#             print("max_scores if causal:", max_scores, max_scores.shape)

#     # Shift scores
#     attention_shifted_scaled_scores = attention_scaled_scores - max_scores
#     if DEBUG_CORE:
#             print("attention_shifted_scaled_scores:", attention_shifted_scaled_scores, attention_shifted_scaled_scores.shape)

#     # Exponentiate
#     if use_exp2:
#         RCP_LN = 1 / math.log(2)
#         exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
#     else:
#         exp_scores = torch.exp(attention_shifted_scaled_scores)

#     if DEBUG_CORE:
#         print("exp_scores:", exp_scores, exp_scores.shape)

#     # Sum of exponentials
#     sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
#     if DEBUG_CORE:
#         print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)
#     if causal:
#         # if sum of exp scores is 0.0 it means scores where -inf, we cannot compute softmax and softmax_lse. Setting to 1 deals with -inf case cleanly 
#         sum_exp_scores = torch.where(
#         sum_exp_scores == 0,
#         torch.ones_like(sum_exp_scores),
#         sum_exp_scores
#         )
#     if DEBUG_CORE:
#         print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)

#     # Compute softmax probabilities
#     p = exp_scores / sum_exp_scores

#     if DEBUG_CORE:
#         print("softmax:", p, p.shape)
        
#     # apply dropout if specified
#     if dropout_p > 0.0:
#         rand_vals = torch.rand(p.shape, generator=torch.Generator(device=p.device).manual_seed(philox_seed), device=p.device, dtype=p.dtype)
#         dropout_mask, dropout_scale = rand_vals > dropout_p,  (1.0 / (1 - dropout_p))
#         if DEBUG_CORE:
#             print("dropout_scale:", dropout_scale)
#             print("dropout_mask:", dropout_mask)
#         # Apply dropout mask and scale
#         # Set -1 for dropped positions and 1 for kept positions in exp_scores 
#         sd_mask = torch.where(dropout_mask, exp_scores, -exp_scores)
#         p = torch.where(dropout_mask, p , torch.zeros_like(p)) * dropout_scale
#         if DEBUG_CORE:
#             print("softmax after dropout:", p)
#             print("sd_mask:", sd_mask)
#     else:
#         sd_mask = exp_scores
    
#     # Compute log-sum-exp
#     if use_exp2:
#         LN2 = math.log(2)
#         RCP_LN = 1 / math.log(2)
#         max_scores_base2 = max_scores * RCP_LN
#         softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
#         softmax_lse = softmax_lse_base2 * LN2
#         softmax_lse.squeeze_(-1)
#     else:
#         softmax_lse = max_scores + torch.log(sum_exp_scores)
#         softmax_lse = softmax_lse.squeeze(-1)

#     if DEBUG_CORE:
#         print("softmax_lse:", softmax_lse, softmax_lse.shape)

#     # Compute output
#     o = torch.matmul(p, v)
#     if DEBUG_CORE:
#         print("o:", o, o.shape)

#     # cast back to original dtype
#     # o = o.to(torch.float16)
#     o = o.to(torch.bfloat16)
#     # softmax_lse = softmax_lse.to(torch.float16) # NOTE: if you cast lse to fp16 it cause accuracy issues. keep fp32
#     sd_mask = sd_mask.to(torch.float16)

#     return o, softmax_lse, sd_mask

# def attention_varlen_forward_pytorch_ref_impl(
#     q,
#     k,
#     v,
#     sm_scale,
#     causal,
#     layout,
#     cu_seqlens_q,
#     cu_seqlens_k,
#     max_seqlen_q,
#     max_seqlen_k,
#     dropout_p, 
#     philox_seed, 
#     philox_offset,
#     alibi_slopes,
#     use_exp2
# ):
#     # Ensure the layout is 'thd'
#     if layout != 'thd':
#         raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

#     batch_size = cu_seqlens_q.shape[0] - 1
#     nheads_q, nheads_k = q.shape[1], k.shape[1]
#     head_dim = q.shape[2]

#     # Pre-allocate outputs
#     total_L_q = q.shape[0]
#     total_L_k = k.shape[0]

#     o = torch.zeros((total_L_q, nheads_q, head_dim), dtype=q.dtype, device=q.device)
#     softmax_lse = torch.zeros((total_L_q, nheads_q), dtype=torch.float32, device=q.device)
#     sd_mask = torch.zeros((batch_size, nheads_q, max_seqlen_q, max_seqlen_k), dtype=torch.float32, device=q.device)

#     # Compute group_size for MQA/GQA handling
#     group_size = nheads_q // nheads_k
#     if nheads_q % nheads_k != 0:
#         raise ValueError("nheads_q must be divisible by nheads_k")

#     for i in range(batch_size):
#         # Get the start and end indices for the current sequence
#         start_q = cu_seqlens_q[i].item()
#         end_q = cu_seqlens_q[i + 1].item()
#         start_k = cu_seqlens_k[i].item()
#         end_k = cu_seqlens_k[i + 1].item()

#         seqlen_q = end_q - start_q
#         seqlen_k = end_k - start_k

#         # if DEBUG:
#         #     print(f"Batch {i} with seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, Hq= {nheads_q}, Hk = {nheads_k}")

#         # Extract q_i, k_i, v_i
#         q_i = q[start_q:end_q, :, :]  # [L_q_i, nheads_q, head_dim]
#         k_i = k[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]
#         v_i = v[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]

#         # Permute to [nheads, L_q_i, head_dim]
#         q_i = q_i.permute(1, 0, 2)
#         k_i = k_i.permute(1, 0, 2)
#         v_i = v_i.permute(1, 0, 2)

#         # Handle MQA/GQA by adjusting shapes based on group_size
#         if group_size != 1:
#             # Reshape q_i to [nheads_k, group_size, L_q_i, head_dim]
#             q_i = q_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
#             # Expand k_i and v_i to match group_size
#             k_i = k_i.unsqueeze(1).expand(-1, group_size, -1, -1)
#             v_i = v_i.unsqueeze(1).expand(-1, group_size, -1, -1)
#             # Flatten the first two dimensions for computation
#             q_i = q_i.reshape(nheads_k * group_size, seqlen_q, head_dim)
#             k_i = k_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
#             v_i = v_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
#         else:
#             # Standard case
#             q_i = q_i.reshape(nheads_q, seqlen_q, head_dim)
#             k_i = k_i.reshape(nheads_k, seqlen_k, head_dim)
#             v_i = v_i.reshape(nheads_k, seqlen_k, head_dim)

#         if alibi_slopes is not None:
#             alibi_slopes_i = alibi_slopes[i]
#         else:
#             alibi_slopes_i = None

#         # Call the core attention function for this sequence
#         o_i, softmax_lse_i, sd_mask_i = attention_forward_core_ref_impl(q_i, k_i, v_i, sm_scale, causal, dropout_p, philox_seed, philox_offset, alibi_slopes_i, use_exp2)

#         # Reshape outputs back to original dimensions
#         if group_size != 1:
#             # Reshape outputs to [nheads_k, group_size, seqlen_q, head_dim]
#             o_i = o_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
#             # Combine the first two dimensions back to nheads_q
#             o_i = o_i.reshape(nheads_q, seqlen_q, head_dim)
#             # Reshape softmax_lse_i similarly
#             softmax_lse_i = softmax_lse_i.reshape(nheads_k, group_size, seqlen_q)
#             softmax_lse_i = softmax_lse_i.reshape(nheads_q, seqlen_q)
#         else:
#             # Outputs are already in the correct shape
#             pass

#         # Convert back to 'thd' layout
#         o_i = o_i.permute(1, 0, 2)  # [L_q_i, nheads_q, head_dim]
#         softmax_lse_i = softmax_lse_i.permute(1, 0)  # [L_q_i, nheads_q]
#         sd_mask_i = sd_mask_i # [nheads_q, L_q_i, L_k_i]

#         # Place outputs in pre-allocated tensors
#         o[start_q:end_q, :, :] = o_i
#         softmax_lse[start_q:end_q, :] = softmax_lse_i
#         sd_mask[i, :, :seqlen_q, :seqlen_k] = sd_mask_i

#     return o, softmax_lse, sd_mask

# def ref_varlen_attention(
#     packed_query_states,   # (total_q, n_heads, head_dim)
#     merged_key_states,     # (total_k, n_kv_heads, head_dim)
#     merged_value_states,   # (total_k, n_kv_heads, head_dim)
#     cu_seqlens_q,          # (B+1,)
#     cu_seqlens_k,          # (B+1,)
#     max_seqlen_q,
#     max_seqlen_k,
#     causal: bool,
#     mode: str = "und",
#     timestep: Optional[int] = None,
#     layer_idx: Optional[int] = None,
# ):
#   scale = packed_query_states.size(-1) ** -0.5
#   return attention_varlen_forward_pytorch_ref_impl(
#       q=packed_query_states,
#       k=merged_key_states,
#       v=merged_value_states,
#       sm_scale=scale,
#       causal=causal,
#       layout='thd',
#       cu_seqlens_q=cu_seqlens_q,
#       cu_seqlens_k=cu_seqlens_k,
#       max_seqlen_q=max_seqlen_q,
#       max_seqlen_k=max_seqlen_k,
#       dropout_p=0.0, 
#       philox_seed=0, 
#       philox_offset=0,
#       alibi_slopes=None,
#       use_exp2=True
#   )[0]