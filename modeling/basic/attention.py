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
    vae_vit: bool = True, self_attn: bool = False,
    quant_type: str = "int4"
  ):
    print("Initializing TrickAttention with settings: ")
    print(f"  attention_backend: {attention_backend}")
    print(f"  sparse_threshold: {sparse_threshold}")
    print(f"  sparse_gsize: {sparse_gsize}")
    print(f"  quant_gsize: {quant_gsize}")
    print(f"  quant_type: {quant_type}")
    print(f"  vae_vit: {vae_vit}")
    print(f"  self_attn: {self_attn}")
    print(f"  sparse_topk: {sparse_topk}")
    print(f"  posterior_truncate_threshold: {posterior_truncate_threshold}")
    print(f"  is_save: {is_save}")
    print(f"  save_dir: {save_dir}")

    self.block_sparsifier = BlockSparsify(group_size=sparse_gsize, top_k=sparse_topk, threshold=sparse_threshold)
    self.block_quantizer = BlockQuantize(group_size=quant_gsize, mode=quant_type)
    
    self.posterior_truncate_threshold = posterior_truncate_threshold
    self.save_dir = save_dir
    self.is_save = is_save
    self.is_plot = is_plot
    self.plot_dir = plot_dir
    self.is_truncate = is_truncate

    self.heads_to_plot = heads_to_plot

    self.attention_backend = attention_backend

    self.sparsity = {
      "normal": np.zeros((2, 49, 28), dtype=np.float32),
      "cfg_text": np.zeros((2, 49, 28), dtype=np.float32),
      "cfg_img": np.zeros((2, 49, 28), dtype=np.float32)
    }

    self.average_self_attn_scores = {
      "normal": np.zeros((49 ,28), dtype=np.float32),
      "cfg_text": np.zeros((49 ,28), dtype=np.float32),
      "cfg_img": np.zeros((49 ,28), dtype=np.float32)
    }

    self.vae_vit = vae_vit
    self.self_attn = self_attn


  def get_sparsity(self):
    return self.sparsity

  def get_ave_self_attn_scores(self):
    return self.average_self_attn_scores

  def clear_sparsity(self):
    self.sparsity = {
      "normal": np.zeros((2, 49, 28), dtype=np.float32),
      "cfg_text": np.zeros((2, 49, 28), dtype=np.float32),
      "cfg_img": np.zeros((2, 49, 28), dtype=np.float32)
    }

  def clear_ave_self_attn_scores(self):
    self.average_self_attn_scores = {
      "normal": np.zeros((49 ,28), dtype=np.float32),
      "cfg_text": np.zeros((49 ,28), dtype=np.float32),
      "cfg_img": np.zeros((49 ,28), dtype=np.float32)
    }

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
  
  @staticmethod
  def get_vit_range(
    kv_cache: KVCacheStructure,
  ):
    vit_range = kv_cache.vit
    return (vit_range[0], vit_range[1]+1) if vit_range is not None else None
  
  @staticmethod
  def get_cfg_txt_self_range(
    kv_cache: KVCacheStructure, 
  ):
    self_range = kv_cache.cfg_text_gen_image
    return (self_range[0], self_range[1]+1) if self_range is not None else None
  
  @staticmethod
  def get_cfg_img_self_range(
    kv_cache: KVCacheStructure, 
  ):
    self_range = kv_cache.cfg_img_gen_image
    return (self_range[0], self_range[1]+1) if self_range is not None else None

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
  
  def naive_varlen_attention_cal_ave_self_attn_score(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: Optional[KVCacheStructure],
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

          if kv_cache is not None and cfg_type is not None and layer_idx is not None:
              vae_vit_range = torch.tensor(self.get_vae_vit_range(kv_cache), device=q_bmm.device, dtype=torch.long)
              self_range_tensor = torch.tensor(self.get_self_range(kv_cache), device=q_bmm.device, dtype=torch.long)
              if cfg_type == "normal":
                self_range_tensor = torch.tensor(self.get_self_range(kv_cache), device=q_bmm.device, dtype=torch.long)
              elif cfg_type == "cfg_text":
                self_range_tensor = torch.tensor(self.get_cfg_txt_self_range(kv_cache), device=q_bmm.device, dtype=torch.long)
              elif cfg_type == "cfg_img":
                self_range_tensor = torch.tensor(self.get_cfg_img_self_range(kv_cache), device=q_bmm.device, dtype=torch.long)
              self_attn_probs = attn_probs[:, :, self_range_tensor[0]:self_range_tensor[1]] # (n_heads, Lq, Lself)
              avg_self_attn_probs = torch.mean(self_attn_probs, dim=0) # (Lq, Lself)
              avg_self_attn_probs = torch.mean(torch.sum(avg_self_attn_probs, dim=1), dim=0) # scala
              self.average_self_attn_scores[cfg_type][timestep][layer_idx] = avg_self_attn_probs.cpu().numpy()



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

  def naive_varlen_posterior_truncate_attention(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: Optional[KVCacheStructure],
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
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
          if kv_cache is not None:
            vae_vit_range = self.get_vae_vit_range(kv_cache)
            self_range = self.get_self_range(kv_cache)
            if vae_vit_range is not None and self_range is not None:
              attn_probs_vae_vit = attn_probs[:, :, vae_vit_range[0]:vae_vit_range[1]]
              attn_probs_self = attn_probs[:, :, self_range[0]:self_range[1]]
              if mode == "gen" and timestep is not None and timestep >= 0:
                if self.vae_vit:
                  to_zero = torch.abs(attn_probs_vae_vit) < min_threshold
                  self.sparsity[cfg_type][0][timestep][layer_idx] = torch.mean(to_zero.float()).item()
                  attn_probs[:, :, vae_vit_range[0]:vae_vit_range[1]] = attn_probs_vae_vit.masked_fill(to_zero, 0.0)
                if self.self_attn:
                  to_zero_self = torch.abs(attn_probs_self) < min_threshold
                  self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(to_zero_self.float()).item()
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
              self.sparsity[cfg_type][0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()


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
      kv_cache: Optional[KVCacheStructure] = None,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
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

          q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_int4(q_bmm)
          # q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_nvfp4(q_bmm)

          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          ref_mask = ref_attn_probs < min_threshold

          attn_scores_q_quant_fp4 = torch.bmm(q_bmm_quant_fp4, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          assert attn_scores_q_quant_fp4.shape == attn_scores.shape, "Quantized attention scores shape mismatch."

          mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
          representative_attn_scores = None
          q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
          
          if kv_cache is not None:
            vae_vit_range = self.get_vae_vit_range(kv_cache)
            vae_range = self.get_vae_range(kv_cache)
            vit_range = self.get_vit_range(kv_cache)
            self_range = self.get_self_range(kv_cache)

            if vae_vit_range is not None and vae_range is not None and self_range is not None:
              # if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal" or cfg_type == "cfg_text"):
              if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal"):
                vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
                vit_range_tensor = torch.tensor(vit_range, device=q_bmm.device, dtype=torch.long)
                vae_range_tensor = torch.tensor(vae_range, device=q_bmm.device, dtype=torch.long)
                # vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor, vae_range_tensor, vit_range_tensor)
                vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_fine(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor, vae_range_tensor, vit_range_tensor)
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
                  self.sparsity[cfg_type][0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()
                  attn_scores[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
                if self.self_attn:
                  self_attn_mask = mask_sparse[:, :, vae_range[0]:vae_range[1]]
                  self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
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
  
  def naive_varlen_sparse_quant_attention_v2(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: Optional[KVCacheStructure] = None,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      """
      Directly use the attention scores multipled by a scale of self-attn as the score for VAE
      """
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
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

          q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_int4(q_bmm)
          # q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_nvfp4(q_bmm)

          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          ref_mask = ref_attn_probs < min_threshold

          attn_scores_q_quant_fp4 = torch.bmm(q_bmm_quant_fp4, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          assert attn_scores_q_quant_fp4.shape == attn_scores.shape, "Quantized attention scores shape mismatch."

          mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
          representative_attn_scores = None
          q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
          
          if kv_cache is not None:
            vae_vit_range = self.get_vae_vit_range(kv_cache)
            vae_range = self.get_vae_range(kv_cache)
            vit_range = self.get_vit_range(kv_cache)
            self_range = self.get_self_range(kv_cache)

            if vae_vit_range is not None and vae_range is not None and self_range is not None:
              # if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal" or cfg_type == "cfg_text"):
              if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal"):
                vae_self_scale = 1
                
                vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
                vit_range_tensor = torch.tensor(vit_range, device=q_bmm.device, dtype=torch.long)
                vae_range_tensor = torch.tensor(vae_range, device=q_bmm.device, dtype=torch.long)
                self_range_tensor = torch.tensor(self_range, device=q_bmm.device, dtype=torch.long)
                vae_mask, vit_mask, self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_vit_self_fine(
                    q_bmm, k_bmm, vae_range_tensor, vit_range_tensor, self_range_tensor
                )
                vae_mask = vae_mask.to(torch.bool)
                vit_mask = vit_mask.to(torch.bool)
                self_mask = self_mask.to(torch.bool)

                vae_vit_mask = torch.cat([vae_mask, vit_mask], dim=-1)
                assert vae_vit_mask.shape[2] == (vae_vit_range_tensor[1] - vae_vit_range_tensor[0]), "VAE+VIT mask shape mismatch."
                mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
                # mask_sparse[:, :, vit_range_tensor[0]:vit_range_tensor[1]] = vit_mask
                
                attn_scores[:, :, vit_range_tensor[0]:vit_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, vit_range_tensor[0]:vit_range_tensor[1]]
                attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(
                    self_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]]
                )
                attn_scores[:, :, vae_range_tensor[0]:vae_range_tensor[1]] = attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] / vae_self_scale
                self.sparsity[0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()
                self.sparsity[1][timestep][layer_idx] = torch.mean(self_mask.float()).item()
                # vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor, vae_range_tensor, vit_range_tensor)
                # vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_fine(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor, vae_range_tensor, vit_range_tensor)
                # vae_vit_mask = vae_vit_mask.to(torch.bool)
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
                # if self.vae_vit:
                #   mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
                #   self.sparsity[0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()
                #   attn_scores[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
                # if self.self_attn:
                #   self_attn_mask = mask_sparse[:, :, vae_range[0]:vae_range[1]]
                #   self.sparsity[1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                #   attn_scores[:, :, self_range[0]:self_range[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range[0]:self_range[1]], attn_scores[:, :, self_range[0]:self_range[1]])

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
  
  def naive_varlen_sparse_quant_attention_cfg_self(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: Optional[KVCacheStructure] = None,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
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
          q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.simulate_quantization(q_bmm)

          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          ref_mask = ref_attn_probs < min_threshold

          attn_scores_q_quant_fp4 = torch.bmm(q_bmm_quant_fp4, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          assert attn_scores_q_quant_fp4.shape == attn_scores.shape, "Quantized attention scores shape mismatch."

          mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
          representative_attn_scores = None
          q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
          
          if kv_cache is not None:
            # vae_vit_range = self.get_vae_vit_range(kv_cache)
            # vae_range = self.get_vae_range(kv_cache)
            # vit_range = self.get_vit_range(kv_cache)
            self_range = self.get_self_range(kv_cache)
            cfg_txt_self_range = self.get_cfg_txt_self_range(kv_cache)
            # cfg_img_self_range = self.get_cfg_img_self_range(kv_cache)

            if self_range is not None:
              if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal" or cfg_type == "cfg_text" or cfg_type == "cfg_img"):
                assert cfg_type != "cfg_img", "CFG image self-attention only sparse-quant not implemented yet."
                if cfg_type == "normal":
                  self_range_tensor = torch.tensor(self_range, device=q_bmm.device, dtype=torch.long)
                  self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_self_fine(q_bmm, k_bmm, self_range_tensor)
                  self_mask = self_mask.to(torch.bool)
                  if self.self_attn:
                    self_attn_mask = self_mask
                    self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                    attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]])
                elif cfg_type == "cfg_text":
                  self_range_tensor = torch.tensor(cfg_txt_self_range, device=q_bmm.device, dtype=torch.long)
                  self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_self_fine(q_bmm, k_bmm, self_range_tensor)
                  self_mask = self_mask.to(torch.bool)
                  if self.self_attn:
                    self_attn_mask = self_mask
                    self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                    attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]])

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

  def naive_varlen_sparse_quant_attention_cfg(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: Optional[KVCacheStructure] = None,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
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

          q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.simulate_quantization(q_bmm)
          # q_bmm_quant_fp4, q_quant_scale = self.block_quantizer.quantize_nvfp4(q_bmm)

          attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)

          ref_attn_probs = torch.softmax(attn_scores, dim=-1) # (n_heads, Lq, Lk)
          ref_mask = ref_attn_probs < min_threshold

          attn_scores_q_quant_fp4 = torch.bmm(q_bmm_quant_fp4, k_bmm.transpose(1, 2) / math.sqrt(D)) # (n_heads, Lq, Lk)
          assert attn_scores_q_quant_fp4.shape == attn_scores.shape, "Quantized attention scores shape mismatch."

          mask_sparse = attn_scores.new_zeros(attn_scores.size(), dtype=torch.bool)
          representative_attn_scores = None
          q_range_tensor = torch.tensor([0, q_bmm.size(1)], device=q_bmm.device, dtype=torch.long)
          
          if kv_cache is not None:
            vae_vit_range = self.get_vae_vit_range(kv_cache)
            vae_range = self.get_vae_range(kv_cache)
            vit_range = self.get_vit_range(kv_cache)
            self_range = self.get_self_range(kv_cache)
            cfg_txt_self_range = self.get_cfg_txt_self_range(kv_cache)
            cfg_img_self_range = self.get_cfg_img_self_range(kv_cache)

            if vae_vit_range is not None and vae_range is not None and self_range is not None:
              if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal" or cfg_type == "cfg_text" or cfg_type == "cfg_img"):
                vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
                vit_range_tensor = torch.tensor(vit_range, device=q_bmm.device, dtype=torch.long)
                vae_range_tensor = torch.tensor(vae_range, device=q_bmm.device, dtype=torch.long)
                self_range_tensor = torch.tensor(self_range, device=q_bmm.device, dtype=torch.long)
                if cfg_type == "normal" or cfg_type == "cfg_text":
                  self_range_tensor = torch.tensor(cfg_txt_self_range, device=q_bmm.device, dtype=torch.long) if cfg_type == "cfg_text" else self_range_tensor
                  
                  assert (vae_range_tensor[1] - vae_range_tensor[0]) == (self_range_tensor[1] - self_range_tensor[0]), "VAE and Self range length mismatch."
                  vae_mask, vit_mask, self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_vae_vit_self_fine(q_bmm, k_bmm, vae_range_tensor, vit_range_tensor, self_range_tensor)
                  vae_mask = vae_mask.to(torch.bool)
                  vit_mask = vit_mask.to(torch.bool)
                  # self_mask = self_mask.to(torch.bool)
                  vae_vit_mask = torch.cat([vae_mask, vit_mask], dim=-1)
                  assert vae_vit_mask.shape[2] == (vae_vit_range_tensor[1] - vae_vit_range_tensor[0]), "VAE+VIT mask shape mismatch."
                  if self.vae_vit:
                    mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
                    self.sparsity[cfg_type][0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()
                    attn_scores[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
                  if self.self_attn:
                    self_attn_mask = mask_sparse[:, :, vae_range[0]:vae_range[1]]
                    self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                    attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]])
                elif cfg_type == "cfg_img":
                  self_range_tensor = torch.tensor(cfg_img_self_range, device=q_bmm.device, dtype=torch.long)

                  self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_self_fine(q_bmm, k_bmm, self_range_tensor)
                  self_mask = self_mask.to(torch.bool)
                  if self.self_attn:
                    self_attn_mask = self_mask
                    self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                    attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]])

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

  def naive_varlen_sparse_quant_attention_cfg_v2(
      self,
      packed_query_states,   # (total_q, n_heads, head_dim)
      merged_key_states,     # (total_k, n_kv_heads, head_dim)
      merged_value_states,   # (total_k, n_kv_heads, head_dim)
      cu_seqlens_q,          # (B+1,)
      cu_seqlens_k,          # (B+1,)
      causal: bool,
      kv_cache: Optional[KVCacheStructure] = None,
      mode: str = "und",
      timestep: Optional[int] = None,
      layer_idx: Optional[int] = None,
      cfg_type: Optional[str] = None,
  ):
      ''''
      self-attention fully quantized
      '''
      outputs = []
      B = cu_seqlens_q.numel() - 1
      D = packed_query_states.shape[-1]
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
          
          if kv_cache is not None:
            vae_vit_range = self.get_vae_vit_range(kv_cache)
            vae_range = self.get_vae_range(kv_cache)
            vit_range = self.get_vit_range(kv_cache)
            self_range = self.get_self_range(kv_cache)
            cfg_txt_self_range = self.get_cfg_txt_self_range(kv_cache)
            cfg_img_self_range = self.get_cfg_img_self_range(kv_cache)

            if vae_vit_range is not None and vae_range is not None and self_range is not None:
              if mode == "gen" and timestep is not None and timestep >= 0 and cfg_type is not None and (cfg_type == "normal" or cfg_type == "cfg_text" or cfg_type == "cfg_img"):
                vae_vit_range_tensor = torch.tensor(vae_vit_range, device=q_bmm.device, dtype=torch.long)
                vit_range_tensor = torch.tensor(vit_range, device=q_bmm.device, dtype=torch.long)
                vae_range_tensor = torch.tensor(vae_range, device=q_bmm.device, dtype=torch.long)
                self_range_tensor = torch.tensor(self_range, device=q_bmm.device, dtype=torch.long)
                if cfg_type == "normal" or cfg_type == "cfg_text":
                  self_range_tensor = torch.tensor(cfg_txt_self_range, device=q_bmm.device, dtype=torch.long) if cfg_type == "cfg_text" else self_range_tensor
                  assert (vae_range_tensor[1] - vae_range_tensor[0]) == (self_range_tensor[1] - self_range_tensor[0]), "VAE and Self range length mismatch."
                  vae_mask, vit_mask, self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_vae_vit_self_fine(q_bmm, k_bmm, vae_range_tensor, vit_range_tensor, self_range_tensor)
                  vae_mask = vae_mask.to(torch.bool)
                  vit_mask = vit_mask.to(torch.bool)
                  # self_mask = self_mask.to(torch.bool)
                  vae_vit_mask = torch.cat([vae_mask, vit_mask], dim=-1)
                  assert vae_vit_mask.shape[2] == (vae_vit_range_tensor[1] - vae_vit_range_tensor[0]), "VAE+VIT mask shape mismatch."
                  if self.vae_vit:
                    mask_sparse[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = vae_vit_mask
                    self.sparsity[cfg_type][0][timestep][layer_idx] = torch.mean(vae_vit_mask.float()).item()
                    attn_scores[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, vae_vit_range_tensor[0]:vae_vit_range_tensor[1]]
                  if self.self_attn:
                    self_attn_mask = mask_sparse[:, :, vae_range[0]:vae_range[1]]
                    self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                    attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]]
                    # attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]])
                elif cfg_type == "cfg_img":
                  self_range_tensor = torch.tensor(cfg_img_self_range, device=q_bmm.device, dtype=torch.long)
                  self_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_self_fine(q_bmm, k_bmm, self_range_tensor)
                  self_mask = self_mask.to(torch.bool)
                  if self.self_attn:
                    self_attn_mask = self_mask
                    self.sparsity[cfg_type][1][timestep][layer_idx] = torch.mean(self_attn_mask.float()).item()
                    attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]]
                    # attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]] = torch.where(self_attn_mask, attn_scores_q_quant_fp4[:, :, self_range_tensor[0]:self_range_tensor[1]], attn_scores[:, :, self_range_tensor[0]:self_range_tensor[1]])
                # vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor, vae_range_tensor, vit_range_tensor)
                # vae_vit_mask, representative_attn_scores = self.block_sparsifier.sparsify_kv_cache_threshold_fine(q_bmm, k_bmm, q_range_tensor, vae_vit_range_tensor, vae_range_tensor, vit_range_tensor)
                # vae_vit_mask = vae_vit_mask.to(torch.bool)
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
    elif self.attention_backend == "naive_cal_ave_self_attn_score":
      return self.naive_varlen_attention_cal_ave_self_attn_score(**kwargs)
    elif self.attention_backend == "naive_truncate":
      return self.naive_varlen_posterior_truncate_attention(**kwargs)
    elif self.attention_backend == "naive_sparse":
      return self.naive_varlen_sparse_attention(**kwargs)
    elif self.attention_backend == "naive_sparse_quant":
      return self.naive_varlen_sparse_quant_attention(**kwargs)
    elif self.attention_backend == "naive_sparse_quant_cfg":
      return self.naive_varlen_sparse_quant_attention_cfg(**kwargs)
    elif self.attention_backend == "naive_sparse_quant_relocate":
      return self.naive_varlen_sparse_quant_attention_v2(**kwargs)
    elif self.attention_backend == "naive_sparse_quant_cfg_v2":
      return self.naive_varlen_sparse_quant_attention_cfg_v2(**kwargs)
    elif self.attention_backend == "naive_sparse_quant_cfg_self":
      return self.naive_varlen_sparse_quant_attention_cfg_self(**kwargs)
    else:
      raise ValueError(f"Unsupported attention backend: {self.attention_backend}")
