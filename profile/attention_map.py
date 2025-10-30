import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from safetensors.torch import load_file
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from transformers import AutoConfig
import numpy as np

import glob

from matplotlib.patches import Rectangle

import argparse

def attention_plot(attention, ax, x_texts=None, y_texts=None, annot=False,
                   vmin=None, vmax=None, square=True,
                   norm=None, cbar=False, title=None, cmap="rocket_r", tick_density=100,
                   xlabel=None, ylabel=None,
                   box_coords=None, box_style=None): # 修改了参数
    """
    修改后的版本：直接接收一个 norm 对象。
    """
    sns.set_theme(font_scale=1.25)

    arr = attention.detach().cpu().float().numpy() if hasattr(attention, "detach") else np.asarray(attention)
    
    # 移除 percentile 和 norm_type 的逻辑，因为 norm 对象是直接传入的
    # from matplotlib.colors import Normalize, PowerNorm, LogNorm
    # ... (移除所有 norm 创建相关的代码) ...
    
    sns.heatmap(arr,
                ax=ax,
                cbar=cbar,
                cmap=cmap,
                norm=norm, # 直接使用传入的 norm 对象
                annot=annot,
                square=square,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=tick_density,
                xticklabels=tick_density,
                vmin=vmin, vmax=vmax
                )

    if title:
        ax.set_title(title)

    if xlabel:
      ax.set_xlabel(xlabel, color='lightgray', fontsize=12)
    if ylabel:
      ax.set_ylabel(ylabel, color='lightgray', fontsize=12)
      
    # ---------- 在此处添加矩形框 ----------
    if box_coords is not None:
        row0, row1, col0, col1 = box_coords
        # 对齐到 heatmap 的格子边界，通常减 0.5
        x = col0 - 0.5
        y = row0 - 0.5
        width = col1 - col0
        height = row1 - row0

        # 默认样式
        default_style = {'edgecolor': 'white', 'linewidth': 2.0, 'linestyle': '-', 'alpha': 0.9}
        if box_style:
            default_style.update(box_style)

        rect = Rectangle((x, y), width, height, fill=False,
                         edgecolor=default_style['edgecolor'],
                         linewidth=default_style['linewidth'],
                         linestyle=default_style.get('linestyle', '-'),
                         alpha=default_style.get('alpha', 1.0),
                         transform=ax.transData,
                         clip_on=False)
        ax.add_patch(rect)
    # ----------------------------------------

    # 1. 调整刻度标签的外观
    ax.tick_params(
        axis='both',          # 应用于 x 和 y 轴
        which='major',        # 应用于主刻度
        labelsize=8,          # 设置一个较小的字体大小
        labelcolor='lightgray', # 在暗色背景下使用浅灰色
        bottom=True, top=False, left=True, right=False # 只在左侧和底部显示标签
    )
    # 旋转 x 轴标签以防重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 2. 显示子图的边框 (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray') 
        spine.set_linewidth(0.5)

def plot_attn_map_distribution(attn_map, title="distribution", vmin=None, vmax=None, figure_path='./figures', figure_name='attention_weight.png'):
    attn_map_np = attn_map.detach().cpu().numpy() if hasattr(attn_map, "detach") else np.asarray(attn_map)
    plt.figure(figsize=(10, 6))
    sns.histplot(attn_map_np.flatten(), bins=50, kde=False, color='blue', binrange=(vmin, vmax) if vmin is not None and vmax is not None else None)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()

def kv_plot(kv, ax, x_texts=None, y_texts=None, annot=False,
              vmin=None, vmax=None, square=True,
              norm=None, cbar=False, title=None, cmap="rocket_r", tick_density=100,
              xlabel=None, ylabel=None, aspect_ratio='auto'):
    sns.set_theme(font_scale=1.25)

    arr = kv.detach().cpu().float().numpy() if hasattr(kv, "detach") else np.asarray(kv)

    # 移除 percentile 和 norm_type 的逻辑，因为 norm 对象是直接传入的
    # from matplotlib.colors import Normalize, PowerNorm, LogNorm
    # ... (移除所有 norm 创建相关的代码) ...
    
    sns.heatmap(arr,
                ax=ax,
                cbar=cbar,
                cmap=cmap,
                norm=norm, # 直接使用传入的 norm 对象
                annot=annot,
                square=square,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=tick_density,
                xticklabels=tick_density,
                vmin=vmin, vmax=vmax
                )

    if title:
        ax.set_title(title)

    if xlabel:
      ax.set_xlabel(xlabel, color='lightgray', fontsize=12)
    if ylabel:
      ax.set_ylabel(ylabel, color='lightgray', fontsize=12)

    ax.set_aspect(aspect_ratio)

    # 1. 调整刻度标签的外观
    ax.tick_params(
        axis='both',          # 应用于 x 和 y 轴
        which='major',        # 应用于主刻度
        labelsize=8,          # 设置一个较小的字体大小
        labelcolor='lightgray', # 在暗色背景下使用浅灰色
        bottom=True, top=False, left=True, right=False # 只在左侧和底部显示标签
    )
    # 旋转 x 轴标签以防重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 2. 显示子图的边框 (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray') 
        spine.set_linewidth(0.5)

boundary_indices = {
    "global": (0, 4868),
    "system_prompt": (0, 47),
    "vae": (48, 1425),
    "vit": (1426, 3240),
    "vae_vit": (48, 3240),
    "input_prompt": (3241, 3253),
    "gen_text": (3254, 3490),
    "self": (3491, 4868),
}

# vae = 3202, vit = 3852
# boundary_indices = {
#     "global": (0, 10390),
#     "system_prompt": (0, 47),
#     "vae": (48, 3249),
#     "vit": (3250, 7101),
#     "vae_vit": (48, 7101),
#     "input_prompt": (7102, 7120),
#     "gen_text": (7121, 7188),
#     "self": (7189, 10390),
# }

# boundary_indices
def extract_global(attn_map, head=-1):
  assert head < attn_map.shape[0], f"Head index {head} out of range for attention map with {attn_map.shape[0]} heads."
  if head < 0:
    attn_map = attn_map.mean(dim=0)
  else:
    attn_map = attn_map[head]

  flatten_boundary = [ pair for pair in boundary_indices.values() if pair != boundary_indices["global"] ]
  cols_to_remove = list()
  for start, end in flatten_boundary:
    cols_to_remove.append(start)
    cols_to_remove.append(end)
  # cols_to_remove = [0, 47, 48, 1425, 1426, 3240, 3241, 3253, 3254, 3482, 3483, 4860]
  rows_to_remove = [0, boundary_indices["self"][1]-boundary_indices["self"][0]]
  num_cols = attn_map.shape[1]
  num_rows = attn_map.shape[0]
  cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]
  return attn_map

def extract_self(attn_map):
  attn_map = attn_map.mean(dim=0)
  
  row_range = (1, boundary_indices["self"][1]-boundary_indices["self"][0])
  col_range = (boundary_indices["self"][0]+1, boundary_indices["self"][1])
  print(col_range)
  attn_map = attn_map[row_range[0]:row_range[1], col_range[0]:col_range[1]]
  # cols_to_remove = [0, 47, 48, 1425, 1426, 3240, 3241, 3253, 3254, 3482, 3483, 4860]
  # rows_to_remove = [0, 1377]
  # num_cols = attn_map.shape[1]
  # num_rows = attn_map.shape[0]
  # cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
  # rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  # attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]
  return attn_map

def extract_vae(attn_map):
  attn_map = attn_map.mean(dim=0)
  
  row_range = (1, boundary_indices["self"][1]-boundary_indices["self"][0])
  col_range = (boundary_indices["vae"][0]+1, boundary_indices["vae"][1])
  # rows_to_remove = [0, 1377]
  # num_rows = attn_map.shape[0]
  # rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[row_range[0]:row_range[1], col_range[0]:col_range[1]]
  return attn_map

def extract_vit(attn_map):
  attn_map = attn_map.mean(dim=0)

  rows_range = (1, boundary_indices["self"][1]-boundary_indices["self"][0])
  cols_range = (boundary_indices["vit"][0]+1, boundary_indices["vit"][1])
  attn_map = attn_map[rows_range[0]:rows_range[1], cols_range[0]:cols_range[1]]
  return attn_map

def extract_vae_vit(attn_map, head=-1):
  assert head < attn_map.shape[0], f"Head index {head} out of range for attention map with {attn_map.shape[0]} heads."
  if head < 0:
    attn_map = attn_map.mean(dim=0)
  else:
    attn_map = attn_map[head]

  rows_range = (1, boundary_indices["self"][1]-boundary_indices["self"][0])
  cols_range_vae = (boundary_indices["vae"][0]+1, boundary_indices["vae"][1])
  cols_range_vit = (boundary_indices["vit"][0]+1, boundary_indices["vit"][1])
  # Combine the two column ranges
  attn_map = torch.cat([(attn_map[rows_range[0]:rows_range[1], cols_range_vae[0]:cols_range_vae[1]]), 
                       (attn_map[rows_range[0]:rows_range[1], cols_range_vit[0]:cols_range_vit[1]])], dim=1)
  return attn_map
  attn_map = attn_map[:, 48:3241]
  
  cols_to_remove = [0, 1377, 1378, 3193]
  rows_to_remove = [0, 1377]

  num_cols = attn_map.shape[1]
  num_rows = attn_map.shape[0]
  cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]
  return attn_map

def extract_vae_vit_und_full_head(attn_map):
  rows_range = (0, 1)
  cols_range_vae = (boundary_indices["vae"][0]+1, boundary_indices["vae"][1])
  cols_range_vit = (boundary_indices["vit"][0]+1, boundary_indices["vit"][1])
  # Combine the two column ranges
  attn_map = torch.cat([(attn_map[:, rows_range[0]:rows_range[1], cols_range_vae[0]:cols_range_vae[1]]), 
                       (attn_map[:, rows_range[0]:rows_range[1], cols_range_vit[0]:cols_range_vit[1]])], dim=-1)
  return attn_map

def kv_extract_vae_vit(kv, head=-1):
  assert head < kv.shape[0], f"Head index {head} out of range for attention map with {kv.shape[0]} heads."
  if head < 0:
    kv = kv.mean(dim=0)
  else:
    kv = kv[head]
  kv = kv[boundary_indices["vae_vit"][0]:boundary_indices["vae_vit"][1]+1, :]

  # rows_to_remove = [0, 1377, 1378, 3193]

  # num_rows = attn_map.shape[0]
  # rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  # attn_map = attn_map[rows_to_keep, :]
  return kv

def kv_extract_global(kv, head=-1):
  assert head < kv.shape[0], f"Head index {head} out of range for attention map with {kv.shape[0]} heads."
  if head < 0:
    kv = kv.mean(dim=0)
  else:
    kv = kv[head]
  return kv
  
def gen_vae_vit_attn_map(load_dir = "attn_probs_qkv_dump", is_truncate = False, min_threshold = 4e-5, heads_to_plot = None, name=""):
  if heads_to_plot is None:
    heads_to_plot = list(range(-1, 28))

  fn = extract_vae_vit
  layer_idxes = [0, 5, 10, 15, 20, 25, 27]
  timestep = 20
  elem = 'attn_probs'
  
  # --- 提前创建好 norm 对象和 cmap ---
  from matplotlib.colors import Normalize, PowerNorm, LogNorm
  cmap_to_use = "inferno_r" # 统一颜色图
  vmin, vmax = None, None

  if is_truncate:
      # 对于截断的情况，我们使用 LogNorm。
      # 假设一个合理的 vmax，例如 0.01 或 0.1，这需要根据数据分布来调整
      # 也可以通过采样一小部分数据来估算一个更合理的 vmax
      vmax_estimate = 0.01 
      norm_to_use = LogNorm(vmin=min_threshold, vmax=vmax_estimate)
  else:
      # --- 计算全局 vmin 和 vmax ---
      local_vmins = []
      local_vmaxs = []
      low_percentile = 1.0
      high_percentile = 99.0
      for layer_idx in range(layer_idxes[0], layer_idxes[-1]+1):
        tmp_file = f"./{load_dir}/gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
        all_entries = torch.load(tmp_file)
        attn_map = all_entries[0][elem]
      
        attn_map = fn(attn_map)
        flat_attn = attn_map.flatten()
        sample_size = min(1_000_000, flat_attn.numel())  # 最多取1M个样本
        indices = torch.randperm(flat_attn.numel(), device=flat_attn.device)[:sample_size]
        sample = flat_attn[indices]

        # local_vmins.append(torch.min(attn_map.flatten()).item())
        # local_vmaxs.append(torch.max(attn_map.flatten()).item())
        local_vmins.append(torch.quantile(sample.float(), low_percentile / 100.0).item())
        local_vmaxs.append(torch.quantile(sample.float(), high_percentile / 100.0).item())
    
        # local_vmins.append(torch.min(attn_map.flatten()).item())
        # local_vmaxs.append(torch.max(attn_map.flatten()).item())
        # local_vmins.append(torch.quantile(attn_map.float(), low_percentile / 100.0))
        # local_vmaxs.append(torch.quantile(attn_map.float(), high_percentile / 100.0))
      vmin = torch.tensor(local_vmins).mean().item()
      vmax = torch.tensor(local_vmaxs).mean().item()
      print(f"Overall vmin: {vmin}\nOverall vmax: {vmax}")
      norm_to_use = Normalize(vmin=vmin, vmax=vmax)

  postfix0 = "thredshold" if is_truncate else ""
  
  for layer_idx in range(layer_idxes[0], layer_idxes[-1]+1):
      tmp_file = f"./{load_dir}/gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
      all_entries = torch.load(tmp_file)
      print(f"Processing: {tmp_file}")

      sns.set_theme(style="darkgrid")
      
      num_heads = len(heads_to_plot)
      ncols = 6
      nrows = (num_heads + ncols - 1) // ncols
      fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
      fig.suptitle(f'Attention Heads for Layer {layer_idx}, Timestep {timestep}', fontsize=24)
      axes_flat = axes.flatten()

      for i, head in enumerate(heads_to_plot):
          ax = axes_flat[i]
          entry = all_entries[0]
          attn_map = entry[elem]
          print(f"Original Tensor Shape: {attn_map.shape}")
          attn_map = fn(attn_map, head=head)
          print(f"Processed Tensor Shape: {attn_map.shape}")
          
          if is_truncate:
            attn_map = torch.where(attn_map < min_threshold, torch.tensor(0.0, device=attn_map.device), attn_map)
          
          postfix1 = f"Head {head}" if head >=0 else "Mean Head"
          
          print(f"Plotting {postfix1} for Layer {layer_idx}")
          # --- 调用修改后的 attention_plot 函数 ---
          attention_plot(attn_map, ax=ax, title=postfix1,
                         vmin=vmin, vmax=vmax, 
                         norm=norm_to_use,      # 传入统一的 norm 对象
                         cmap=cmap_to_use)      # 传入统一的 cmap

      for i in range(num_heads, len(axes_flat)):
          axes_flat[i].set_visible(False)
      # --- 添加一个全局颜色条 ---
      sm = plt.cm.ScalarMappable(cmap=cmap_to_use, norm=norm_to_use)
      sm.set_array([])
      fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=20, label="Attention Score")

      # --- 保存整个大图 ---
      figure_path = f"plot/attn_map_vae_vit_{name}{postfix0}/t_{timestep}"
      if not os.path.exists(figure_path):
          os.makedirs(figure_path)
      figure_name = f"layer_{layer_idx}_all_heads.png"
      plt.savefig(os.path.join(figure_path, figure_name), dpi=150)
      plt.close(fig)
      print(f"Saved figure to {os.path.join(figure_path, figure_name)}")

def gen_global_attn_map(load_dir = "attn_probs_qkv_dump", timestep = 20, 
                        is_truncate = False, min_threshold = 4e-5, heads_to_plot = None, 
                        layer_idxes = None, name="", with_box=True):
  if heads_to_plot is None:
    heads_to_plot = list(range(-1, 28))

  fn = extract_global
  if layer_idxes is None:
    layer_idxes = range(0, 28)
  elem = 'attn_probs'
  
  # --- 提前创建好 norm 对象和 cmap ---
  from matplotlib.colors import Normalize, PowerNorm, LogNorm
  cmap_to_use = "inferno_r" # 统一颜色图
  vmin, vmax = None, None

  if True:
      # 对于截断的情况，我们使用 LogNorm。
      # 假设一个合理的 vmax，例如 0.01 或 0.1，这需要根据数据分布来调整
      # 也可以通过采样一小部分数据来估算一个更合理的 vmax
      vmax_estimate = 0.1
      norm_to_use = LogNorm(vmin=min_threshold, vmax=vmax_estimate)
  else:
      # --- 计算全局 vmin 和 vmax ---
      local_vmins = []
      local_vmaxs = []
      low_percentile = 1.0
      high_percentile = 99.0
      for layer_idx in range(layer_idxes[0], layer_idxes[-1]+1):
        tmp_file = f"./{load_dir}/gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
        all_entries = torch.load(tmp_file)
        attn_map = all_entries[0][elem]
      
        attn_map = fn(attn_map)
        flat_attn = attn_map.flatten()
        sample_size = min(1_000_000, flat_attn.numel())  # 最多取1M个样本
        indices = torch.randperm(flat_attn.numel(), device=flat_attn.device)[:sample_size]
        sample = flat_attn[indices]

        # local_vmins.append(torch.min(attn_map.flatten()).item())
        # local_vmaxs.append(torch.max(attn_map.flatten()).item())
        local_vmins.append(torch.quantile(sample.float(), low_percentile / 100.0).item())
        local_vmaxs.append(torch.quantile(sample.float(), high_percentile / 100.0).item())
    
        # local_vmins.append(torch.min(attn_map.flatten()).item())
        # local_vmaxs.append(torch.max(attn_map.flatten()).item())
        # local_vmins.append(torch.quantile(attn_map.float(), low_percentile / 100.0))
        # local_vmaxs.append(torch.quantile(attn_map.float(), high_percentile / 100.0))
      vmin = torch.tensor(local_vmins).mean().item()
      vmax = torch.tensor(local_vmaxs).mean().item()
      print(f"Overall vmin: {vmin}\nOverall vmax: {vmax}")
      norm_to_use = Normalize(vmin=vmin, vmax=vmax)

  postfix0 = "thredshold" if is_truncate else ""
  figure_path = f"plot/attn_map_global{name}{postfix0}/t_{timestep}"
  if not os.path.exists(figure_path):
    os.makedirs(figure_path)
  write_log = f"load_dir: {load_dir}\nis_truncate: {is_truncate}\nmin_threshold: {min_threshold}\n"
  with open(os.path.join(figure_path, f"global_attn_map{name}_log.txt"), 'w') as f:
    f.write(write_log)

  box_coords = None
  box_style = None
  if with_box:
    box_coords = (0, boundary_indices["self"][1]-boundary_indices["self"][0], boundary_indices["input_prompt"][0], boundary_indices["gen_text"][1])
    box_style={'edgecolor':'blue','linewidth':2,'linestyle':'--'}

  for layer_idx in layer_idxes:
      tmp_file = f"./{load_dir}/gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
      all_entries = torch.load(tmp_file)
      print(f"Processing: {tmp_file}")

      sns.set_theme(style="dark")
      
      num_heads = len(heads_to_plot)
      ncols = 6
      nrows = (num_heads + ncols - 1) // ncols
      fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
      fig.suptitle(f'Attention Heads for Layer {layer_idx}, Timestep {timestep}', fontsize=24)
      axes_flat = axes.flatten()

      for i, head in enumerate(heads_to_plot):
          ax = axes_flat[i]
          entry = all_entries[0]
          attn_map = entry[elem]
          print(f"Original Tensor Shape: {attn_map.shape}")
          attn_map = fn(attn_map, head=head)
          print(f"Processed Tensor Shape: {attn_map.shape}")
          
          if is_truncate:
            attn_map = torch.where(attn_map < min_threshold, torch.tensor(0.0, device=attn_map.device), attn_map)
          
          postfix1 = f"Head {head}" if head >=0 else "Mean Head"
          
          print(f"Plotting {postfix1} for Layer {layer_idx}")
          # --- 调用修改后的 attention_plot 函数 ---
          attention_plot(attn_map, ax=ax, title=postfix1,
                         vmin=vmin, vmax=vmax, 
                         norm=norm_to_use,      # 传入统一的 norm 对象
                         cmap=cmap_to_use, 
                         xlabel="Key", ylabel="Query",
                         tick_density=200,
                         box_coords=box_coords, box_style=box_style)      # 传入统一的 cmap

      for i in range(num_heads, len(axes_flat)):
          axes_flat[i].set_visible(False)
      # --- 添加一个全局颜色条 ---
      sm = plt.cm.ScalarMappable(cmap=cmap_to_use, norm=norm_to_use)
      sm.set_array([])
      fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=20, label="Attention Score")

      # --- 保存整个大图 ---
      figure_name = f"layer_{layer_idx}_all_heads.png"
      plt.savefig(os.path.join(figure_path, figure_name), dpi=150)
      plt.close(fig)
      print(f"Saved figure to {os.path.join(figure_path, figure_name)}")

def und_vae_vit_attn_map(load_dir = "attn_probs_qkv_dump", heads_to_plot = None, name=""):
  if heads_to_plot is None:
    heads_to_plot = list(range(-1, 28))

  fn = extract_vae_vit_und_full_head
  layer_idxes = range(0, 28)
  # layer_idxes = [0, 5, 10, 15, 20, 25, 27]
  # layer_idxes = [0, 5]
  timesteps = list(range(0, boundary_indices["gen_text"][1]-boundary_indices["gen_text"][0]))
  print(timesteps)
  elem = 'attn_probs'
  
  # --- 提前创建好 norm 对象和 cmap ---
  from matplotlib.colors import Normalize, PowerNorm, LogNorm
  cmap_to_use = "inferno_r" # 统一颜色图
  vmin, vmax = None, None

  # 对于截断的情况，我们使用 LogNorm。
  # 假设一个合理的 vmax，例如 0.01 或 0.1，这需要根据数据分布来调整
  # 也可以通过采样一小部分数据来估算一个更合理的 vmax
  vmax_estimate = 0.1
  norm_to_use = LogNorm(vmin=1e-8, vmax=vmax_estimate)

  postfix0 = ""

  all_layers = []
  
  for layer_idx in layer_idxes:
  # for layer_idx in layer_idxes:
      all_timesteps = []
      for timestep in timesteps:
        tmp_file = f"./{load_dir}/und_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
        print(f"Processing: {tmp_file}")
        all_entries = torch.load(tmp_file)
        entry = all_entries[0]
        attn_map = entry[elem]
        print(f"Original Tensor Shape: {attn_map.shape}")
        attn_map = fn(attn_map)
        print(f"Processed Tensor Shape: {attn_map.shape}")
        # attn_map = torch.nn.functional.pad(attn_map, (0, len(timesteps)-1-timestep, 0, 0), "constant", 0)
        all_timesteps.append(attn_map)
      all_timesteps_stacked = torch.cat(all_timesteps, dim=1)
      print(f"Layer {layer_idx} Stacked Tensor Shape: {all_timesteps_stacked.shape}")
      all_layers.append(all_timesteps_stacked)

  print(len(all_layers))
  # for layer_idx in range(layer_idxes[0], layer_idxes[-1]+1):
  for layer_idx in range(0, len(layer_idxes)):
      all_heads = all_layers[layer_idx]
      print(f"Processing Layer: {layer_idx}")
      sns.set_theme(style="darkgrid")
      
      num_heads = len(heads_to_plot)
      ncols = 6
      nrows = (num_heads + ncols - 1) // ncols
      fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
      fig.suptitle(f'Und. Attention Heads for Layer {layer_idx}', fontsize=24)
      axes_flat = axes.flatten()

      for i, head in enumerate(heads_to_plot):
          ax = axes_flat[i]
          attn_map = all_heads[head] if head >=0 else all_heads.mean(dim=0)
          print(f"Processed Tensor Shape: {attn_map.shape}")
          
          
          postfix1 = f"Head {head}" if head >=0 else "Mean Head"
          
          print(f"Plotting {postfix1} for Layer {layer_idx}")
          # --- 调用修改后的 attention_plot 函数 ---
          attention_plot(attn_map, ax=ax, title=postfix1,
                         vmin=vmin, vmax=vmax, 
                         norm=norm_to_use,      # 传入统一的 norm 对象
                         cmap=cmap_to_use,
                         xlabel="K Index", ylabel="Time Step")      # 传入统一的 cmap

      for i in range(num_heads, len(axes_flat)):
          axes_flat[i].set_visible(False)
      # --- 添加一个全局颜色条 ---
      sm = plt.cm.ScalarMappable(cmap=cmap_to_use, norm=norm_to_use)
      sm.set_array([])
      fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=20, label="Attention Score")

      # --- 保存整个大图 ---
      figure_path = f"plot/und_attn_map_global{name}{postfix0}/all_timesteps"
      if not os.path.exists(figure_path):
          os.makedirs(figure_path)
      figure_name = f"layer_{layer_idx}_all_heads.png"
      plt.savefig(os.path.join(figure_path, figure_name), dpi=150)
      plt.close(fig)
      print(f"Saved figure to {os.path.join(figure_path, figure_name)}")

def kvcache_global_value_map(load_dir="attn_probs_qkv_dump", elem='k', name='', heads_to_plot=None):
  if heads_to_plot is None:
    heads_to_plot = list(range(-1, 28))

  fn = kv_extract_global
  # layer_idxes = [0, 5, 10, 15, 20, 25, 27]
  layer_idxes = range(0, 28)
  timestep = 20
  if elem not in ['k', 'v']:
    raise ValueError(f"Invalid elem: {elem}. Must be 'k' or 'v'.")

  # --- 提前创建好 norm 对象和 cmap ---
  from matplotlib.colors import Normalize, PowerNorm, LogNorm
  cmap_to_use = "inferno_r" # 统一颜色图
  vmin, vmax = None, None

  # vmax_estimate = 0.1
  # norm_to_use = LogNorm(vmin=1e-8, vmax=vmax_estimate)
  norm_to_use = None

  figure_path = f"plot/{elem}_value_map_global_{name}/t_{timestep}"
  if not os.path.exists(figure_path):
    os.makedirs(figure_path)
  write_log = f"load_dir: {load_dir}\nelem: {elem}\nname: {name}\n"
  with open(os.path.join(figure_path, f"{elem}_value_map_global_{name}log.txt"), 'w') as f:
    f.write(write_log)
  
  for layer_idx in layer_idxes:
      tmp_file = f"./{load_dir}/gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
      all_entries = torch.load(tmp_file)
      print(f"Processing: {tmp_file}")

      sns.set_theme(style="dark")
      
      num_heads = len(heads_to_plot)
      ncols = 6
      nrows = (num_heads + ncols - 1) // ncols
      fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
      fig.suptitle(f'Attention Heads for Layer {layer_idx}, Timestep {timestep}', fontsize=24)
      axes_flat = axes.flatten()

      for i, head in enumerate(heads_to_plot):
          ax = axes_flat[i]
          entry = all_entries[0]
          kv = entry[elem]
          print(f"Original Tensor Shape: {kv.shape}")
          kv = fn(kv, head=head)
          print(f"Processed Tensor Shape: {kv.shape}")
          data_aspect_ratio = kv.shape[1] / kv.shape[0]

          postfix1 = f"Head {head}" if head >=0 else "Mean Head"
          
          print(f"Plotting {postfix1} for Layer {layer_idx}")
          # --- 调用修改后的 attention_plot 函数 ---
          kv_plot(kv, ax=ax, title=postfix1,
                         vmin=vmin, vmax=vmax, 
                         norm=norm_to_use,      # 传入统一的 norm 对象
                         cmap=cmap_to_use, 
                         xlabel="Dim", ylabel="Key",
                         tick_density=100, square=False,
                         aspect_ratio=data_aspect_ratio)      # 传入统一的 cmap

      for i in range(num_heads, len(axes_flat)):
          axes_flat[i].set_visible(False)
      # --- 添加一个全局颜色条 ---
      sm = plt.cm.ScalarMappable(cmap=cmap_to_use, norm=norm_to_use)
      sm.set_array([])
      fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=20, label="Attention Score")

      # --- 保存整个大图 ---
      figure_name = f"layer_{layer_idx}_all_heads.png"
      plt.savefig(os.path.join(figure_path, figure_name), dpi=150)
      plt.close(fig)
      print(f"Saved figure to {os.path.join(figure_path, figure_name)}")

if __name__ == "__main__":
  # gen_vae_vit_attn_map(load_dir = "attn_probs_qkv_dump_new", is_truncate = True, min_threshold = 4e-5, heads_to_plot=None, name="new_")
  # gen_vae_vit_attn_map(load_dir = "attn_probs_qkv_dump", is_truncate = True, min_threshold = 4e-5, heads_to_plot=None)
  # und_vae_vit_attn_map(load_dir = "attn_probs_qkv_dump_octupusy_thredshold", heads_to_plot=None)
  # gen_global_attn_map(load_dir = "attn_probs_qkv_dump_octupusy_thredshold", is_truncate = False, min_threshold = 4e-5, heads_to_plot=None, name='_octupusy_thredshold_with_frame')
  # kvcache_global_value_map(load_dir="attn_probs_qkv_dump_octupusy_thredshold", elem='k', name='octupusy_', heads_to_plot=None)
  parser = argparse.ArgumentParser(description="Generate Attention Map Visualizations")
  parser.add_argument('--mode', type=str, choices=['gen_vae_vit', 'gen_global', 'und_vae_vit', 'kvcache_global_value'], required=False, default='gen_global',
                      help="Mode of operation: 'gen_vae_vit', 'gen_global', 'und_vae_vit', 'kvcache_global_value'")
  parser.add_argument('--load_dir', type=str, default='attn_probs_qkv_dump',
                      help="Directory to load attention probabilities from")
  parser.add_argument('--is_truncate', action='store_true',
                      help="Whether to truncate attention values below a threshold")
  parser.add_argument('--min_threshold', type=float, default=4e-5,
                      help="Minimum threshold for truncation")
  parser.add_argument('--name', type=str, default='',
                      help="Name suffix for saving plots")
  parser.add_argument('--timestep', type=int, default=20,
                      help="Timestep to process")
  parser.add_argument('--layer_base', type=int, default=0,
                      help="Starting layer index")
  parser.add_argument('--layer_bias', type=int, default=28,
                      help="Number of layers to process")
  args = parser.parse_args()
  if args.mode == 'gen_vae_vit':
      gen_vae_vit_attn_map(load_dir=args.load_dir, is_truncate=args.is_truncate,
                           min_threshold=args.min_threshold, name=args.name)
  elif args.mode == 'gen_global':
      layer_idxes = list(range(args.layer_base, args.layer_base + args.layer_bias))
      gen_global_attn_map(load_dir=args.load_dir, timestep=args.timestep,
                          is_truncate=args.is_truncate, min_threshold=args.min_threshold,
                          layer_idxes=layer_idxes, name=args.name)
  elif args.mode == 'und_vae_vit':
      und_vae_vit_attn_map(load_dir=args.load_dir, name=args.name)
  elif args.mode == 'kvcache_global_value':
      kvcache_global_value_map(load_dir=args.load_dir, elem='k', name=args.name)
