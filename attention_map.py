import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from transformers import AutoConfig
import numpy as np

import glob

load_dir = "attn_probs_dump"
gen_attn_file = glob.glob(f"./{load_dir}/gen_attn_probs_layer_*_ts_*_batch_0.pt")
# und_attn_file = glob.glob(f"./{load_dir}/und_attn_probs_layer_*_ts_*_batch_0.pt")
# print(gen_attn_file)
# print(und_attn_file)

# def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
#                    figure_name='attention_weight.png'):
#     plt.clf()
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.set_theme(font_scale=1.25)
#     hm = sns.heatmap(attention,
#                      cbar=True,
#                      cmap="RdBu_r",
#                      annot=annot,
#                      square=True,
#                      fmt='.2f',
#                      annot_kws={'size': 10},
#                     #  yticklabels=y_texts,
#                     #  xticklabels=x_texts
#                      yticklabels=False,
#                      xticklabels=False
#                      )
#     if os.path.exists(figure_path) is False:
#         os.makedirs(figure_path)
#     plt.savefig(os.path.join(figure_path, figure_name))
#     plt.close()

def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png',
                   vmin=None, vmax=None, symmetric=False, percentile=None):
    plt.clf()
    sns.set_theme(font_scale=1.25)

    arr = attention.detach().cpu().numpy() if hasattr(attention, "detach") else np.asarray(attention)
    if percentile is not None:
      lowp, highp = percentile
      vmin_p = np.percentile(arr, lowp)
      vmax_p = np.percentile(arr, highp)
      vmin = vmin_p if vmin is None else vmin
      vmax = vmax_p if vmax is None else vmax
    
    # # Dynamically adjust figsize based on the shape of the attention matrix
    # rows, cols = arr.shape
    # aspect_ratio = cols / rows
    # base_size = 10  # Base size for the shorter dimension
    # if aspect_ratio > 1:  # Wider than tall
    #     figsize = (base_size * aspect_ratio, base_size)
    # else:  # Taller than wide
    #     figsize = (base_size, base_size / aspect_ratio)
    # fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(arr,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=False,
                     fmt='.2f',
                     annot_kws={'size': 10},
                    #  yticklabels=y_texts,
                    #  xticklabels=x_texts
                     yticklabels=False,
                     xticklabels=False,
                     vmin=vmin, vmax=vmax
                     )
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()

def plot_gen_all():
  save_dir = "attn_figures/gen"
  for i, file in enumerate(gen_attn_file):
    print(f"Processing {i}th file: {file}")
    all_entries = torch.load(file)
    attn_prob = all_entries[0]['data']
    attn_prob = attn_prob.mean(dim=0)

    # Remove the special tokens
    cols_to_remove = [0, 47, 48, 1425, 1426, 3240, 3241, 3253, 3254, 3482, 3483, 4860]
    rows_to_remove = [3483, 4860]
    num_cols = attn_prob.shape[1]
    num_rows = attn_prob.shape[0]
    cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
    rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
    attn_prob = attn_prob[rows_to_keep, :][:, cols_to_keep]
    attention_plot(attn_prob, x_texts=[f'Token_{i}' for i in range(attn_prob.shape[1])],
                    y_texts=[f'Token_{i}' for i in range(attn_prob.shape[0])],
                    figure_path=save_dir+"/all",
                    figure_name=os.path.basename(file).replace('.pt', '.png'))
    # break	# 仅显示第一个

def plot_gen_vae():
  save_dir = "attn_figures/gen"
  for i, file in enumerate(gen_attn_file):
    print(f"Processing {i}th file: {file}")
    all_entries = torch.load(file)
    attn_prob = all_entries[0]['data']
    attn_prob = attn_prob.mean(dim=0)[:, 49:1425]
    attention_plot(attn_prob, x_texts=[f'Token_{i}' for i in range(attn_prob.shape[1])],
                     y_texts=[f'Token_{i}' for i in range(attn_prob.shape[0])],
                     figure_path=save_dir+"/vae",
                     figure_name=os.path.basename(file).replace('.pt', '.png'))
    # break	# 仅显示第一个
# for file in und_attn_file:
#   all_entries = torch.load(file)
#   attn_prob = all_entries[0]['data']
#   print(attn_prob.shape)
#   print(attn_prob)
#   attn_prob = attn_prob.mean(dim=0)[:, 3250:]
#   attention_plot(attn_prob, x_texts=[f'Token_{i}' for i in range(attn_prob.shape[1])],
#                    y_texts=[f'Token_{i}' for i in range(attn_prob.shape[0])],
#                    figure_path=save_dir+"/und",
#                    figure_name=os.path.basename(file).replace('.pt', '.png'))
#   break	# 仅显示第一个

def extract_global(attn_map):
  attn_map = attn_map.mean(dim=0)

  cols_to_remove = [0, 47, 48, 1425, 1426, 3240, 3241, 3253, 3254, 3482, 3483, 4860]
  rows_to_remove = [0, 1377]
  num_cols = attn_map.shape[1]
  num_rows = attn_map.shape[0]
  cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]
  return attn_map

def extract_vae(attn_map):
  attn_map = attn_map.mean(dim=0)
  
  rows_to_remove = [0, 1377]
  num_rows = attn_map.shape[0]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, 49:1425]
  return attn_map

def extract_vit(attn_map):
  attn_map = attn_map.mean(dim=0)
  
  rows_to_remove = [0, 1377]
  num_rows = attn_map.shape[0]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, 1427:3240]
  return attn_map

def extract_vae_vit(attn_map):
  attn_map = attn_map.mean(dim=0)
  attn_map = attn_map[:, 48:3241]
  
  cols_to_remove = [0, 1377, 1378, 3193]
  rows_to_remove = [0, 1377]

  num_cols = attn_map.shape[1]
  num_rows = attn_map.shape[0]
  cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]
  return attn_map

def kv_extract_vae_vit(attn_map):
  attn_map = attn_map.mean(dim=0)
  attn_map = attn_map[48:3241, :]
  
  rows_to_remove = [0, 1377, 1378, 3193]

  num_rows = attn_map.shape[0]
  rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
  attn_map = attn_map[rows_to_keep, :]
  return attn_map

# tmp_file = gen_attn_file[0]
# tmp_file = f"./{load_dir}/gen_attn_probs_layer_15_ts_1_batch_0.pt"
fn = kv_extract_vae_vit

# vmax = 0
# vmin = 1
vmax = None
vmin = None

layer_idxes = [0, 5, 10, 15, 20, 25, 27]
timestep = 20
# for layer_idx in layer_idxes:
#     tmp_file = f"./{load_dir}/gen_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
#     all_entries = torch.load(tmp_file)

#     attn_map = all_entries[0]['data']
#     # attn_map = attn_map.mean(dim=0) # [1:-1, 1:-1]
#     # cols_to_remove = [0, 47, 48, 1425, 1426, 3240, 3241, 3253, 3254, 3482, 3483, 4860]
#     # rows_to_remove = [0, 1377]
#     # num_cols = attn_map.shape[1]
#     # num_rows = attn_map.shape[0]
#     # cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
#     # rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
#     # attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]'
#     attn_map = fn(attn_map)

#     vmax = max(torch.max(attn_map), vmax)
#     vmin = min(torch.min(attn_map), vmin)
# all_data_for_percentiles = []
# local_vmins = []
# local_vmaxs = []

# low_percentile = 1.0
# high_percentile = 99.0

# for layer_idx in layer_idxes:
#     tmp_file = f"./{load_dir}/gen_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
#     all_entries = torch.load(tmp_file)
#     attn_map = all_entries[0]['data']
    
#     attn_map = fn(attn_map)
    
#     local_vmins.append(torch.quantile(attn_map.float(), low_percentile / 100.0))
#     local_vmaxs.append(torch.quantile(attn_map.float(), high_percentile / 100.0))


# vmin = torch.tensor(local_vmins).mean().item()
# vmax = torch.tensor(local_vmaxs).mean().item()

# print(f"Overall vmin: {vmin}, vmax: {vmax}")

load_dir = "attn_probs_qkv_dump"

for layer_idx in layer_idxes:
    tmp_file = f"./{load_dir}/gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt"
    all_entries = torch.load(tmp_file)
    print(tmp_file)

    print(f"文件中总共保存了 {len(all_entries)} 个注意力图。")

    for i, entry in enumerate(all_entries):
        # metadata = entry['metadata']
        attn_map = entry['v']
        print(f"  张量形状: {attn_map.shape}")
        # attn_map = attn_map.mean(dim=0) # [1:-1, 1:-1]

        # attn_map = attn_map[:, 49:1425]
        # attn_map = attn_map[:, 1427:3240]

        # cols_to_remove = [0, 47, 48, 1425, 1426, 3240, 3241, 3253, 3254, 3482, 3483, 4860]
        # rows_to_remove = [0, 1377]
        # num_cols = attn_map.shape[1]
        # num_rows = attn_map.shape[0]
        # cols_to_keep = [i for i in range(num_cols) if i not in cols_to_remove]
        # rows_to_keep = [i for i in range(num_rows) if i not in rows_to_remove]
        # attn_map = attn_map[rows_to_keep, :][:, cols_to_keep]
        attn_map = fn(attn_map)

				
				
        attention_plot(abs(attn_map.float()), x_texts=None,
                    y_texts=None,
                    figure_path = "./",
                    figure_name=f"attn_figures/gen/v/v_{layer_idx}.png", vmin=vmin, vmax=vmax)
        break