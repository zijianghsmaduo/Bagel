import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import json
from typing import Tuple

# 从 attention_map.py 复制过来的边界定义，确保分析区域一致
# 注意：如果您的序列长度或分块有变，请务必更新这里的数值
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

def analyze_attention_scale(load_dir: str, save_dir: str, layer_range: Tuple[int, int], timestep_range: Tuple[int, int]):
    """
    分析并统计 VAE 和 Self-Attention 部分 attention map 数值的 element-wise scale。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_mean_scales = []
    analysis_log = []
    epsilon = 1e-9  # 防止除以零

    # 定义提取区域
    # Query 来自于 self-attention 区域
    query_rows = (0, boundary_indices["self"][1] - boundary_indices["self"][0] + 1)
    # Keys 分别来自于 VAE 和 self-attention 区域
    key_cols_vae = (boundary_indices["vae"][0], boundary_indices["vae"][1] + 1)
    key_cols_self = (boundary_indices["self"][0], boundary_indices["self"][1] + 1)

    # 检查两个区域的形状是否一致
    query_len = query_rows[1] - query_rows[0]
    key_len_vae = key_cols_vae[1] - key_cols_vae[0]
    key_len_self = key_cols_self[1] - key_cols_self[0]
    if query_len != key_len_vae or query_len != key_len_self or key_len_vae != key_len_self:
        print(f"Warning: VAE part shape ({query_len}, {key_len_vae}) and Self part shape ({query_len}, {key_len_self}) do not match. Element-wise analysis might be incorrect.")

    pbar = tqdm(total=(layer_range[1] - layer_range[0]) * (timestep_range[1] - timestep_range[0]), desc="Analyzing Scales")

    for layer_idx in range(layer_range[0], layer_range[1]):
        for timestep in range(timestep_range[0], timestep_range[1]):
            file_path = os.path.join(load_dir, f"gen_qkv_attn_probs_layer_{layer_idx}_ts_{timestep}_batch_0.pt")
            
            if not os.path.exists(file_path):
                pbar.update(1)
                continue

            try:
                attn_data = torch.load(file_path, map_location='cpu')
                attn_map = attn_data[0]['attn_probs']
                # print(f"Processing file: {file_path}, attn_map shape: {attn_map.shape}")
                num_heads = attn_map.shape[0]

                for head_idx in range(num_heads):
                    head_map = attn_map[head_idx]

                    # 提取 VAE 注意力部分和 Self 注意力部分
                    vae_attn_part = head_map[query_rows[0]:query_rows[1], key_cols_vae[0]:key_cols_vae[1]]
                    self_attn_part = head_map[query_rows[0]:query_rows[1], key_cols_self[0]:key_cols_self[1]]

                    vae_attn_part = vae_attn_part.to(torch.float32)
                    self_attn_part = self_attn_part.to(torch.float32)

                    # --- 修改核心逻辑 ---
                    # # 1. 计算 element-wise scale
                    elementwise_scales = self_attn_part / (vae_attn_part + epsilon)

                    # # 2. 计算这些 element-wise scale 的均值和方差
                    mean_of_scales = elementwise_scales.mean().item()
                    var_of_scales = elementwise_scales.var().item()
                    # --- 更稳健的核心逻辑（避免 NaN/Inf / 0/0） ---
                    # vae_clean = torch.nan_to_num(vae_attn_part, nan=0.0, posinf=1e9, neginf=0.0)
                    # self_clean = torch.nan_to_num(self_attn_part, nan=0.0, posinf=1e9, neginf=0.0)
                    # denom = vae_clean + epsilon
                    # denom = denom.clamp_min(epsilon)
  
                    # elementwise_scales = self_clean / denom
                    # elementwise_scales = torch.nan_to_num(elementwise_scales, nan=float('nan'), posinf=float('inf'), neginf=float('-inf'))

                    # print(elementwise_scales.shape)
                    # # 仅对有限值统计
                    # finite_mask = torch.isfinite(elementwise_scales)
                    # if finite_mask.sum() == 0:
                    #     mean_of_scales = float('nan')
                    #     var_of_scales = float('nan')
                    # else:
                    #     vals = elementwise_scales[finite_mask].to(torch.float64)  # 用更高精度统计
                    #     mean_of_scales = float(vals.mean().item())
                    #     var_of_scales = float(vals.var(unbiased=False).item())

                    all_mean_scales.append(mean_of_scales)

                    # 记录日志
                    analysis_log.append({
                        "layer": layer_idx,
                        "timestep": timestep,
                        "head": head_idx,
                        "mean_of_scales": mean_of_scales,
                        "var_of_scales": var_of_scales,
                    })
                
                pbar.update(1)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                pbar.update(1)
                continue
    
    pbar.close()

    # --- 保存分析日志 ---
    log_path = os.path.join(save_dir, "elementwise_scale_analysis_log.json")
    with open(log_path, 'w') as f:
        json.dump(analysis_log, f, indent=2)
    print(f"Analysis log saved to {log_path}")

    # --- 绘制 Scale 分布图 ---
    plt.style.use('seaborn-v0_8-darkgrid')
    # fig, ax = plt.subplots(figsize=(12, 7))
    
    # scales_array = np.array(all_mean_scales)
    # # 过滤掉极端的离群值以便更好地可视化核心分布
    # p_low, p_high = np.percentile(scales_array, [1, 99])
    # filtered_scales = scales_array[(scales_array > p_low) & (scales_array < p_high)]

    # sns.histplot(filtered_scales, bins=100, kde=True, ax=ax)
    
    # mean_scale = np.mean(filtered_scales)
    # median_scale = np.median(filtered_scales)
    # ax.axvline(mean_scale, color='r', linestyle='--', label=f'Mean: {mean_scale:.2f}')
    # ax.axvline(median_scale, color='g', linestyle='-', label=f'Median: {median_scale:.2f}')
    
    # ax.set_title(f'Distribution of Mean Element-wise Scale\n(Mean of [Self-Attn / VAE-Attn])\n(Filtered to 1st-99th percentile)', fontsize=16)
    # ax.set_xlabel('Mean Scale Value', fontsize=12)
    # ax.set_ylabel('Frequency', fontsize=12)
    # ax.legend()
    
    # fig_path = os.path.join(save_dir, "elementwise_scale_distribution.png")
    # plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    # plt.close(fig)
    # print(f"Scale distribution plot saved to {fig_path}")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios':[3,2]})

    scales_array = np.array(all_mean_scales, dtype=float)
    # 丢弃非有限值
    scales_array = scales_array[np.isfinite(scales_array)]
    if scales_array.size == 0:
        print("No finite scale values found. Skipping plot.")
        return

    # 使用更严格的分位数截断以去除极端离群值（可调）
    lower_pct = 0.5
    upper_pct = 99.5
    p_low, p_high = np.percentile(scales_array, [lower_pct, upper_pct])
    filtered_scales = scales_array[(scales_array >= p_low) & (scales_array <= p_high)]
    # 若截断后为空则退回到不截断
    if filtered_scales.size == 0:
        filtered_scales = scales_array
        p_low, p_high = filtered_scales.min(), filtered_scales.max()

    # 主图：线性尺度、截断后的直方图 + KDE
    sns.histplot(filtered_scales, bins=120, kde=True, ax=axes[0], color='C0')
    mean_scale = np.mean(filtered_scales)
    median_scale = np.median(filtered_scales)
    axes[0].axvline(mean_scale, color='r', linestyle='--', label=f'Mean: {mean_scale:.2f}')
    axes[0].axvline(median_scale, color='g', linestyle='-', label=f'Median: {median_scale:.2f}')
    axes[0].set_xlim(p_low, p_high)
    axes[0].set_xlabel('Mean Scale Value (clipped)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].set_title(f'Distribution (clipped {lower_pct}%–{upper_pct}% )')

    # 侧图：对数尺度查看长尾（只展示正值）
    pos_vals = scales_array[scales_array > 0]
    if pos_vals.size > 0:
        sns.histplot(np.log10(pos_vals), bins=120, kde=True, ax=axes[1], color='C1')
        axes[1].set_xlabel('log10(Scale)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Log-scale view (all positive samples)')
    else:
        axes[1].text(0.5, 0.5, 'No positive samples for log view', ha='center')
        axes[1].set_axis_off()

    # 标注被截断的比例
    total = scales_array.size
    kept = filtered_scales.size
    clipped_frac = (total - kept) / total
    fig.suptitle(f'Distribution of Mean Element-wise Scale  — clipped {clipped_frac*100:.2f}% of samples', fontsize=14)

    fig_path = os.path.join(save_dir, "elementwise_scale_distribution_clipped.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Scale distribution plot saved to {fig_path} (clipped {clipped_frac*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the element-wise scale between VAE and Self-Attention map values.")
    parser.add_argument('--load_dir', type=str, required=True, help="Directory to load attention map files from.")
    parser.add_argument('--save_dir', type=str, default='./scale_analysis_output', help="Directory to save the output plots and logs.")
    parser.add_argument('--layer_range', type=str, default="0,28", help="Comma-separated range of layers to analyze (e.g., '0,28').")
    parser.add_argument('--timestep_range', type=str, default="0,50", help="Comma-separated range of timesteps to analyze (e.g., '0,50').")

    args = parser.parse_args()

    try:
        layer_start, layer_end = map(int, args.layer_range.split(','))
        ts_start, ts_end = map(int, args.timestep_range.split(','))
        
        analyze_attention_scale(
            load_dir=args.load_dir,
            save_dir=args.save_dir,
            layer_range=(layer_start, layer_end),
            timestep_range=(ts_start, ts_end)
        )
    except ValueError:
        print("Error: --layer_range and --timestep_range must be in the format 'start,end' (e.g., '0,28').")
