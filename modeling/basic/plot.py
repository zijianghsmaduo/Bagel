import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, LogNorm
import seaborn as sns
from matplotlib.patches import Rectangle


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

def plot_mask_heads(masks, heads=None, out_dir=None, filename=None,
                    ncols=6, cmap='gray_r', tick_density=100, norm="log",
                    box_coords=None, box_style=None, figsize_per_plot=(5,5)):
    """
    Plot boolean/int mask for multiple heads in a grid.
    - masks: tensor or ndarray with shape (H, Lq, Lk) or (num_heads, Hq, Hk)
             boolean True means masked (1); function plots numeric 0/1 image.
    - heads: list of head indices to plot. use -1 in list to plot mean across heads.
             if None, plot all heads (0..H-1).
    - out_dir / filename: optional save location. if provided, save the figure.
    - ncols: number of columns in the grid.
    - cmap: colormap for binary mask (default 'gray_r' : masked=white, kept=black)
    - tick_density: passed to attention_plot for tick thinning.
    - box_coords / box_style: forwarded to attention_plot to draw rectangle.
    - Returns matplotlib.figure.Figure
    """

    # convert to numpy array
    arr = masks.detach().cpu().float().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if arr.ndim != 3:
        raise ValueError(f"mask array must be 3D (heads, rows, cols), got shape {arr.shape}")

    H = arr.shape[0]
    if heads is None:
        heads = list(range(H))
    # support single int
    if isinstance(heads, int):
        heads = [heads]

    num_heads = len(heads)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    # Use Normalize for consistent coloring 0..1
    vmax_estimate = 1.0  # since mask is binary 0/1
    if norm == "linear":
      norm = Normalize(vmin=0.0, vmax=vmax_estimate)
    elif norm == "log":
      norm = LogNorm(vmin=4e-5, vmax=vmax_estimate)

    for i, head in enumerate(heads):
        ax = axes_flat[i]
        if head < 0:
            data = arr.mean(axis=0)  # mean across heads
            title = "Mean Head"
        else:
            if head >= H:
                raise IndexError(f"head index {head} out of range (0..{H-1})")
            data = arr[head].astype(float)
            title = f"Head {head}"

        # call existing attention_plot (keeps styling / boxes)
        attention_plot(data, ax=ax,
                       title=title,
                       vmin=0.0, vmax=1.0,
                       norm=norm,
                       cmap=cmap,
                       tick_density=tick_density,
                       xlabel="Key", ylabel="Query",
                       box_coords=box_coords, box_style=box_style)

    # hide unused axes
    for j in range(num_heads, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # global colorbar (binary 0..1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.flatten().tolist(), shrink=0.6, aspect=20, label="Mask (1=masked)")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            filename = "mask_all_heads.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=150)

    return fig