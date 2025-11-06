#!/bin/bash

# ==============================================================================
# Bash script to launch the VAE vs. Self-Attention scale analysis.
# ==============================================================================

# --- 用户可配置参数 ---

# 1. 数据加载路径 (必需)
#    请将此路径修改为您存放 attention map .pt 文件的目录。
#    例如: "attn_probs_qkv_dump/your_experiment_name"
LOAD_DIR="./attn_probs_qkv_dump"

# 2. 结果保存路径 (可选)
#    分析结果（图表和日志）将保存在此目录。
SAVE_DIR="./plot/scale_analysis_output"

# 3. 分析的层范围 (可选, 默认为 "0,28")
#    格式为 "起始层,结束层" (不包含结束层)。
LAYER_RANGE="0,1"

# 4. 分析的时间步范围 (可选, 默认为 "0,50")
#    格式为 "起始步,结束步" (不包含结束步)。
TIMESTEP_RANGE="0,1"

# --- 执行命令 ---

# 检查 LOAD_DIR 是否已设置
# if [ "$LOAD_DIR" == "/path/to/your/attn_probs_qkv_dump" ]; then
#     echo "错误: 请在脚本中修改 LOAD_DIR 变量，使其指向您的数据目录。"
#     exit 1
# fi

echo "================================================="
echo "开始执行 VAE-Self Attention Scale 分析"
echo "加载目录: $LOAD_DIR"
echo "保存目录: $SAVE_DIR"
echo "分析层范围: $LAYER_RANGE"
echo "分析时间步范围: $TIMESTEP_RANGE"
echo "================================================="

# 启动 Python 分析脚本
# 使用 \ 来换行，使命令更清晰
python profile/plot_vae_self_scale.py \
    --load_dir "$LOAD_DIR" \
    --save_dir "$SAVE_DIR" \
    --layer_range "$LAYER_RANGE" \
    --timestep_range "$TIMESTEP_RANGE"

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "================================================="
    echo "分析完成！结果已保存至: $SAVE_DIR"
    echo "================================================="
else
    echo "================================================="
    echo "分析过程中出现错误。"
    echo "================================================="
fi