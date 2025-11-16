#!/bin/bash

# --- Configuration ---
# 定义要搜索的 threshold 和 group_size 列表
# 您可以根据需要向这些数组中添加任意数量的值
# THRESHOLDS=(4e-7 4e-6 1e-5 2e-5 4e-5 8e-5 1e-4 4e-4 8e-4 1e-3)
# GROUP_SIZES=(8 10 14 16 32 64)
# THRESHOLDS=(4e-7 4e-6 1e-5 2e-5)
# GROUP_SIZES=(8 10 14 16)
# THRESHOLDS=(4e-4)
THRESHOLDS=(4e-5 1e-4 4e-4 8e-4 1e-3)
GROUP_SIZES=(16)
# GROUP_SIZES=(10)
# --- 1. 定义 Threshold (浮点数) 的搜索范围 ---
# 使用 awk 来生成浮点数序列，支持科学记数法
# THRESH_START=1e-5
# THRESH_END=4e-4
# THRESH_STEP=1e-5 # 步长
# # awk 命令解释:
# # BEGIN { ... } : 在处理任何输入之前执行的代码块
# # for (i=start; i<=end; i+=step) : C 风格的 for 循环
# # printf "%.0e ", i : 以科学记数法格式化数字 (例如 1e-05)，并用空格分隔
# THRESHOLDS=($(awk -v start="$THRESH_START" -v end="$THRESH_END" -v step="$THRESH_STEP" 'BEGIN { for (i=start; i<=end; i+=step) printf "%.0e ", i }'))


# --- 2. 定义 Group Size (整数) 的搜索范围 ---
# 使用 Bash 的 C 风格 for 循环来生成整数序列
# GSIZE_START=8
# GSIZE_END=64
# GSIZE_STEP=8 # 步长

# GROUP_SIZES=() # 初始化一个空数组
# for (( i=$GSIZE_START; i<=$GSIZE_END; i+=$GSIZE_STEP )); do
#   GROUP_SIZES+=($i)
# done

# 定义主输出目录和最终的汇总 CSV 文件
MAIN_OUTPUT_DIR="outputs/profile_results/grid_search"
MAIN_OUTPUT_DIR=$PWD/$MAIN_OUTPUT_DIR
timestamp=$(date +"%Y%m%d_%H%M%S")
MAIN_OUTPUT_DIR=$MAIN_OUTPUT_DIR\_$timestamp
echo "Main Output Directory: $MAIN_OUTPUT_DIR"
SUMMARY_CSV="$MAIN_OUTPUT_DIR/grid_search_summary.csv"

# --- Initialization ---
# 创建主输出目录
mkdir -p "$MAIN_OUTPUT_DIR"

# 在开始搜索前，创建或清空汇总 CSV 文件，并写入表头
# 这确保了每次运行网格搜索都会生成一个全新的结果文件
echo "threshold,group_size,avg_semantics,avg_quality,avg_overall,avg_vae_vit_sparsity,avg_self_attn_sparsity,avg_cfg_text_similarity,avg_cfg_img_similarity" > "$SUMMARY_CSV"

total_start_time=$SECONDS

# --- Grid Search Loop ---
# 使用嵌套循环遍历所有 threshold 和 group_size 的组合
for threshold in "${THRESHOLDS[@]}"; do
	for gsize in "${GROUP_SIZES[@]}"; do

		run_start_time=$SECONDS
			
		echo "======================================================================"
		echo "Starting run with Threshold: $threshold, Group Size: $gsize"
		echo "======================================================================"

		# 为当前参数组合定义一个唯一的输出目录
		RUN_OUTPUT_DIR="$MAIN_OUTPUT_DIR/thresh_${threshold}_gsize_${gsize}"
		METRICS_JSON_PATH="$RUN_OUTPUT_DIR/summary_metrics.json"
		DEVICE_BASE=0
		SHUFFLE_BASE=16
		N_GPU=8

		attn_backend="naive_sparse_quant_cfg"
		# attn_backend="naive_cal_ave_self_attn_score"
		vae_vit_sparse="--vae_vit_sparse"
		self_attn_sparse="--self_attn_sparse"
		use_custom_mlp="--use_custom_mlp"
		mlp_use_similarity="--mlp_use_similarity"
		use_quantized_mlp_und_w=--quantized_mlp_und_w
		use_quantized_mlp_gen_w=--quantized_mlp_gen_w
		# use_full_head_similarity="--use_full_head_similarity"


		# 调用基础脚本执行实验，并将输出目录、threshold 和 group_size 作为参数传入
		# 假设 gedit_search_base.sh 已经被修改为可以接收这三个参数
		./profile_scripts/gedit_search_base.sh "$RUN_OUTPUT_DIR" "$threshold" "$gsize" \
			"$METRICS_JSON_PATH" "$DEVICE_BASE" "$SHUFFLE_BASE" "$N_GPU" \
			"$attn_backend" "$vae_vit_sparse" "$self_attn_sparse" \
			"$use_custom_mlp" "$mlp_use_similarity" "$use_full_head_similarity" \
			"$use_quantized_mlp_und_w" "$use_quantized_mlp_gen_w"

		# 检查基础脚本是否成功执行 (可选但推荐)
		if [ $? -ne 0 ]; then
			echo "ERROR: gedit_search_base.sh failed for Threshold: $threshold, Group Size: $gsize"
			continue # 跳过当前失败的组合，继续下一个
		fi

		run_end_time=$SECONDS
		duration=$((run_end_time - run_start_time))
		minutes=$((duration / 60))
		seconds=$((duration % 60))
		# echo "----------------------------------------------------------------------"
		# echo "Run finished. Collecting results for Threshold: $threshold, Group Size: $gsize"
		# echo "----------------------------------------------------------------------"
		# 写入到日志文件（追加），并同时输出到控制台
		LOG_FILE="$RUN_OUTPUT_DIR/run.log"
		mkdir -p "$(dirname "$LOG_FILE")"
		echo "Run finished at: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a "$LOG_FILE"
		echo "Duration: ${minutes} minutes ${seconds} seconds (${duration} seconds)" | tee -a "$LOG_FILE"
		echo "Threshold: $threshold, Group Size: $gsize" | tee -a "$LOG_FILE"
		echo "Shuffle Base: $SHUFFLE_BASE, Device Base: $DEVICE_BASE, N_GPU: $N_GPU" | tee -a "$LOG_FILE"
		echo "----------------------------------------" | tee -a "$LOG_FILE"

		# 调用 Python 脚本来分析本次运行的结果，并追加到汇总 CSV 文件中
		# 参数：1. 阈值, 2. 组大小, 3. 本次运行的输出目录, 4. 汇总CSV文件的路径
		python ./profile/grid_search/collect_grid_search_results.py \
			--threshold "$threshold" \
			--group_size "$gsize" \
			--metrics_json_path "$METRICS_JSON_PATH" \
			--summary_csv "$SUMMARY_CSV"

	done
done

echo "======================================================================"
echo "Grid search completed."
echo "Final results are stored in: $SUMMARY_CSV"
echo "======================================================================"