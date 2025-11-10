## ---------------------- Arguments ----------------------
## 1. OUTPUT_DIR
## 2. THRESHOLD
## 3. GROUP_SIZE
## 4. METRICS_JSON_PATH


# Clean up function
cleanup() {
    echo -e "\n\nCtrl+C detected. Terminating all child processes..."
    # Kill all the subprocess
    pkill -P $$
    echo "All child processes terminated."

    exit 1
}
# Set up the trap which will trigger our cleanup function
trap cleanup SIGINT SIGTERM

export PYTHONPATH=.

N_GPU=4  # Number of GPU used in for the evaluation
DEVICE_BASE=0
N_EVAL=-1  # Number of images to be evaluated
MODEL_PATH="./models/BAGEL-7B-MoT"

OUTPUT_DIR=${1:-"/share/liujun/xinhaolee/workspace/mllms/Bagel/outputs/gedit_results_grid"}
THRESHOLD=${2:-"4e-5"}
GROUP_SIZE=${3:-"10"}

# 注意：METRICS_JSON_PATH 依赖于 THRESHOLD 和 GROUP_SIZE，所以它必须在它们之后定义
METRICS_JSON_PATH=${4:-"$OUTPUT_DIR/metrics_threshold_${THRESHOLD}_group_size_${GROUP_SIZE}.json"}

# OUTPUT_DIR=$OUTPUT_DIR\_$THRESHOLD\_$GROUP_SIZE
GEN_DIR="$OUTPUT_DIR/gen_image"
LOG_DIR="$OUTPUT_DIR/logs"

EVAL_BACKBONE="qwen25vl_api"
KEYS_FILE="./qwen_key.txt"

mapfile -t API_KEYS < $KEYS_FILE

N_GPT_PARALLEL=5

threshold=$THRESHOLD
sparse_gsize=$GROUP_SIZE

attn_backend="naive_sparse_quant_cfg"
vae_vit_sparse=--vae_vit_sparse
self_attn_sparse=--self_attn_sparse
# use_custom_mlp=--use_custom_mlp
reorder_method="None"
save_dir="maps/gedit_maps"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$LOG_DIR"


# # ----------------------------
# #    Download GEdit Dataset
# # ----------------------------
python -c "from datasets import load_dataset; dataset = load_dataset('stepfun-ai/GEdit-Bench')"
echo "Dataset Downloaded"


# # ---------------------
# #    Generate Images
# # ---------------------
echo "Arguments: threshold=$threshold, attn_backend=$attn_backend, sparse_gsize=$sparse_gsize, vae_vit_sparse=$vae_vit_sparse, self_attn_sparse=$self_attn_sparse, reorder_method=$reorder_method" > "$LOG_DIR"/generation_args.log
for ((i=0; i<$N_GPU; i++)); do
	CUDA_VISIBLE_DEVICES=$((DEVICE_BASE + i)) \
	python3 profile/gen_images_gedit_gemini.py --model_path "$MODEL_PATH" \
		--output_dir "$GEN_DIR"  --shard_id $i --total_shards "$N_GPU" --device 0 \
		--eval_num "$N_EVAL" --use_think \
		--threshold $threshold --attn_backend $attn_backend \
		--save_dir $save_dir \
		--sparse_gsize $sparse_gsize \
		$vae_vit_sparse $self_attn_sparse \
		$mlp_save $mlp_save_dir \
		--reorder_method $reorder_method \
		$use_custom_mlp 2>&1 | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
done

wait
echo "Image Generation Done"


# # ---------------------
# #    GPT Evaluation
# # ---------------------
cd eval/gen/gedit
python test_gedit_score_googleaistudio.py \
	--save_path "$OUTPUT_DIR" \
	--api_keys "${API_KEYS[@]}" \
	--max_workers "$N_GPT_PARALLEL" \
	--backbone "$EVAL_BACKBONE"
wait
echo "Evaluation Done"


# # --------------------
# #    Print Results
# # --------------------
python calculate_statistics.py \
	--save_path "$OUTPUT_DIR"  \
	--language en \
	--backbone "$EVAL_BACKBONE" \
	--output_json_path "$METRICS_JSON_PATH"
wait
echo "Results Printed"

