# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

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

# run this script at the root of the project folder
pip install httpx==0.23.0
pip install openai==1.87.0
pip install datasets
pip install megfile

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6,7

N_GPU=2  # Number of GPU used in for the evaluation
N_EVAL=-1  # Number of images to be evaluated
DEVICE_BASE=6
MODEL_PATH="./models/BAGEL-7B-MoT"
OUTPUT_DIR="./outputs/gedit_results"
timestamp=$(date + "%Y%m%d_%H%M%S")
OUTPUT_DIR=$OUTPUT_DIR\_$timestamp
GEN_DIR="$OUTPUT_DIR/gen_image"
LOG_DIR="$OUTPUT_DIR/logs"
EVAL_BACKBONE="gemini"

# AZURE_ENDPOINT="https://azure_endpoint_url_you_use"  # set up the azure openai endpoint url
# AZURE_OPENAI_KEY=""  # set up the azure openai key
API_KEY="AIzaSyAj3vjB9PrPDNPkAd2cSgxgkccVaZI_CCM"
N_GPT_PARALLEL=10

threshold=4e-5
attn_backend="navie_sparse_quant"
sparse_gsize=10
vae_vit_sparse=--vae_vit_sparse
self_attn_sparse=--self_attn_sparse

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
for ((i=0; i<$N_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=$((DEVICE_BASE + i)) \ 
		python3 eval/gen/gedit/gen_images_gedit.py --model_path "$MODEL_PATH" \
		 --output_dir "$GEN_DIR"  --shard_id $i --total_shards "$N_GPU" --device 0 \
		 --eval_num "$N_EVAL" --use_think 2>&1 \
		 --threshold $threshold --attn_backend $attn_backend \
		 --sparse_gsize $sparse_gsize \ 
		 $vae_vit_sparse $self_attn_sparse | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
done

wait
echo "Image Generation Done"


# # ---------------------
# #    GPT Evaluation
# # ---------------------
cd eval/gen/gedit
python test_gedit_score_googleaistudio.py \
	--save_path "$OUTPUT_DIR" \
	--api_keys "$API_KEY" \
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
	--backbone "$EVAL_BACKBONE"
wait
echo "Results Printed"

