# 1. 定义一个清理函数
#    这个函数会在接收到信号时被调用
cleanup() {
    echo -e "\n\nCtrl+C detected. Terminating all child processes..."
    # 使用 pkill 来杀死所有由这个脚本启动的 Python 子进程
    # -P $$ 会查找父进程ID是当前脚本ID($$)的进程
    pkill -P $$
    echo "All child processes terminated."
    # 退出脚本
    exit 1
}

# 2. 设置陷阱 (trap)
#    当脚本接收到 SIGINT (Ctrl+C) 或 SIGTERM (来自 kill 命令) 信号时，
#    调用我们定义的 cleanup 函数。
trap cleanup SIGINT SIGTERM

# MODE="gen_global"
MODE="cfg_similarity"
# MODE="gen_global_special_tokens"
# TIME_STEPS=10
TIME_STEPS_S=0
TIME_STEPS_E=9
TOTAL_LAYERS=28
LAYER_SLICES=1
LAYER_BIAS=$((TOTAL_LAYERS / LAYER_SLICES))
# LOAD_DIR="attn_probs_qkv_dump_octupusy_thredshold"
# LOAD_DIR="attn_probs_qkv_dump"
# LOAD_DIR="q_cfg_dump_octupusy_naive_threshold_4e-5"
LOAD_DIR="q_cfg_dump_octupusy_naive_threshold_4e-5"
# LOAD_DIR="attn_probs_mask_sparse_dump_octupusy_64_02"
# LOAD_DIR="maps/mlp_octupusy_flash"
# LOAD_DIR="maps/mlp_after_octupusy_flash"
# LOAD_DIR="maps/mlp_before_res_octupusy_flash"
# LOAD_DIR="maps/mlp_before_norm_octupusy_flash"
# LOAD_DIR="maps/mlp_octupusy_flash_order"
# LOAD_DIR="maps/mlp_octupusy_naive_order"
# LOAD_DIR="q_cfg_dump_octupusy_naive"

# PREFIX="mlp"
PREFIX="qkv_attn_probs"
SUFIX=--sufix

# CFG_MODE="mse"
# CFG_MODE="norm_mse"
# NAME="octupusy_cfg_mse_naive_threshold_4e-5_tmp"
# CFG_MODE="group_mse"
# NAME="octupusy_cfg_group_mse_naive_threshold_4e-5_tmp"
# CFG_MODE="group_cosine"
# NAME="octupusy_cfg_group_cosine_naive"
# NAME="octupusy_cfg_group_cosine_naive_threshold_4e-5_tmp"
CFG_MODE="cosine"
# NAME="octupusy_cfg_cosine_naive"
# CFG_MODE="cosine"
# NAME="octupusy_cfg_cosine_naive_threshold_4e-5_tmp"
# CFG_MODE="cosine"
# NAME="octupusy_cfg_mlp_flash"
# NAME="octupusy_cfg_mlp_after_flash"
# NAME="octupusy_cfg_mlp_before_res_flash"
# NAME="octupusy_cfg_mlp_before_norm_flash"
# NAME="octupusy_cfg_mse_mlp_before_norm_flash"
# NAME="octupusy_cfg_mse_mlp_flash"
# NAME="octupusy_cfg_norm_mse_mlp_before_norm_flash"
# NAME="octupusy_cfg_norm_mse_mlp_flash"
# NAME="_octupusy_with_frame"
# NAME="octupusy_cfg_cos_mlp_flash_order"
# NAME="octupusy_cfg_cos_mlp_naive_order"
# NAME="octupusy_cfg_cos_q_naive_maybe_wrong"
NAME="octupusy_cfg_cos_q_naive_threshold_4e-5_maybe_wrong"
MIN_THRESHOLD=4e-5
HEAD_TO_PLOT='[-1, 0, 1, 2, 3, 4, 5, 6, 7]'

set -x
export PYTHONPATH=.

for timestep in $(seq $TIME_STEPS_S $TIME_STEPS_E); do
	for layer_slice in $(seq 0 $((LAYER_SLICES - 1))); do
		# strip=2
		start_layer=$((layer_slice * LAYER_BIAS + strip))
		bias=$((LAYER_BIAS - strip))

		echo "Starting attention map generation for timestep $timestep, layer slice $layer_slice (layers $start_layer to $((start_layer + bias - 1)))"
		python3 ./profile/attention_map.py --timestep $timestep --min_threshold $MIN_THRESHOLD \
			--layer_base $start_layer --layer_bias $bias \
			--load_dir $LOAD_DIR --mode $MODE --name $NAME \
			--heads "$HEAD_TO_PLOT" --cfg_mode $CFG_MODE \
			--prefix $PREFIX $SUFIX &
	done
done

wait

TIME_STEPS_S=10
TIME_STEPS_E=19

for timestep in $(seq $TIME_STEPS_S $TIME_STEPS_E); do
	for layer_slice in $(seq 0 $((LAYER_SLICES - 1))); do
		# strip=2
		start_layer=$((layer_slice * LAYER_BIAS + strip))
		bias=$((LAYER_BIAS - strip))

		echo "Starting attention map generation for timestep $timestep, layer slice $layer_slice (layers $start_layer to $((start_layer + bias - 1)))"
		python3 ./profile/attention_map.py --timestep $timestep --min_threshold $MIN_THRESHOLD \
			--layer_base $start_layer --layer_bias $bias \
			--load_dir $LOAD_DIR --mode $MODE --name $NAME \
			--heads "$HEAD_TO_PLOT" --cfg_mode $CFG_MODE \
			--prefix $PREFIX $SUFIX &
	done
done

wait

TIME_STEPS_S=20
TIME_STEPS_E=29

for timestep in $(seq $TIME_STEPS_S $TIME_STEPS_E); do
	for layer_slice in $(seq 0 $((LAYER_SLICES - 1))); do
		# strip=2
		start_layer=$((layer_slice * LAYER_BIAS + strip))
		bias=$((LAYER_BIAS - strip))

		echo "Starting attention map generation for timestep $timestep, layer slice $layer_slice (layers $start_layer to $((start_layer + bias - 1)))"
		python3 ./profile/attention_map.py --timestep $timestep --min_threshold $MIN_THRESHOLD \
			--layer_base $start_layer --layer_bias $bias \
			--load_dir $LOAD_DIR --mode $MODE --name $NAME \
			--heads "$HEAD_TO_PLOT" --cfg_mode $CFG_MODE \
			--prefix $PREFIX $SUFIX &
	done
done

wait

TIME_STEPS_S=30
TIME_STEPS_E=39

for timestep in $(seq $TIME_STEPS_S $TIME_STEPS_E); do
	for layer_slice in $(seq 0 $((LAYER_SLICES - 1))); do
		# strip=2
		start_layer=$((layer_slice * LAYER_BIAS + strip))
		bias=$((LAYER_BIAS - strip))

		echo "Starting attention map generation for timestep $timestep, layer slice $layer_slice (layers $start_layer to $((start_layer + bias - 1)))"
		python3 ./profile/attention_map.py --timestep $timestep --min_threshold $MIN_THRESHOLD \
			--layer_base $start_layer --layer_bias $bias \
			--load_dir $LOAD_DIR --mode $MODE --name $NAME \
			--heads "$HEAD_TO_PLOT" --cfg_mode $CFG_MODE \
			--prefix $PREFIX $SUFIX &
	done
done

wait

TIME_STEPS_S=40
TIME_STEPS_E=48

for timestep in $(seq $TIME_STEPS_S $TIME_STEPS_E); do
	for layer_slice in $(seq 0 $((LAYER_SLICES - 1))); do
		# strip=2
		start_layer=$((layer_slice * LAYER_BIAS + strip))
		bias=$((LAYER_BIAS - strip))

		echo "Starting attention map generation for timestep $timestep, layer slice $layer_slice (layers $start_layer to $((start_layer + bias - 1)))"
		python3 ./profile/attention_map.py --timestep $timestep --min_threshold $MIN_THRESHOLD \
			--layer_base $start_layer --layer_bias $bias \
			--load_dir $LOAD_DIR --mode $MODE --name $NAME \
			--heads "$HEAD_TO_PLOT" --cfg_mode $CFG_MODE \
			--prefix $PREFIX $SUFIX &
	done
done

echo "All attention map generation tasks started."
wait
echo "All attention map generation tasks completed."