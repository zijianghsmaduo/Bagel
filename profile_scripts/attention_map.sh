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

MODE="gen_global"
TIME_STEPS=10
TIME_STEPS_S=40
TIME_STEPS_E=49
TOTAL_LAYERS=28
LAYER_SLICES=4
LAYER_BIAS=$((TOTAL_LAYERS / LAYER_SLICES))
# LOAD_DIR="attn_probs_qkv_dump_octupusy_thredshold"
LOAD_DIR="attn_probs_qkv_dump"
# LOAD_DIR="attn_probs_qkv_dump_woman"
# NAME="_octupusy_thredshold_with_frame"
NAME="_octupusy_with_frame"
MIN_THRESHOLD=4e-10

set -x
export PYTHONPATH=.

for timestep in $(seq $TIME_STEPS_S $TIME_STEPS_E); do
	for layer_slice in $(seq 0 $((LAYER_SLICES - 1))); do
		# strip=2
		start_layer=$((layer_slice * LAYER_BIAS + strip))
		bias=$((LAYER_BIAS - strip))

		echo "Starting attention map generation for timestep $timestep, layer slice $layer_slice (layers $start_layer to $((start_layer + bias - 1)))"
		python3 ./profile/attention_map.py --timestep $timestep --min_threshold $MIN_THRESHOLD --layer_base $start_layer --layer_bias $bias \
		--load_dir $LOAD_DIR --mode $MODE --name $NAME &
	done
done

echo "All attention map generation tasks started."
wait
echo "All attention map generation tasks completed."