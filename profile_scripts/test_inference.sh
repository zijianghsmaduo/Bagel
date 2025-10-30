#!/bin/bash

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

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=5

# is_save=--is_save
is_truncate=--is_truncate
save_dir="attn_probs_qkv_dump_woman"
threshold=4e-5

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 ./profile/test_inference.py $is_save $is_truncate --save_dir $save_dir --threshold $threshold

echo "is_save: $is_save" >> $save_dir/arg_log.txt
echo "is_truncate: $is_truncate" >> $save_dir/arg_log.txt
echo "save_dir: $save_dir" >> $save_dir/arg_log.txt
echo "threshold: $threshold" >> $save_dir/arg_log.txt

wait