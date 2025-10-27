#!/bin/bash

GPUS=(6 7)
NR_TASKS=2

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

if [ $NR_TASKS -ne ${#GPUS[@]} ]; then
		echo "Number of tasks must equal number of GPUs"
		exit 1
fi
TASKS=( $(seq 0 $((NR_TASKS - 1))) )

set -x
export PYTHONPATH=.

for task_id in ${TASKS[@]}; do
	GPU_ID=${GPUS[$task_id]}
	echo "Starting task $task_id on GPU $GPU_ID, total tasks: $NR_TASKS"
	CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./profile/search_inference.py --nr_tasks $NR_TASKS --task_id $task_id &
done

echo "All tasks started."
wait
echo "All tasks completed."