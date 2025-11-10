import argparse
import json
import csv
import os

def main():
    parser = argparse.ArgumentParser(description="Collect grid search results from a metrics JSON file.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value for this run.")
    parser.add_argument("--group_size", type=int, required=True, help="Group size value for this run.")
    parser.add_argument("--metrics_json_path", type=str, required=True, help="Path to the JSON file containing metrics for the run.")
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to the final summary CSV file.")
    args = parser.parse_args()

    try:
        with open(args.metrics_json_path, 'r') as f:
            metrics = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing metrics JSON file {args.metrics_json_path}: {e}")
        # 创建一个空的 metrics 字典，以写入 NaN 值，避免中断整个网格搜索
        metrics = {}

    # 3. 将参数和解析出的指标写入总的 CSV 文件
    try:
        with open(args.summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            # 写入顺序必须与 grid_search_main.sh 中创建的表头一致
            writer.writerow([
                args.threshold,
                args.group_size,
                metrics.get('avg_semantics'),
                metrics.get('avg_quality'),
                metrics.get('avg_overall'),
                metrics.get('avg_vae_vit_sparsity'),
                metrics.get('avg_self_attn_sparsity'),
                metrics.get('avg_cfg_text_similarity'),
                metrics.get('avg_cfg_img_similarity'),
            ])
        print(f"Successfully appended results for thresh={args.threshold}, gsize={args.group_size} to {args.summary_csv}")
    except IOError as e:
        print(f"Error writing to summary CSV file {args.summary_csv}: {e}")

if __name__ == "__main__":
    main()