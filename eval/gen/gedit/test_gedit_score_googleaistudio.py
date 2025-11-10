# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from viescore import VIEScore
import PIL
import os
import megfile
from PIL import Image
from tqdm import tqdm
import json
from datasets import load_dataset
import sys
import csv
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
GROUPS = [
    "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
]

model_name = 'bagel'

def process_single_item(item, vie_score, vie_idx, max_retries=10000):

    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    intersection_exist = item['Intersection_exist']
    sample_prefix = key
    save_path_fullset_source_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}_SRCIMG.png"
    save_path_fullset_result_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}.png"
    
    sparsity_csv_path = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}_sparsity.csv"
    similarity_csv_path = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}_similarity.csv"

    src_image_path = save_path_fullset_source_image
    save_path_item = save_path_fullset_result_image

    vae_vit_sparsity = np.nan
    self_attn_sparsity = np.nan
    cfg_text_similarity = np.nan
    cfg_img_similarity = np.nan

    # --- 新增：读取并解析 sparsity 文件 ---
    try:
        if megfile.smart_exists(sparsity_csv_path):
            with megfile.smart_open(sparsity_csv_path, 'r') as f:
                df_sparsity = pd.read_csv(f)
                # 计算 'cfg_text' 和 'cfg_img' 的平均稀疏度
                # 如果只有其中一种，也能正常工作
                self_relevant_rows = df_sparsity[df_sparsity['cfg_type'].isin(['normal', 'cfg_text', 'cfg_img'])]
                if not self_relevant_rows.empty:
                    self_attn_sparsity = self_relevant_rows['self_attn_sparsity'].mean()
                vae_vit_relevant_rows = df_sparsity[df_sparsity['cfg_type'].isin(['normal', 'cfg_text'])]
                if not vae_vit_relevant_rows.empty:
                    vae_vit_sparsity = vae_vit_relevant_rows['vae_vit_sparsity'].mean()
    except Exception as e:
        print(f"Warning: Could not read or parse sparsity file {sparsity_csv_path}. Error: {e}")

    # --- 新增：读取并解析 similarity 文件 ---
    try:
        if megfile.smart_exists(similarity_csv_path):
            with megfile.smart_open(similarity_csv_path, 'r') as f:
                df_similarity = pd.read_csv(f)
                cfg_txt_relevant_rows = df_similarity[df_similarity['cfg_type'].isin(['cfg_text'])]
                if not cfg_txt_relevant_rows.empty:
                    cfg_text_similarity = cfg_txt_relevant_rows['similarity'].mean()
                cfg_img_relevant_rows = df_similarity[df_similarity['cfg_type'].isin(['cfg_img'])]
                if not cfg_img_relevant_rows.empty:
                    cfg_img_similarity = cfg_img_relevant_rows['similarity'].mean()
    except Exception as e:
        print(f"Warning: Could not read or parse similarity file {similarity_csv_path}. Error: {e}")


    # print(f"vie_idx: {vie_idx}")
    
    for retry in range(max_retries):
        try:
            pil_image_raw =Image.open(megfile.smart_open(src_image_path, 'rb'))
            pil_image_edited = Image.open(megfile.smart_open(save_path_item, 'rb')).convert("RGB").resize((pil_image_raw.size[0], pil_image_raw.size[1]))

            text_prompt = instruction
            score_list = vie_score.evaluate([pil_image_raw, pil_image_edited], text_prompt)
            sementics_score, quality_score, overall_score = score_list


            # print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")
            
            return {
                "source_image": src_image_path,
                "edited_image": save_path_item,
                "instruction": instruction,
                "sementics_score": sementics_score,
                "quality_score": quality_score,
                "intersection_exist" : item['Intersection_exist'],
                "instruction_language" : item['instruction_language'],
                "vae_vit_sparsity": vae_vit_sparsity,
                "self_attn_sparsity": self_attn_sparsity,
                "cfg_text_similarity": cfg_text_similarity,
                "cfg_img_similarity": cfg_img_similarity
            }
        except Exception as e:
            
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # 指数退避：2秒, 4秒, 6秒...
                print(f"Using vie_score index: {vie_idx}. Error processing {save_path_item} (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                # if KEY_INDEX + 1 > len(API_KEYS):
                print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
                return
                # else:
                #   print(f"Switching to next API key due to repeated failures.")
                #   KEY_INDEX += 1
                #   vie_score.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="gemini", choices=["gemini", "gpt4o", "qwen25vl", "qwen25vl_api"])
    parser.add_argument("--api_keys", type=str, required=True, nargs='+', help="Google AI Studio API keys (file paths or direct keys)")
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()
    save_path_dir = args.save_path
    evaluate_group = [model_name]
    backbone = args.backbone

    # For Gemini, we don't need azure_endpoint
    vie_scores = [VIEScore(backbone=backbone, task="tie", key_path=k, azure_endpoint='') for k in args.api_keys]
    print(f"len vie_scores: {len(vie_scores)}")
    dataset = load_dataset("stepfun-ai/GEdit-Bench")['train'].remove_columns(['input_image_raw', 'input_image'])

    for model_name in evaluate_group:
        save_path = os.path.join(save_path_dir, 'gen_image')
        save_path_new = os.path.join(save_path_dir, backbone, "eval_results_new")
        all_csv_list = []  # Store all results for final combined CSV
        
        # Load existing processed samples from final CSV if it exists
        processed_samples = set()
        final_csv_path = os.path.join(save_path_new, f"{model_name}_combined_gpt_score.csv")
        if megfile.smart_exists(final_csv_path):
            with megfile.smart_open(final_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create a unique identifier for each sample
                    sample_key = (row['source_image'], row['edited_image'])
                    processed_samples.add(sample_key)
            print(f"Loaded {len(processed_samples)} processed samples from existing CSV")

        for group_name in GROUPS:
            group_csv_list = []
            group_dataset_list = []  
            for item in tqdm(dataset, desc=f"Processing {model_name} - {group_name}"):
                if item['task_type'] == group_name:
                    group_dataset_list.append(item)
            # Load existing group CSV if it exists
            group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_gpt_score.csv")
            if megfile.smart_exists(group_csv_path):
                with megfile.smart_open(group_csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    group_results = list(reader)
                    group_csv_list.extend(group_results)
            
                print(f"Loaded existing results for {model_name} - {group_name}")
            
            print(f"Processing group: {group_name}")
            print(f"Processing model: {model_name}")
            
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                for i, item in enumerate(group_dataset_list):
                    instruction = item['instruction']
                    key = item['key']
                    instruction_language = item['instruction_language']
                    intersection_exist = item['Intersection_exist']
                    sample_prefix = key
                    save_path_fullset_source_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}_SRCIMG.png"
                    save_path_fullset_result_image = f"{save_path}/fullset/{group_name}/{instruction_language}/{key}.png"

                    if not megfile.smart_exists(save_path_fullset_result_image) or not megfile.smart_exists(save_path_fullset_source_image):
                        print(f"Skipping {sample_prefix}: Source or edited image does not exist")
                        continue

                    # Check if this sample has already been processed
                    sample_key = (save_path_fullset_source_image, save_path_fullset_result_image)
                    exists = sample_key in processed_samples
                    if exists:
                        print(f"Skipping already processed sample: {sample_prefix}")
                        continue

                    future = executor.submit(process_single_item, item, vie_scores[i%len(vie_scores)], i%len(vie_scores))
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model_name} - {group_name}"):
                    result = future.result()
                    if result:
                        group_csv_list.append(result)

            # Save group-specific CSV
            group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_gpt_score.csv")
            with megfile.smart_open(group_csv_path, 'w', newline='') as f:
                # fieldnames = ["source_image", "edited_image", "instruction", "sementics_score", "quality_score", "intersection_exist", "instruction_language"]
                fieldnames = [
                    "source_image", "edited_image", "instruction", 
                    "sementics_score", "quality_score", "intersection_exist", 
                    "instruction_language", "vae_vit_sparsity", 
                    "self_attn_sparsity", "cfg_text_similarity", "cfg_img_similarity"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in group_csv_list:
                    writer.writerow(row)
            all_csv_list.extend(group_csv_list)

            print(f"Saved group CSV for {group_name}, length: {len(group_csv_list)}")

        # After processing all groups, calculate and save combined results
        if not all_csv_list:
            print(f"Warning: No results for model {model_name}, skipping combined CSV generation")
            continue

        # Save combined CSV
        combined_csv_path = os.path.join(save_path_new, f"{model_name}_combined_gpt_score.csv")
        with megfile.smart_open(combined_csv_path, 'w', newline='') as f:
            # fieldnames = ["source_image", "edited_image", "instruction", "sementics_score", "quality_score", "intersection_exist", "instruction_language"]
            fieldnames = [
                "source_image", "edited_image", "instruction", 
                "sementics_score", "quality_score", "intersection_exist", 
                "instruction_language", "vae_vit_sparsity", 
                "self_attn_sparsity", "cfg_text_similarity", "cfg_img_similarity"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_csv_list:
                writer.writerow(row)

                
            
            
