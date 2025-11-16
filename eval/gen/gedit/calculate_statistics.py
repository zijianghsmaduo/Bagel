# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import megfile
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math
import json

GROUPS = [
    "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
]

model_name = 'bagel'

def analyze_scores(save_path_dir, evaluate_group, language):
    results = defaultdict(dict)
    save_path_new = save_path_dir
    model_total_score = defaultdict(dict)

    group_dict_sub = {}
    group_scores_semantics = defaultdict(lambda: defaultdict(list))
    group_scores_quality = defaultdict(lambda: defaultdict(list))
    group_scores_overall = defaultdict(lambda: defaultdict(list))

    group_scores_semantics_intersection = defaultdict(lambda: defaultdict(list))
    group_scores_quality_intersection = defaultdict(lambda: defaultdict(list))
    group_scores_overall_intersection = defaultdict(lambda: defaultdict(list))

    group_scores_vae_vit_sparsity = defaultdict(lambda: defaultdict(list))
    group_scores_self_attn_sparsity = defaultdict(lambda: defaultdict(list))
    group_scores_cfg_text_similarity = defaultdict(lambda: defaultdict(list))
    group_scores_cfg_img_similarity = defaultdict(lambda: defaultdict(list))
    group_scores_avg_self_attn = defaultdict(lambda: defaultdict(list))

    length_total = 0
    save_path_dir_raw = save_path_dir
    
    for group_name in GROUPS:

        csv_path = os.path.join(save_path_new, f"{evaluate_group[0]}_{group_name}_gpt_score.csv")
        csv_file = megfile.smart_open(csv_path)
        df = pd.read_csv(csv_file)
        
        filtered_semantics_scores = []
        filtered_quality_scores = []
        filtered_overall_scores = []
        filtered_semantics_scores_intersection = []
        filtered_quality_scores_intersection = []
        filtered_overall_scores_intersection = []
        filtered_vae_vit_sparsity = []
        filtered_self_attn_sparsity = []
        filtered_cfg_text_similarity = []
        filtered_cfg_img_similarity = []
        filtered_avg_self_attn_score = []
        
        for _, row in df.iterrows():
            source_image = row['source_image']
            edited_image = row['edited_image']
            instruction = row['instruction']
            semantics_score = row['sementics_score']
            quality_score = row['quality_score']
            intersection_exist = row['intersection_exist']
            instruction_language = row['instruction_language']

            try:
                vae_vit_sparsity = row['vae_vit_sparsity']
                self_attn_sparsity = row['self_attn_sparsity']
                cfg_text_similarity = row['cfg_text_similarity']
                cfg_img_similarity = row['cfg_img_similarity']
                avg_self_attn_score = row['avg_self_attn_score']
            except KeyError:
                # 如果旧的 CSV 文件中没有这些列，则跳过
                vae_vit_sparsity = np.nan
                self_attn_sparsity = np.nan
                cfg_text_similarity = np.nan
                cfg_img_similarity = np.nan
                avg_self_attn_score = np.nan

            if instruction_language == language:
                pass
            else:
                continue
            
            overall_score = math.sqrt(semantics_score * quality_score)
            
            filtered_semantics_scores.append(semantics_score)
            filtered_quality_scores.append(quality_score)
            filtered_overall_scores.append(overall_score)

            filtered_vae_vit_sparsity.append(vae_vit_sparsity)
            filtered_self_attn_sparsity.append(self_attn_sparsity)
            filtered_cfg_text_similarity.append(cfg_text_similarity)
            filtered_cfg_img_similarity.append(cfg_img_similarity)
            filtered_avg_self_attn_score.append(avg_self_attn_score)
            if intersection_exist:
                filtered_semantics_scores_intersection.append(semantics_score)
                filtered_quality_scores_intersection.append(quality_score)
                filtered_overall_scores_intersection.append(overall_score)
        
        avg_semantics_score = np.mean(filtered_semantics_scores)
        avg_quality_score = np.mean(filtered_quality_scores)
        avg_overall_score = np.mean(filtered_overall_scores)
        group_scores_semantics[evaluate_group[0]][group_name] = avg_semantics_score
        group_scores_quality[evaluate_group[0]][group_name] = avg_quality_score
        group_scores_overall[evaluate_group[0]][group_name] = avg_overall_score
        group_scores_vae_vit_sparsity[evaluate_group[0]][group_name] = np.nanmean(filtered_vae_vit_sparsity)
        group_scores_self_attn_sparsity[evaluate_group[0]][group_name] = np.nanmean(filtered_self_attn_sparsity)
        group_scores_cfg_text_similarity[evaluate_group[0]][group_name] = np.nanmean(filtered_cfg_text_similarity)
        group_scores_cfg_img_similarity[evaluate_group[0]][group_name] = np.nanmean(filtered_cfg_img_similarity)
        group_scores_avg_self_attn[evaluate_group[0]][group_name] = np.nanmean(filtered_avg_self_attn_score)

        avg_semantics_score_intersection = np.mean(filtered_semantics_scores_intersection)
        avg_quality_score_intersection = np.mean(filtered_quality_scores_intersection)
        avg_overall_score_intersection = np.mean(filtered_overall_scores_intersection)
        group_scores_semantics_intersection[evaluate_group[0]][group_name] = avg_semantics_score_intersection
        group_scores_quality_intersection[evaluate_group[0]][group_name] = avg_quality_score_intersection
        group_scores_overall_intersection[evaluate_group[0]][group_name] = avg_overall_score_intersection


    # print("\n--- Overall Model Averages ---")

    # print("\nSemantics:")
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_semantics[model_name]["avg_semantics"] = model_avg

    # print("\nSemantics Intersection:")
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics_intersection[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_semantics_intersection[model_name]["avg_semantics"] = model_avg
    
    # print("\nQuality:")
    for model_name in evaluate_group:
        model_scores = [group_scores_quality[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_quality[model_name]["avg_quality"] = model_avg

    # print("\nQuality Intersection:")
    for model_name in evaluate_group:
        model_scores = [group_scores_quality_intersection[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_quality_intersection[model_name]["avg_quality"] = model_avg

    # print("\nOverall:")
    for model_name in evaluate_group:
        model_scores = [group_scores_overall[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_overall[model_name]["avg_overall"] = model_avg

    # print("\nOverall Intersection:")
    for model_name in evaluate_group:
        model_scores = [group_scores_overall_intersection[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_overall_intersection[model_name]["avg_overall"] = model_avg
    
    for model_name in evaluate_group:
        group_scores_vae_vit_sparsity[model_name]["avg_vae_vit_sparsity"] = np.nanmean([group_scores_vae_vit_sparsity[model_name][group] for group in GROUPS])
        group_scores_self_attn_sparsity[model_name]["avg_self_attn_sparsity"] = np.nanmean([group_scores_self_attn_sparsity[model_name][group] for group in GROUPS])
        group_scores_cfg_text_similarity[model_name]["avg_cfg_text_similarity"] = np.nanmean([group_scores_cfg_text_similarity[model_name][group] for group in GROUPS])
        group_scores_cfg_img_similarity[model_name]["avg_cfg_img_similarity"] = np.nanmean([group_scores_cfg_img_similarity[model_name][group] for group in GROUPS])
        group_scores_avg_self_attn[model_name]["avg_avg_self_attn_score"] = np.nanmean([group_scores_avg_self_attn[model_name][group] for group in GROUPS])


    return (
        group_scores_semantics, group_scores_quality, group_scores_overall, 
        group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection,
        group_scores_vae_vit_sparsity, group_scores_self_attn_sparsity, 
        group_scores_cfg_text_similarity, group_scores_cfg_img_similarity,
        group_scores_avg_self_attn
    )
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["gpt4o", "qwen25vl", "gemini", "qwen25vl_api"])
    parser.add_argument("--language", type=str, default="en", choices=["en", "cn"])
    parser.add_argument("--output_json_path", type=str, default=None, help="Path to save the final average metrics as a JSON file.")
    args = parser.parse_args()
    save_path_dir = args.save_path
    evaluate_group = [model_name]
    backbone = args.backbone

    save_path_new = os.path.join(save_path_dir, backbone, "eval_results_new")

    print("\nOverall:")
   
    for model_name in evaluate_group:
        # group_scores_semantics, group_scores_quality, group_scores_overall, group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection = analyze_scores(save_path_new, [model_name], language=args.language)
        (
            group_scores_semantics, group_scores_quality, group_scores_overall, 
            group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection,
            group_scores_vae_vit_sparsity, group_scores_self_attn_sparsity, 
            group_scores_cfg_text_similarity, group_scores_cfg_img_similarity,
            group_scores_avg_self_attn
        ) = analyze_scores(save_path_new, [model_name], language=args.language)
    for group_name in GROUPS:
        print(f"{group_name}: {group_scores_semantics[model_name][group_name]:.3f}, {group_scores_quality[model_name][group_name]:.3f}, {group_scores_overall[model_name][group_name]:.3f}")

    print(f"Average: {group_scores_semantics[model_name]['avg_semantics']:.3f}\t{group_scores_quality[model_name]['avg_quality']:.3f}\t{group_scores_overall[model_name]['avg_overall']:.3f}")
    
    print("\n--- Sparsity and Similarity Averages ---")
    print(f"Average VAE+ViT Sparsity: {group_scores_vae_vit_sparsity[model_name]['avg_vae_vit_sparsity']:.3f}")
    print(f"Average Self-Attn Sparsity: {group_scores_self_attn_sparsity[model_name]['avg_self_attn_sparsity']:.3f}")
    print(f"Average CFG-Text Similarity: {group_scores_cfg_text_similarity[model_name]['avg_cfg_text_similarity']:.3f}")
    print(f"Average CFG-Img Similarity: {group_scores_cfg_img_similarity[model_name]['avg_cfg_img_similarity']:.3f}")
    print(f"Average Avg Self-Attn Score: {group_scores_avg_self_attn[model_name]['avg_avg_self_attn_score']:.3f}")
    
    if args.output_json_path:
        final_metrics = {
            "avg_semantics": group_scores_semantics[model_name]['avg_semantics'],
            "avg_quality": group_scores_quality[model_name]['avg_quality'],
            "avg_overall": group_scores_overall[model_name]['avg_overall'],
            "avg_vae_vit_sparsity": group_scores_vae_vit_sparsity[model_name]['avg_vae_vit_sparsity'],
            "avg_self_attn_sparsity": group_scores_self_attn_sparsity[model_name]['avg_self_attn_sparsity'],
            "avg_cfg_text_similarity": group_scores_cfg_text_similarity[model_name]['avg_cfg_text_similarity'],
            "avg_cfg_img_similarity": group_scores_cfg_img_similarity[model_name]['avg_cfg_img_similarity'],
            "avg_self_attn_score": group_scores_avg_self_attn[model_name]['avg_avg_self_attn_score'],
        }
        
        try:
            with open(args.output_json_path, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            print(f"\nSuccessfully wrote final metrics to {args.output_json_path}")
        except IOError as e:
            print(f"\nError writing to JSON file {args.output_json_path}: {e}")

    # print("\nIntersection:")

    # for group_name in GROUPS:
    #     print(f"{group_name}: {group_scores_semantics_intersection[model_name][group_name]:.3f}, {group_scores_quality_intersection[model_name][group_name]:.3f}, {group_scores_overall_intersection[model_name][group_name]:.3f}")

    # print(f"Average Intersection: {group_scores_semantics_intersection[model_name]['avg_semantics']:.3f}, {group_scores_quality_intersection[model_name]['avg_quality']:.3f}, {group_scores_overall_intersection[model_name]['avg_overall']:.3f}")
