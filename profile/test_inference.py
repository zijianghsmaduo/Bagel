import os
from copy import deepcopy
from typing import (
		Any,
		AsyncIterable,
		Callable,
		Dict,
		Generator,
		List,
		NamedTuple,
		Optional,
		Tuple,
		Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# import data.transforms
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.basic.util import MLPArgs
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file

from inferencer import InterleaveInferencer

import random
import numpy as np

from modeling.basic.attention import (
	# set_new_threshold, get_sparsity, set_dump_dir, set_save, set_truncate,
	TrickAttention
)
import argparse

def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
				torch.cuda.manual_seed(seed)
				torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Image editing with text instructions")
	parser.add_argument("--threshold", type=float, help="New attention probability threshold to set for sparsity in attention mechanism.")
	parser.add_argument("--is_save", action='store_true', help="Whether to save the attention probabilities for analysis.")
	parser.add_argument("--save_dir", type=str, default="qkv_attn_probs_dump", help="Directory to save the attention probabilities.")
	parser.add_argument("--is_truncate", action='store_true', help="Whether to truncate the attention probabilities to test the robustness.")
	parser.add_argument("--attn_backend", type=str, default="naive_sparse_quant", help="Attention backend to use.")
	parser.add_argument("--sparse_gsize", type=int, default=1, help="Group size for sparse attention.")
	parser.add_argument("--vae_vit_sparse", action='store_true', help="Whether to apply sparsity to VAE and ViT attention.")
	parser.add_argument("--self_attn_sparse", action='store_true', help="Whether to apply sparsity to self-attention.")
	parser.add_argument("--mlp_save", type=str, default=None, help="Whether to save MLP activations.")
	parser.add_argument("--mlp_save_dir", type=str, default="maps/mlp_octupusy_flash", help="Directory to save MLP activations.")
	parser.add_argument("--reorder_method", type=str, default="None", choices=["None", "front", "excavate"], help="Method to reorder gen image context.")
	parser.add_argument("--use_custom_mlp", action='store_true', help="Whether to use custom MLP with CFG sparsity tracking.")
	args = parser.parse_args()

	attention_backend = args.attn_backend
	base_attention = None
	if attention_backend != "flash":
		base_attention = TrickAttention(
			attention_backend=attention_backend,
			sparse_gsize=args.sparse_gsize, sparse_topk=0.2, sparse_threshold=args.threshold if args.threshold is not None else 4e-5,
			quant_gsize=32,
			posterior_truncate_threshold=args.threshold if args.threshold is not None else 4e-5,
			save_dir=args.save_dir if args.save_dir else "attn_probs_qkv_dump_tmp",
			is_save=args.is_save, is_plot=False, is_truncate=args.is_truncate,
			plot_dir="plot/sparse_attention_scores", heads_to_plot=[0, 1, 2],
			vae_vit=args.vae_vit_sparse, self_attn=args.self_attn_sparse
		)

	mlp_args = MLPArgs(
		use_custom_mlp=args.use_custom_mlp,
		save_mlp=args.mlp_save,
		save_dir=args.mlp_save_dir
	)

	model_path = "./models/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

	# LLM config preparing
	llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
	llm_config.qk_norm = True
	llm_config.tie_word_embeddings = False
	llm_config.layer_module = "Qwen2MoTDecoderLayer"

	# ViT config preparing
	vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
	vit_config.rope = False
	vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

	# VAE loading
	vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

	# Bagel config preparing
	config = BagelConfig(
			visual_gen=True,
			visual_und=True,
			llm_config=llm_config, 
			vit_config=vit_config,
			vae_config=vae_config,
			vit_max_num_patch_per_side=70,
			connector_act='gelu_pytorch_tanh',
			latent_patch_size=2,
			max_latent_size=64,
	)

	with init_empty_weights():
			language_model = Qwen2ForCausalLM(llm_config, mlp_args=mlp_args, trick_attn=base_attention)
			vit_model      = SiglipVisionModel(vit_config)
			model          = Bagel(language_model, vit_model, config)
			model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

	# Tokenizer Preparing
	tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
	tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

	# Image Transform Preparing
	vae_transform = ImageTransform(1024, 512, 16)
	vit_transform = ImageTransform(980, 224, 14)

	max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

	device_map = infer_auto_device_map(
			model,
			max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
			no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
	)
	print(device_map)

	same_device_modules = [
			'language_model.model.embed_tokens',
			'time_embedder',
			'latent_pos_embed',
			'vae2llm',
			'llm2vae',
			'connector',
			'vit_pos_embed'
	]

	if torch.cuda.device_count() == 1:
			first_device = device_map.get(same_device_modules[0], "cuda:0")
			for k in same_device_modules:
					if k in device_map:
							device_map[k] = first_device
					else:
							device_map[k] = "cuda:0"
	else:
			first_device = device_map.get(same_device_modules[0])
			for k in same_device_modules:
					if k in device_map:
							device_map[k] = first_device

	# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
	model = load_checkpoint_and_dispatch(
			model,
			checkpoint=os.path.join(model_path, "ema.safetensors"),
			device_map=device_map,
			offload_buffers=True,
			dtype=torch.bfloat16,
			force_hooks=True,
			offload_folder="/tmp/offload"
	)

	model = model.eval()
	print('Model loaded')

	seed = 500
	set_seed(seed)

	inferencer = InterleaveInferencer(
			model=model, 
			vae_model=vae_model, 
			tokenizer=tokenizer, 
			vae_transform=vae_transform, 
			vit_transform=vit_transform, 
			new_token_ids=new_token_ids,
			reorder_method=args.reorder_method,
	)

	def gen_inference(prompt, enable_taylorseer = False):
		inference_hyper=dict(
			cfg_text_scale=4.0,
			cfg_img_scale=1.0,
			cfg_interval=[0.4, 1.0],
			timestep_shift=3.0,
			num_timesteps=50,
			cfg_renorm_min=0.0,
			cfg_renorm_type="global",
			enable_taylorseer=enable_taylorseer,
		)
		print(prompt)
		print('-' * 10)
		output_dict = inferencer(text=prompt, **inference_hyper)

		save_path = "outputs/gen_inference_output.jpg"
		output_dict['image'].save(save_path)
		print(f"Image saved to {save_path}")

	def gen_inference_with_thinking(prompt, enable_taylorseer = False):
		inference_hyper=dict(
			max_think_token_n=1000,
			do_sample=False,
			# text_temperature=0.3,
			cfg_text_scale=4.0,
			cfg_img_scale=1.0,
			cfg_interval=[0.4, 1.0],
			timestep_shift=3.0,
			num_timesteps=50,
			cfg_renorm_min=0.0,
			cfg_renorm_type="global",
			enable_taylorseer=enable_taylorseer,
		)
		print(prompt)
		print('-' * 10)
		output_dict = inferencer(text=prompt, think=True, **inference_hyper)

		print(output_dict['text'])
		save_path = "outputs/gen_inference_with_thinking_output.jpg"
		output_dict['image'].save(save_path)
		print(f"Image saved to {save_path}")

	def editing_inference(prompt, image, enable_tayloreer = False):
		inference_hyper=dict(
				cfg_text_scale=4.0,
				cfg_img_scale=2.0,
				cfg_interval=[0.0, 1.0],
				timestep_shift=3.0,
				num_timesteps=50,
				cfg_renorm_min=0.0,
				cfg_renorm_type="text_channel",
				enable_taylorseer=enable_tayloreer,
		)
		print(prompt)
		print('-' * 10)
		output_dict = inferencer(text=prompt, image=image, **inference_hyper)
		save_path = "outputs/editing_inference_output.jpg"
		output_dict['image'].save(save_path)
		print(f"Image saved to {save_path}")

	def editing_inference_with_thinking(prompt, image, enable_taylorseer = False, save_path="outputs/editing_inference_with_thinking_output.jpg"):
		inference_hyper=dict(
				max_think_token_n=1000,
				do_sample=False,
				# text_temperature=0.3,
				cfg_text_scale=4.0,
				cfg_img_scale=2.0,
				cfg_interval=[0.0, 1.0],
				timestep_shift=3.0,
				num_timesteps=50,
				cfg_renorm_min=0.0,
				cfg_renorm_type="text_channel",
				enable_taylorseer=enable_taylorseer,
		)
		print(prompt)
		print('-' * 10)

		output_dict = inferencer(text=prompt, image=image, think=True, **inference_hyper)
		print(output_dict['text'])
		output_dict['image'].save(save_path)
		print(f"Image saved to {save_path}")

	def understanding_inference(prompt, image):
		inference_hyper=dict(
			max_think_token_n=1000,
			do_sample=False,
			# text_temperature=0.3,
		)
		print(prompt)
		print('-'*10)
		output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
		print(output_dict['text'])

	# image1 = Image.open('test_images/women.jpg')
	# editing_inference("She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.", image1)
	image2 = Image.open('test_images/octupusy.jpg')
	editing_inference_with_thinking("Could you display the sculpture that takes after this design?", image2)
	# image1 = Image.open('test_images/women.jpg')
	# editing_inference_with_thinking("She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.", image1)

	# image3 = Image.open('test_images/meme.jpg')
	# understanding_inference("Can someone explain what’s funny about this meme??", image3)

	inferencer.model.language_model.model

	if base_attention is not None:
		if args.is_truncate and args.is_save:
			sparsity = base_attention.get_sparsity()
			write_txt = f"Sparsity levels for each timestep and layer:\n{sparsity}" + "\nAverage sparsity per layer:\n" + str(np.mean(sparsity, axis=0))
			with open(os.path.join(args.save_dir, "sparsity_levels.txt"), "w") as f:
				f.write(write_txt)
			print(sparsity)
		else:
			sparsity = base_attention.get_sparsity()
			# --- 格式化输出 ---
			print("\n" + "="*60)
			print(" " * 15 + "Attention Sparsity Summary")
			print("="*60)
			
			# 表头
			print(f"{'Attention Type':<20} | {'Component':<20} | {'Average Sparsity':>15}")
			print("-"*60)

			# 循环处理每种类型
			for cfg_type in ["normal", "cfg_text", "cfg_img"]:
				if cfg_type in sparsity:
					# 计算 VAE + ViT 的稀疏度
					vae_vit_sparsity_list = np.array(sparsity[cfg_type][0])
					if vae_vit_sparsity_list.size > 0: # 确保数组不为空
						avg_vae_vit = np.mean(vae_vit_sparsity_list)
						print(f"{cfg_type.replace('_', ' ').title():<20} | {'VAE + ViT':<20} | {avg_vae_vit:>14.4%}")
					
					# 计算 Self-Attention 的稀疏度
					self_attn_sparsity_list = np.array(sparsity[cfg_type][1])
					if self_attn_sparsity_list.size > 0: # 确保数组不为空
						avg_self_attn = np.mean(self_attn_sparsity_list)
						print(f"{'':<20} | {'Self-Attention':<20} | {avg_self_attn:>14.4%}")
					
					print("-"*60)

	if args.use_custom_mlp:
		sparsity = mlp_args.mot_sparsity
		# --- 格式化输出 ---
		print("\n" + "="*60)
		print(" " * 15 + "MLP Sparsity Summary")
		print("="*60)
		
		# 表头
		print(f"{'CFG Type':<20} | {'Component':<20} | {'Average Sparsity':>15}")
		print("-"*60)

		# 循环处理每种类型
		for cfg_type in ["cfg_text", "cfg_img"]:
			if cfg_type in sparsity:
				# 计算 MLP 的稀疏度
				mlp_sparsity_list = np.array(sparsity[cfg_type])
				if mlp_sparsity_list.size > 0: # 确保数组不为空
					avg_mlp = np.mean(mlp_sparsity_list)
					print(f"{cfg_type.replace('_', ' ').title():<20} | {'MLP':<20} | {avg_mlp:>14.4%}")
				
				print("-"*60)