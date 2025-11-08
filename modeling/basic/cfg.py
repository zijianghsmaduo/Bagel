from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache

from modeling.basic import KVCacheStructure

class ReorderRopeContext:
	def __init__(
		self,
		model, 
		vae_model, 
		tokenizer, 
		vae_transform, 
		vit_transform, 
		new_token_ids,
		method: str = "front",
	):
		self.method = method
		self.model = model
		self.vae_model = vae_model
		self.tokenizer = tokenizer
		self.vae_transform = vae_transform
		self.vit_transform = vit_transform
		self.new_token_ids = new_token_ids
		self.device = "cuda"

	def init_gen_context(self): 
		gen_context = {
				'kv_lens': [0],
				'ropes': [0],
				'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
		}
		return gen_context

	def move_generation_input_to_device(self, generation_input, device):
		# Utility to move all tensors in generation_input to device
		for k, v in generation_input.items():
				if isinstance(v, torch.Tensor):
						generation_input[k] = v.to(device)
		return generation_input
	
	@torch.no_grad()
	def update_context_text(self, text, gen_context):
		# used for interleave data, currently only support 1 data inference, 

		past_key_values = gen_context['past_key_values']
		kv_lens = gen_context['kv_lens']
		ropes = gen_context['ropes']
		generation_input, kv_lens, ropes = self.model.prepare_prompts(
			curr_kvlens=kv_lens,
			curr_rope=ropes, 
			prompts=[text],
			tokenizer=self.tokenizer, 
			new_token_ids=self.new_token_ids,
		)
		with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
			generation_input = self.move_generation_input_to_device(generation_input, device=self.device)
			past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
		
		gen_context['kv_lens'] = kv_lens
		gen_context['ropes'] = ropes
		gen_context['past_key_values'] = past_key_values
		
		return gen_context

	@torch.no_grad()
	def update_context_image(self, image, gen_context, vae=True, vit=True):
		# used for interleave data, currently only support 1 data inference, 

		assert vae or vit
		past_key_values = gen_context['past_key_values']
		kv_lens = gen_context['kv_lens']
		ropes =  gen_context['ropes']

		if vae:
			## update vae
			og_len = kv_lens[0]
			generation_input, kv_lens, ropes = self.model.prepare_vae_images(
				curr_kvlens=kv_lens,
				curr_rope=ropes, 
				images=[image],
				transforms=self.vae_transform, 
				new_token_ids=self.new_token_ids,
			)
			with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
				generation_input = self.move_generation_input_to_device(generation_input, device=self.device)
				past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
		
		if vit:
			## update vit
			og_len = kv_lens[0]
			generation_input, kv_lens, ropes = self.model.prepare_vit_images(
				curr_kvlens=kv_lens,
				curr_rope=ropes, 
				images=[image],
				transforms=self.vit_transform, 
				new_token_ids=self.new_token_ids,
			)
			with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
				generation_input = self.move_generation_input_to_device(generation_input, device=self.device)
				past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

		gen_context['kv_lens'] = kv_lens
		gen_context['ropes'] = ropes
		gen_context['past_key_values'] = past_key_values
		
		return gen_context
	
	def prepare_latent(
		self,
		image_shape,
		gen_context,
		cfg_text_context,
		cfg_img_context,
	):
		# print(cfg_renorm_type)
		past_key_values = gen_context['past_key_values']
		kv_lens = gen_context['kv_lens']
		ropes = gen_context['ropes']
		print(f"ropes: {ropes}")
		# Modify
		# ropes = [0]
		generation_input = self.model.prepare_vae_latent(
				curr_kvlens=kv_lens,
				curr_rope=ropes, 
				image_sizes=[image_shape], 
				new_token_ids=self.new_token_ids,
		) 
		
		# text cfg
		cfg_text_past_key_values = cfg_text_context['past_key_values']
		kv_lens_cfg = cfg_text_context['kv_lens']
		ropes_cfg = cfg_text_context['ropes']
		print(f"cfg_text_ropes: {ropes_cfg}")
		# Modify
		# ropes_cfg = [0]
		generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
				curr_kvlens=kv_lens_cfg,
				curr_rope=ropes_cfg, 
				image_sizes=[image_shape], 
		)

		# img cfg
		cfg_img_past_key_values = cfg_img_context['past_key_values']
		kv_lens_cfg = cfg_img_context['kv_lens']
		ropes_cfg = cfg_img_context['ropes']
		print(f"cfg_img_ropes: {ropes_cfg}")
		# Modify
		# ropes_cfg = [0]
		generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
				curr_kvlens=kv_lens_cfg,
				curr_rope=ropes_cfg, 
				image_sizes=[image_shape], 
		)

		return generation_input, past_key_values, generation_input_cfg_text, cfg_text_past_key_values, generation_input_cfg_img, cfg_img_past_key_values

	def reorder_context_front(
		self,
		system_prompt: str,
		input_lists: List[Union[str, Image.Image]],
		gen_text: str,
		kv_struct: KVCacheStructure,
		think: bool = True
	):
		gen_context = self.init_gen_context()
		cfg_text_context = deepcopy(gen_context)
		cfg_img_context = deepcopy(gen_context)

		gen_context['ropes'] = [1]
		# cfg_text_context['ropes'] = [1]
		cfg_img_context['ropes'] = [1]

		image_shapes = (1024, 1024)

		with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
				if think:
						gen_context = self.update_context_text(system_prompt, gen_context)
						cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
						# cfg_text_context = self.update_context_text(system_prompt, cfg_text_context)
						# print(f"After adding system prompt, kv_lens: {gen_context['kv_lens']}")

				for input_term in input_lists:
						if isinstance(input_term, str):
								cfg_text_context = deepcopy(gen_context)
								gen_context = self.update_context_text(input_term, gen_context)
								print(f"After adding input text, kv_lens: {gen_context['kv_lens']}")
								cfg_img_context = self.update_context_text(input_term, cfg_img_context)

						elif isinstance(input_term, Image.Image):
								input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
								print("VAE input image size:", input_term.size)
								gen_context = self.update_context_image(input_term, gen_context, vae=True)
								print(f"After adding input image, kv_lens: {gen_context['kv_lens']}")
								image_shapes = input_term.size[::-1]
								# cfg_text_context = self.update_context_image(input_term, cfg_text_context, vae=True)
								cfg_text_context = deepcopy(gen_context)

						else:
								raise ValueError(f"Unsupported input type: {type(input_term)}")

				if think:
						gen_context = self.update_context_text(gen_text, gen_context)
						cfg_img_context = self.update_context_text(gen_text, cfg_img_context)

		gen_context["ropes"] = [0]
		cfg_text_context["ropes"] = [0]
		cfg_img_context["ropes"] = [0]

		return self.prepare_latent(
			image_shape=image_shapes,
			gen_context=gen_context,
			cfg_text_context=cfg_text_context,
			cfg_img_context=cfg_img_context,
		)

	
	def reorder_context_encavate(
		self,
		system_prompt: str,
		input_lists: List[Union[str, Image.Image]],
		gen_text: str,
		kv_struct: KVCacheStructure,
		think: bool = True
	):
		gen_context = self.init_gen_context()
		cfg_text_context = deepcopy(gen_context)
		cfg_img_context = deepcopy(gen_context)

		image_shapes = (1024, 1024)

		with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
				if think:
						gen_context = self.update_context_text(system_prompt, gen_context)
						cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
						# cfg_text_context = self.update_context_text(system_prompt, cfg_text_context)
						# print(f"After adding system prompt, kv_lens: {gen_context['kv_lens']}")

				for input_term in input_lists:
						if isinstance(input_term, str):
								cfg_text_context = deepcopy(gen_context)
								gen_context = self.update_context_text(input_term, gen_context)
								print(f"After adding input text, kv_lens: {gen_context['kv_lens']}")
								cfg_img_context["ropes"] = deepcopy(gen_context["ropes"]) ##
								cfg_img_context = self.update_context_text(input_term, cfg_img_context)

						elif isinstance(input_term, Image.Image):
								input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
								print("VAE input image size:", input_term.size)
								gen_context = self.update_context_image(input_term, gen_context, vae=True)
								print(f"After adding input image, kv_lens: {gen_context['kv_lens']}")
								image_shapes = input_term.size[::-1]
								# cfg_text_context = self.update_context_image(input_term, cfg_text_context, vae=True)
								cfg_text_context = deepcopy(gen_context)

						else:
								raise ValueError(f"Unsupported input type: {type(input_term)}")

				if think:
						gen_context = self.update_context_text(gen_text, gen_context)
						cfg_img_context = self.update_context_text(gen_text, cfg_img_context)

		cfg_text_context["ropes"] = deepcopy(gen_context["ropes"]) ##

		return self.prepare_latent(
			image_shape=image_shapes,
			gen_context=gen_context,
			cfg_text_context=cfg_text_context,
			cfg_img_context=cfg_img_context,
		)

	
	def forward_reorder(self, **kwargs):
		if self.method == "front":
			return self.reorder_context_front(**kwargs)
		elif self.method == "encavate":
			return self.reorder_context_encavate(**kwargs)
		else:
			raise ValueError(f"Unsupported reorder method: {self.method}")