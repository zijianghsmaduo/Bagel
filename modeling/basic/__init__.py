from typing import Optional, Tuple


OctopusKVCache = {
	"len": 4869,
	"system_prompt": (0, 47),
	"vae": (48, 1425),
	"vit": (1426, 3240),
	"input_prompt": (3241, 3253),
	"gen_text": (3254, 3490),
	"gen_image": (3491, 4868)
}

WomanKVCache = {
	"len": 10390,
	"system_prompt": (0, 47),
	"vae": (48, 3249),
	"vit": (3250, 7101),
	"input_prompt": (7102, 7120),
	"gen_text": (7121, 7188),
	"gen_image": (7189, 10390),
}


class KVCacheStructure:
	def __init__(self):
		self.len: Optional[int] = None
		self.system_prompt: Optional[Tuple[int, int]] = None
		self.vae: Optional[Tuple[int, int]] = None
		self.vit: Optional[Tuple[int, int]] = None
		self.input_prompt: Optional[Tuple[int, int]] = None
		self.gen_text: Optional[Tuple[int, int]] = None
		self.gen_image: Optional[Tuple[int, int]] = None
		self.cfg_text_gen_image: Optional[Tuple[int, int]] = None
		self.cfg_img_gen_image: Optional[Tuple[int, int]] = None


	def read_from_dict(self, kv_cache_dict: dict):
		self.len = kv_cache_dict.get("len", None)
		self.system_prompt = kv_cache_dict.get("system_prompt", None)
		self.vae = kv_cache_dict.get("vae", None)
		self.vit = kv_cache_dict.get("vit", None)
		self.input_prompt = kv_cache_dict.get("input_prompt", None)
		self.gen_text = kv_cache_dict.get("gen_text", None)
		self.gen_image = kv_cache_dict.get("gen_image", None)

	def print_structure(self):
		print("KV Cache Structure:")
		print(f"  Total Length: {self.len}")
		print(f"  System Prompt: {self.system_prompt}")
		print(f"  VAE: {self.vae}")
		print(f"  ViT: {self.vit}")
		print(f"  Input Prompt: {self.input_prompt}")
		print(f"  Generated Text: {self.gen_text}")
		print(f"  Generated Image: {self.gen_image}")
		print(f"  CFG Text + Gen Image: {self.cfg_text_gen_image}")
		print(f"  CFG Img + Gen Image: {self.cfg_img_gen_image}")

	def calculate_gen_image(self):
		if self.system_prompt and self.vae and self.vit and self.input_prompt and self.gen_text:
			start = None
			end = None
			if self.gen_text:
				start = self.gen_text[1] + 1
				end = start + self.vae[1] - self.vae[0]
				self.gen_image = (start, end)
			else:
				start = self.input_prompt[1] + 1
				end = start + self.vae[1] - self.vae[0]
				self.gen_image = (start, end)
			self.cfg_text_gen_image = (self.vit[1]+1, self.vit[1]+1+(end-start))
			input_prompt_len = self.input_prompt[1] - self.input_prompt[0] + 1
			system_prompt_len = self.system_prompt[1] - self.system_prompt[0] + 1
			pre_len = system_prompt_len + input_prompt_len ## may have gen_text in between
			self.cfg_img_gen_image = (pre_len, pre_len + (end - start))
			self.len = end + 1