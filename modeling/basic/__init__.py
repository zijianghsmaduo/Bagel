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


	def read_from_dict(self, kv_cache_dict: dict):
		self.len = kv_cache_dict.get("len", None)
		self.system_prompt = kv_cache_dict.get("system_prompt", None)
		self.vae = kv_cache_dict.get("vae", None)
		self.vit = kv_cache_dict.get("vit", None)
		self.input_prompt = kv_cache_dict.get("input_prompt", None)
		self.gen_text = kv_cache_dict.get("gen_text", None)
		self.gen_image = kv_cache_dict.get("gen_image", None)