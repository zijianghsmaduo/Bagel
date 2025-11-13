# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache

from modeling.basic import KVCacheStructure



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, reorder_method="None"):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.reorder_method = reorder_method
        print(f"InterleaveInferencer initialized with reorder_method: {self.reorder_method}")

    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

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

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, kv_struct: KVCacheStructure, vae=True, vit=True):
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
            kv_struct.vae = (og_len, kv_lens[0]-1)
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
            kv_struct.vit = (og_len, kv_lens[0]-1)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context,
        kv_struct: KVCacheStructure,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
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
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        print(f"cfg_text_ropes: {ropes_cfg}")
        # Modify
        # ropes_cfg = [0]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        print(f"cfg_img_ropes: {ropes_cfg}")
        # Modify
        # ropes_cfg = [0]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        
        # if kv_struct.gen_image is None:
            

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
            kv_cache_struct=kv_struct,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
    
    def gen_image_context(
        self,
        input_lists: List[Union[str, Image.Image]],
        gen_text,
        normal_start,
        cfg_text_start,
        cfg_img_start,
        think
    ):
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        gen_context['ropes'] = [normal_start]
        cfg_text_context['ropes'] = [cfg_text_start]
        cfg_img_context['ropes'] = [cfg_img_start]

        kv_struct = KVCacheStructure()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                # if understanding_output:
                #     system_prompt = VLM_THINK_SYSTEM_PROMPT 
                # else:
                system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
                cfg_text_context = self.update_context_text(system_prompt, cfg_text_context)
                print(f"After adding system prompt, kv_lens: {gen_context['kv_lens']}")

            for input_term in input_lists:
                if isinstance(input_term, str):
                    # cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    print(f"After adding input text, kv_lens: {gen_context['kv_lens']}")
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    print("VAE input image size:", input_term.size)
                    gen_context = self.update_context_image(input_term, gen_context, kv_struct=kv_struct, vae=True)
                    print(f"After adding input image, kv_lens: {gen_context['kv_lens']}")
                    cfg_text_context = self.update_context_image(input_term, cfg_text_context, kv_struct=kv_struct, vae=True)
                    # cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if think:
                gen_context = self.update_context_text(gen_text, gen_context)

        return gen_context, cfg_text_context, cfg_img_context

    def gen_image_context_excavate(
        self,
        input_lists: List[Union[str, Image.Image]],
        gen_text,
        think
    ):
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        gen_context['ropes'] = [0]
        cfg_text_context['ropes'] = [0]
        cfg_img_context['ropes'] = [0]
        kv_struct = KVCacheStructure()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                # if understanding_output:
                #     system_prompt = VLM_THINK_SYSTEM_PROMPT 
                # else:
                system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
                cfg_text_context = self.update_context_text(system_prompt, cfg_text_context)
                print(f"After adding system prompt, kv_lens: {gen_context['kv_lens']}")

            for input_term in input_lists:
                if isinstance(input_term, str):
                    # cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    print(f"After adding input text, kv_lens: {gen_context['kv_lens']}")
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    print("VAE input image size:", input_term.size)
                    gen_context = self.update_context_image(input_term, gen_context, kv_struct=kv_struct, vae=True)
                    print(f"After adding input image, kv_lens: {gen_context['kv_lens']}")
                    cfg_text_context = self.update_context_image(input_term, cfg_text_context, kv_struct=kv_struct, vae=True)
                    # cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if think:
                gen_context = self.update_context_text(gen_text, gen_context)

        cfg_text_context["ropes"] = deepcopy(gen_context["ropes"])
        cfg_img_context["ropes"] = deepcopy(gen_context["ropes"])

        return gen_context, cfg_text_context, cfg_img_context

    def gen_image_context_fronting(
        self,
        input_lists: List[Union[str, Image.Image]],
        gen_text,
        think
    ):
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        gen_context['ropes'] = [1]
        cfg_text_context['ropes'] = [1]
        cfg_img_context['ropes'] = [1]

        kv_struct = KVCacheStructure()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                # if understanding_output:
                #     system_prompt = VLM_THINK_SYSTEM_PROMPT 
                # else:
                system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
                cfg_text_context = self.update_context_text(system_prompt, cfg_text_context)
                print(f"After adding system prompt, kv_lens: {gen_context['kv_lens']}")

            for input_term in input_lists:
                if isinstance(input_term, str):
                    # cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    print(f"After adding input text, kv_lens: {gen_context['kv_lens']}")
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    print("VAE input image size:", input_term.size)
                    gen_context = self.update_context_image(input_term, gen_context, kv_struct=kv_struct, vae=True)
                    print(f"After adding input image, kv_lens: {gen_context['kv_lens']}")
                    cfg_text_context = self.update_context_image(input_term, cfg_text_context, kv_struct=kv_struct, vae=True)
                    # cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if think:
                gen_context = self.update_context_text(gen_text, gen_context)

        gen_context["ropes"] = [0]
        cfg_text_context["ropes"] = [0]
        cfg_img_context["ropes"] = [0]

        return gen_context, cfg_text_context, cfg_img_context

            
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        kv_struct = KVCacheStructure()

        is_editing = False

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                og_len = gen_context['kv_lens'][0]
                gen_context = self.update_context_text(system_prompt, gen_context)
                new_len = gen_context['kv_lens'][0]
                kv_struct.system_prompt = (og_len, new_len-1)

                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
                print(f"After adding system prompt, kv_lens: {gen_context['kv_lens']}")

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)

                    og_len = gen_context['kv_lens'][0]
                    gen_context = self.update_context_text(input_term, gen_context)
                    new_len = gen_context['kv_lens'][0]
                    kv_struct.input_prompt = (og_len, new_len-1)

                    print(f"After adding input text, kv_lens: {gen_context['kv_lens']}")
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    is_editing = not understanding_output
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    print("VAE input image size:", input_term.size)
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output, kv_struct=kv_struct)
                    print(f"After adding input image, kv_lens: {gen_context['kv_lens']}")
                    image_shapes = input_term.size[::-1]
                    # cfg_text_context = self.update_context_image(input_term, cfg_text_context, vae=not understanding_output)
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                gen_text = ""
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    # ?? 为什么不在 gen_text 执行之后直接返回新的 context = { kvcache, RoPE, kv_lens } 或者在 gen_text 里直接更新 gen_context
                    # ?? 而是生成完 text 之后再 update_context_text
                    og_len = gen_context['kv_lens'][0]
                    gen_context = self.update_context_text(gen_text, gen_context)
                    new_len = gen_context['kv_lens'][0]
                    kv_struct.gen_text = (og_len, new_len-1)

                    print(f"After adding generated text, kv_lens: {gen_context['kv_lens']}")
                    output_list.append(gen_text)

                if is_editing:
                  kv_struct.calculate_gen_image()
                elif not understanding_output: # generation
                  nr_image_tokens = (image_shapes[0] // self.model.latent_downsample) * (image_shapes[1] // self.model.latent_downsample) + 2
                  if think:
                    assert kv_struct.gen_text is not None
                    assert kv_struct.system_prompt is not None
                    kv_struct.gen_image = (kv_struct.gen_text[1]+1, kv_struct.gen_text[1]+1+nr_image_tokens-1)
                    kv_struct.cfg_text_gen_image = (kv_struct.system_prompt[1]+1, kv_struct.system_prompt[1]+1+nr_image_tokens-1)
                    kv_struct.len = kv_struct.gen_image[1]+1
                  else:
                    kv_struct.calculate_gen_image(
                        task_mode="generation", 
                        image_token_len=nr_image_tokens
                    )
                kv_struct.print_structure()

                if self.reorder_method != "None":
                    if self.reorder_method == "front":
                        gen_context, cfg_text_context, cfg_img_context = self.gen_image_context_fronting(
                            input_lists=input_lists,
                            gen_text=gen_text,
                            think=think,
                        )
                    elif self.reorder_method == "excavate":
                        gen_context, cfg_text_context, cfg_img_context = self.gen_image_context_excavate(
                            input_lists=input_lists,
                            gen_text=gen_text,
                            think=think,
                        )
                    else:
                        raise ValueError(f"Unsupported reorder method: {self.reorder_method}")

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    kv_struct=kv_struct,
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
