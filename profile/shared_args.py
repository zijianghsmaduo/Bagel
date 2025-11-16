import argparse

def add_argus_arguments(parser: argparse.ArgumentParser):
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
	parser.add_argument("--quantized_mlp_und_w", action='store_true', help="Whether to use quantized weights for text generation in custom MLP.")
	parser.add_argument("--quantized_mlp_gen_w", action='store_true', help="Whether to use quantized weights for generation in custom MLP.")
	parser.add_argument("--mlp_use_similarity", action='store_true', help="Whether to use similarity in custom MLP.")
	parser.add_argument("--use_full_head_similarity", action='store_true', help="Whether to use full head similarity in custom MLP.")
	return parser