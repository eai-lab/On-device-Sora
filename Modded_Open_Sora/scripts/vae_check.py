import os
import time
from pprint import pformat
import pickle

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

device = "cuda"
cfg = parse_configs(training=False)
cfg_dtype = cfg.get("dtype", "fp32")
assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

""" measure peak memory usage: t5 """
# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()

# text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
# prompts = cfg.get("prompt", None)
# model_args = text_encoder.encode(prompts)

# print(f"현재 메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# print(f"피크 메모리 사용량: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
# torch.cuda.reset_peak_memory_stats()
    
# # 현재 메모리 사용량: 19.06 GB
# # 피크 메모리 사용량: 19.17 GB    

    
""" measure peak memory usage: vae """ 
torch.set_grad_enabled(False)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
breakpoint()
vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
latent = torch.randn(1, 4, 15, 32, 32, device=device)
vae_output = vae.decode(latent.to(dtype), num_frames=get_num_frames(cfg.num_frames))
print(vae_output.shape)
print(f"현재 메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"피크 메모리 사용량: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()

# 현재 메모리 사용량: 44.43 GB
# 피크 메모리 사용량: 44.49 GB


# """ measure peack memory usage: stdit """
# vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
# text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
# enable_sequence_parallelism = False

# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()

# image_size = cfg.get("image_size", None)
# num_frames = get_num_frames(cfg.num_frames)
# input_size = (num_frames, *image_size)
# stdit = (
#     build_module(
#         cfg.model,
#         MODELS,
#         input_size=vae.get_latent_size(input_size),
#         in_channels=vae.out_channels,
#         caption_channels=text_encoder.output_dim,
#         model_max_length=text_encoder.model_max_length,
#         enable_sequence_parallelism=enable_sequence_parallelism,
#     )
#     .to(device, dtype)
#     .eval()
# )
# del vae, text_encoder

# z_in = torch.randint(0, 1, (2, 4, 15, 32, 32), device=device, dtype=torch.float32)
# t = torch.randn(2, device=device, dtype=torch.float32)
# model_args = dict({
#     'y': torch.randn(2, 1, 300, 4096, device=device, dtype=torch.float32),
#     'mask': torch.randint(0, 1, (1, 300), device=device, dtype=torch.int64),
#     'height': torch.randn(1, device=device, dtype=dtype),
#     'width': torch.randn(1, device=device, dtype=dtype),
#     'num_frames': torch.randn(1, device=device, dtype=dtype),
#     'ar': torch.randn(1, device=device, dtype=dtype),
#     'fps': torch.randn(1, device=device, dtype=dtype),
#     'x_mask': torch.randint(0, 1, (2, 15), device=device, dtype=torch.bool)
# # })
# save_path = '/workspace/on-device-diffusion/opensora/scripts/stdit-input'
# z_in = pickle.load(open(os.path.join(save_path, 'z_in.pkl'), 'rb'))
# t = pickle.load(open(os.path.join(save_path, 't.pkl'), 'rb'))
# model_args = pickle.load(open(os.path.join(save_path, 'model_args.pkl'), 'rb'))

# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()

# x = stdit(z_in, t, **model_args)

# print(f"현재 메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# print(f"피크 메모리 사용량: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
# torch.cuda.reset_peak_memory_stats()
# \
