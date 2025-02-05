import coremltools as ct
import pickle
import torch
from opensora.registry import MODELS, build_module
import warnings
import math

warnings.filterwarnings('ignore')
model_config = dict(
    type="STDiT3-XL-Custom-1/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)
latent_size =[20, 20, 27]
vae_output_chnnel = 4
text_encoder_out_dim = 4096
text_encoder_model_max_length = 300
enable_sequence_parallelism = False
dtype = torch.float32

stdit3 = build_module(
        model_config,
        MODELS,
        input_size=latent_size,
        in_channels=vae_output_chnnel,
        caption_channels=text_encoder_out_dim,
        model_max_length=text_encoder_model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
).eval()

z_in = torch.randn([2, 4, 20, 20, 27])
t = torch.tensor([1000.0, 1000.0], dtype=torch.float32)

with open('y.pkl', 'rb') as f:
    y = pickle.load(f)
with open('mask.pkl', 'rb') as f:
    mask = pickle.load(f)
height = torch.tensor([166.], dtype=torch.float32)     
width = torch.tensor([221.], dtype=torch.float32)
with open('fps.pkl', 'rb') as f:
    fps = pickle.load(f)
x_mask = torch.tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True]])

# Tracing the model
input_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=0, upper_bound=300, default=70), 1152))
z_in_shape = ct.Shape(shape=(2, 4, ct.RangeDim(lower_bound=10, upper_bound= 50, default=20), ct.RangeDim(lower_bound=10, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))

def get_dynamic_size(x):
    _, _, T, H, W = x.size()
    patch_size = [1,2,2]
    if T % patch_size[0] != 0:
        T = patch_size[0] - T % patch_size[0]
    else:
        T = 0
    if H % patch_size[1] != 0:
        H = patch_size[1] - H % patch_size[1]
    else:
        H = 0
    if W % patch_size[2] != 0:
        W = patch_size[2] - W % patch_size[2]
    else:
        W = 0
    return (T, H, W)

(T, H, W) = get_dynamic_size(z_in)
T = torch.tensor([T], dtype=torch.int)
H = torch.tensor([H], dtype=torch.int)
W = torch.tensor([W], dtype=torch.int)

traced_model = torch.jit.trace(stdit3, (z_in, t, y, mask, fps, height, width, H, W))
print("Model traced")
# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    convert_to='mlprogram',
    inputs=[
        ct.TensorType(name="z_in", shape=z_in_shape),
        ct.TensorType(name="t", shape=t.shape),
        ct.TensorType(name="y", shape=y.shape),
        ct.TensorType(name="mask", shape=mask.shape),
        ct.TensorType(name="fps", shape=fps.shape),
        ct.TensorType(name="height", shape=height.shape),
        ct.TensorType(name="width", shape=width.shape),
        ct.TensorType(name="padH", shape= H.shape),
        ct.TensorType(name="padW", shape= W.shape),

    ],
    outputs=[
        ct.TensorType(name="x"),
        ct.TensorType(name="outY"),
        ct.TensorType(name="t_mlp"),
        ct.TensorType(name="T"),
        ct.TensorType(name="outT"),
    ],
    # compute_units= ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT32,
)

# Save the model
mlmodel.save("stdit3/stdit3_part1.mlpackage")
print("Model converted and saved as stdit3.mlpackage")

# Test the model between the original and the converted model
converted = mlmodel.predict({"z_in": z_in.float(), "t": t.float(), "y": y.float(), "mask": mask.float(), "fps": fps.float(), "height": height.float(), "width": width.float(), "padH": H, "padW": W})
print("Converted output: ", converted["x"].shape)

(x, y, t_mlp, T, t) = stdit3(z_in, t, y, mask, fps, height, width, H, W)
print("Original output: ", x.shape)


# ============================ Part 2 ============================
model_config_ST = dict(
    type="STDiT3-XL-Custom-ST/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)

stdit3_ST = build_module(
        model_config_ST,
        MODELS,
        input_size=latent_size,
        in_channels=vae_output_chnnel,
        caption_channels=text_encoder_out_dim,
        model_max_length=text_encoder_model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
).eval()

model_config_TDTM = dict(
    type="STDiT3-XL-Custom-TDTM/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)

stdit3_TDTM = build_module(
        model_config_TDTM,
        MODELS,
        input_size=latent_size,
        in_channels=vae_output_chnnel,
        caption_channels=text_encoder_out_dim,
        model_max_length=text_encoder_model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
).eval()

spatial_blocks = stdit3_ST.get_submodule("spatial_blocks")
temporal_blocks = stdit3_ST.get_submodule("temporal_blocks")


T = torch.tensor([T], dtype=torch.float32)
y = torch.tensor(y)
t_mlp = torch.tensor(t_mlp)
x_mask = torch.tensor(x_mask, dtype=torch.float32)

# Dynamic shapes for the inputs of the STDiT model
x_shape = ct.Shape(shape=(2, ct.RangeDim(lower_bound=2000, upper_bound= 7820, default=2800), 1152))
attn_shape = ct.Shape(shape=(1, 16, ct.RangeDim(lower_bound=2000, upper_bound= 15640, default= 5600), ct.RangeDim(lower_bound=0, upper_bound=300, default=70)))    
attn = torch.randn([1, 16, 5600, 70])


# for i, (spatial_block, temporal_block) in enumerate(zip(spatial_blocks, temporal_blocks)):
#     # Tracing the model
#     print(f"Tracing block {i}")
#     # Tracing spatial
#     # Convert non-Tensor inputs to Tensors
#     stdit3_ST.spatial_block = spatial_block
#     stdit3_ST.temporal_block = temporal_block
#     # Convert y_lens to a tensor
#     # Update the trace call
#     trace_ST = torch.jit.trace(stdit3_ST, (x, y, t_mlp, attn, T))

#     mlmodel_ST = ct.convert(trace_ST, 
#                                 convert_to= 'mlprogram', 
#                                 inputs=[ct.TensorType(name="x", shape=x_shape), 
#                                         ct.TensorType(name="y", shape=input_shape),
#                                         ct.TensorType(name="t_mlp", shape=t_mlp.shape),                                        
#                                         ct.TensorType(name="attn", shape=attn_shape),                                        
#                                         ct.TensorType(name="T", shape=T.shape), 
#                                         ], 
#                                 outputs=[ct.TensorType(name="output")], 
#                                 minimum_deployment_target= ct.target.iOS18,
#                                 compute_precision=ct.precision.FLOAT32)
    
#     mlmodel_ST.save(f"stdit3/stdit3_ST_{i}.mlpackage")
#     print(f"Model converted and saved as stdit3_ST_{i}.mlpackage")
    
#     origin_x = stdit3_ST(x, y, t_mlp,attn, T)
#     print(origin_x)
#     print("==================")

#     converted_x = mlmodel_ST.predict({"x": x, "y": y, "t_mlp": t_mlp, "attn": attn,"T": T})["output"]
#     x = torch.tensor(converted_x)
#     print(converted_x)

# For TDTM
attn = torch.randn([1, 16, 2800, 70])
for i, (spatial_block, temporal_block) in enumerate(zip(spatial_blocks, temporal_blocks)):
    # Tracing the model
    print(f"Tracing block {i}")
    # Tracing spatial
    # Convert non-Tensor inputs to Tensors
    stdit3_TDTM.spatial_block = spatial_block
    stdit3_TDTM.temporal_block = temporal_block
    # Convert y_lens to a tensor
    # Update the trace call
    trace_TDTM = torch.jit.trace(stdit3_TDTM, (x, y, t_mlp, attn, T))

    mlmodel_TDTM = ct.convert(trace_TDTM, 
                                convert_to= 'mlprogram', 
                                inputs=[ct.TensorType(name="x", shape=x_shape), 
                                        ct.TensorType(name="y", shape=input_shape),
                                        ct.TensorType(name="t_mlp", shape=t_mlp.shape),                                        
                                        ct.TensorType(name="attn", shape=attn_shape),                                        
                                        ct.TensorType(name="T", shape=T.shape), 
                                        ], 
                                outputs=[ct.TensorType(name="output")], 
                                minimum_deployment_target= ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32)
    
    mlmodel_TDTM.save(f"stdit3/stdit3_TDTM_{i}.mlpackage")
    print(f"Model converted and saved as stdit3_TDTM_{i}.mlpackage")
    
    origin_x = stdit3_TDTM(x, y, t_mlp,attn, T)
    print(origin_x)
    print("==================")

    converted_x = mlmodel_TDTM.predict({"x": x, "y": y, "t_mlp": t_mlp, "attn": attn,"T": T})["output"]
    x = torch.tensor(converted_x)
    print(converted_x)

# ============================ Part 3 ============================
# === Load custom the model ===
custom_model_config = dict(
    type="STDiT3-XL-Custom-2/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)

stdit3_custom = build_module(
        custom_model_config,
        MODELS,
        input_size=latent_size,
        in_channels=vae_output_chnnel,
        caption_channels=text_encoder_out_dim,
        model_max_length=text_encoder_model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
).eval()

z_in = torch.randn([2, 4, 20, 20, 27])
_, pad_H, pad_W = get_dynamic_size(z_in)
pad_H = torch.tensor([pad_H], dtype=torch.int)
pad_W = torch.tensor([pad_W], dtype=torch.int)

traced_custom_model = torch.jit.trace(stdit3_custom, (z_in, x, t, pad_H, pad_W))
print("Model traced")

# Convert to CoreML
# Dynamic shapes for the inputs of the part 2 of the STDiT model
x_shape = ct.Shape(shape=(2, ct.RangeDim(lower_bound=2000, upper_bound= 7820, default=2800), 1152))

mlmodel = ct.convert(
    traced_custom_model,
    convert_to='mlprogram',
    inputs=[
        ct.TensorType(name="z_in", shape=z_in_shape),        
        ct.TensorType(name="x", shape=x_shape),
        ct.TensorType(name="t", shape=t.shape),
        ct.TensorType(name="padH", shape=pad_H.shape),
        ct.TensorType(name="padW", shape=pad_W.shape),

    ],
    outputs= [ct.TensorType(name="output")],
    minimum_deployment_target=ct.target.macOS15,
    compute_precision=ct.precision.FLOAT32,
)

mlmodel.save("stdit3/stdit3_part2.mlpackage")
print("Model converted and saved as stdit3_part2.mlpackage")

conveted = mlmodel.predict({"z_in": z_in, "x": x, "t": t, "padH": pad_H, "padW": pad_W})["output"]
print("Converted output: ", conveted.shape)