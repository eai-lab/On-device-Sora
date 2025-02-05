import coremltools as ct
import pickle
import torch
from opensora.registry import MODELS, build_module
from einops import rearrange
import torch.nn.functional as F
import time

vae_config = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
    force_huggingface=True,
)
device = 'cpu'
dtype = torch.float32
vae = build_module(vae_config, MODELS).to(dtype= dtype).eval()
    
# ===================== spatial model =====================

class VAEWrapper_spatial(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.spatial_vae.decode(x / 0.18215).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

class VAEWrapper_spatial1(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.spatial_vae.decode_part1(x / 0.18215).sample
        # x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x
    
class VAEWrapper_spatial2(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part2(x).sample
        return x
    
class VAEWrapper_spatial3_1(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part3_1(x).sample
        return x
    
class VAEWrapper_spatial3_2(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part3_2(x).sample
        return x

class VAEWrapper_spatial3_3(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part3_3(x).sample
        return x

class VAEWrapper_spatial3_4_1(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part3_4_1(x).sample
        return x
class VAEWrapper_spatial3_4_2(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part3_4_2(x).sample
        return x

class VAEWrapper_spatial3_4_3(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # torch.Size([4, 512, 20, 27])
        x = self.spatial_vae.decode_part3_4_3(x).sample
        return x
    
class VAEWrapper_spatial4(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.spatial_vae = vae.spatial_vae.module
        # self.micro_batch_size = 4
    def forward(self, x):
        # part3 torch.Size([4, 128, 160, 216])
        x = self.spatial_vae.decode_part4(x).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=1)
        return x   
    
# ===================== spatial model =====================
# ===================== part1 =====================
tmp_input = torch.randn([1,4, 4, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper_spatial1(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 4, 4, ct.RangeDim(lower_bound=20, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae, 
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_AND_GPU,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )
print("Model converted")
converted_vae.save('vae_spatial_part1.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,4, 4, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2 =====================

tmp_input = torch.randn([4, 512, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper_spatial2(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(4, 512, ct.RangeDim(lower_bound=20, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_AND_GPU,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part2.mlpackage')
print("Model saved")

tmp_input = torch.randn([4,512, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3-1 =====================
print("part3-1")
tmp_input = torch.randn([4, 512, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper_spatial3_1(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(4, 512, ct.RangeDim(lower_bound=20, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_AND_GPU,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part3_1.mlpackage')
print("Model saved")

tmp_input = torch.randn([4,512, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3-2 =====================
print("part3-2")
tmp_input = torch.randn([4, 512, 40, 54])
print(tmp_input.shape)

wrapper = VAEWrapper_spatial3_2(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(4, 512, ct.RangeDim(lower_bound=40, upper_bound= 100, default=40), ct.RangeDim(lower_bound=40, upper_bound=100, default=54)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_AND_GPU,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part3_2.mlpackage')
print("Model saved")

tmp_input = torch.randn([4,512, 68,92])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3-3 =====================
print("part3-3")
tmp_input = torch.randn([4, 512, 80, 108])
print(tmp_input.shape)

wrapper = VAEWrapper_spatial3_3(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(4, 512, ct.RangeDim(lower_bound=80, upper_bound= 140, default=80), ct.RangeDim(lower_bound=80, upper_bound=200, default=108)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_AND_GPU,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part3_3.mlpackage')
print("Model saved")

tmp_input = torch.randn([4,512, 136,184])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3-4-1 =====================
tmp_input = torch.randn([4, 256, 160, 216])
# tmp_input = torch.randn([4, 128, 160, 216])

print(tmp_input.shape)

wrapper = VAEWrapper_spatial3_4_1(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

# tmp_input_shape = ct.Shape(shape=(4, 256, ct.RangeDim(lower_bound=100, upper_bound= 800, default=160), ct.RangeDim(lower_bound=100, upper_bound=800, default=216)))
tmp_input_shape = ct.Shape(shape=(4, 256, ct.RangeDim(lower_bound=160, upper_bound= 280, default=160), ct.RangeDim(lower_bound=160, upper_bound=370, default=216)))

converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                             
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part3_4_1.mlpackage')
print("Model saved")

tmp_input = torch.randn([4, 256, 272, 368])
# tmp_input = torch.randn([4, 128, 160, 216])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3-4-2 =====================
tmp_input = torch.randn([4, 128, 160, 216])
# tmp_input = torch.randn([4, 128, 160, 216])

print(tmp_input.shape)

wrapper = VAEWrapper_spatial3_4_2(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

# tmp_input_shape = ct.Shape(shape=(4, 256, ct.RangeDim(lower_bound=100, upper_bound= 800, default=160), ct.RangeDim(lower_bound=100, upper_bound=800, default=216)))
tmp_input_shape = ct.Shape(shape=(4, 128, ct.RangeDim(lower_bound=160, upper_bound= 280, default=160), ct.RangeDim(lower_bound=160, upper_bound=370, default=216)))

converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                             
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part3_4_2.mlpackage')
print("Model saved")

tmp_input = torch.randn([4, 128, 272, 368])
# tmp_input = torch.randn([4, 128, 160, 216])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3-4-3 =====================
tmp_input = torch.randn([4, 128, 160, 216])
# tmp_input = torch.randn([4, 128, 160, 216])

print(tmp_input.shape)

wrapper = VAEWrapper_spatial3_4_3(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

# tmp_input_shape = ct.Shape(shape=(4, 256, ct.RangeDim(lower_bound=100, upper_bound= 800, default=160), ct.RangeDim(lower_bound=100, upper_bound=800, default=216)))
tmp_input_shape = ct.Shape(shape=(4, 128, ct.RangeDim(lower_bound=160, upper_bound= 280, default=160), ct.RangeDim(lower_bound=160, upper_bound=370, default=216)))

converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                             
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part3_4_3.mlpackage')
print("Model saved")

tmp_input = torch.randn([4, 128, 272, 368])
# tmp_input = torch.randn([4, 128, 160, 216])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part4 =====================
tmp_input = torch.randn([4, 128, 160, 216])
print(tmp_input.shape)

wrapper = VAEWrapper_spatial4(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(4, 128, ct.RangeDim(lower_bound=160, upper_bound= 280, default=160), ct.RangeDim(lower_bound=160, upper_bound=370, default=216)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_AND_GPU,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_spatial_part4.mlpackage')
print("Model saved")

tmp_input = torch.randn([4,128, 160, 216])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)