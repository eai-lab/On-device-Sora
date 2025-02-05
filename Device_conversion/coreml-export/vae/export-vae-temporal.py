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

class VAEWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, z):
        x_z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)
        # x_z = self.temporal_vae.decode(z, num_frames=self.micro_frame_size)
        x_z = self.temporal_vae.part1(x_z)
        # torch.Size([1, 512, 5, 20, 27])
        x_z = self.temporal_vae.part2(x_z)
        # torch.Size([1, 128, 20, 20, 27])
        x_z = self.temporal_vae.part3(x_z, num_frames=self.micro_frame_size)
        return x_z
    
class VAEWrapper1(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, z):
        # torch.Size([1, 4, 5, 20, 27])
        x_z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)
        x_z = self.temporal_vae.part1(x_z)
        return x_z    
    
class VAEWrapper2_1(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 512, 5, 20, 27])
        x_z = self.temporal_vae.part2_1(x_z)
        return x_z

class VAEWrapper2_2(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 512, 10, 20, 27])
        x_z = self.temporal_vae.part2_2(x_z)
        return x_z

class VAEWrapper2_3_1(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 256, 20, 20, 27])
        x_z = self.temporal_vae.part2_3_1(x_z)
        return x_z
class VAEWrapper2_3_2(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 256, 20, 20, 27])
        x_z = self.temporal_vae.part2_3_2(x_z)
        return x_z
class VAEWrapper2_3_3(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 256, 20, 20, 27])
        x_z = self.temporal_vae.part2_3_3(x_z)
        return x_z
class VAEWrapper2_3_4(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 256, 20, 20, 27])
        x_z = self.temporal_vae.part2_3_4(x_z)
        return x_z

class VAEWrapper2_4(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 256, 20, 20, 27])
        x_z = self.temporal_vae.part2_4(x_z)
        return x_z
    
class VAEWrapper3(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.temporal_vae = vae.temporal_vae
        self.micro_z_frame_size = vae.micro_z_frame_size
        self.micro_frame_size = vae.micro_frame_size
        self.scale = vae.scale
        self.shift = vae.shift
    def forward(self, x_z):
        # torch.Size([1, 128, 20, 20, 27])
        x_z = self.temporal_vae.part3(x_z, num_frames=self.micro_frame_size)
        return x_z    

# ===================== temporal model ===================== 

# ===================== part1 =====================
tmp_input = torch.randn([1, 4, 5, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper1(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 4, 5, ct.RangeDim(lower_bound=20, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae, 
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part1.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,4, 5, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-1 =====================
tmp_input = torch.randn([1, 512, 5, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_1(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 512, 5, ct.RangeDim(lower_bound=20, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae, 
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_1.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,512, 5, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-2 =====================
tmp_input = torch.randn([1, 512, 10, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_2(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 512, 10, ct.RangeDim(lower_bound=20, upper_bound= 50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                # compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_2.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,512, 10, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-3-1 =====================
tmp_input = torch.randn([1, 256, 20, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_3_1(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 256, 20, ct.RangeDim(lower_bound=20, upper_bound=50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_3_1.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,256, 20, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-3-2 =====================
tmp_input = torch.randn([1, 256, 20, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_3_2(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 256, 20, ct.RangeDim(lower_bound=20, upper_bound=50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_3_2.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,256, 20, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-3-3 =====================
tmp_input = torch.randn([1, 256, 20, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_3_3(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 256, 20, ct.RangeDim(lower_bound=20, upper_bound=50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_3_3.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,256, 20, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-3-4 =====================
tmp_input = torch.randn([1, 256, 20, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_3_4(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 256, 20, ct.RangeDim(lower_bound=20, upper_bound=50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_3_4.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,256, 20, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part2-4 =====================
tmp_input = torch.randn([1, 256, 20, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper2_4(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 256, 20, ct.RangeDim(lower_bound=20, upper_bound=50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part2_4.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,256, 20, 34,46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)

# ===================== part3 =====================

tmp_input = torch.randn([1, 128, 20, 20, 27])
print(tmp_input.shape)

wrapper = VAEWrapper3(vae).eval()
scripted_vae = torch.jit.trace(wrapper, (tmp_input))
print("Model traced")

tmp_input_shape = ct.Shape(shape=(1, 128, 20, ct.RangeDim(lower_bound=20, upper_bound=50, default=20), ct.RangeDim(lower_bound=20, upper_bound=50, default=27)))
converted_vae = ct.converters.convert(scripted_vae,
                                convert_to='mlprogram',
                                inputs=[ct.TensorType(name='latents', shape=tmp_input_shape)],
                                outputs=[ct.TensorType(name='output')],
                                compute_units= ct.ComputeUnit.CPU_ONLY,                                
                                minimum_deployment_target=ct.target.iOS18,
                                compute_precision=ct.precision.FLOAT32
                                )

print("Model converted")
converted_vae.save('vae_temporal_part3.mlpackage')
print("Model saved")

tmp_input = torch.randn([1,128, 20, 34, 46])
result = converted_vae.predict({'latents': tmp_input})
print(result["output"].shape)

time.sleep(1)