import coremltools as ct
import pickle
import torch
from opensora.registry import MODELS, build_module

vae_config = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
    force_huggingface=True,
)
device = 'cpu'
dtype = torch.float32
vae = build_module(vae_config, MODELS).to(device, dtype).eval()

vae_input_shape = torch.Size([1, 4, 4, 20, 27])
tmp_input = torch.randn(vae_input_shape, dtype=dtype, device=device)
num_frames = 16
num_frames = torch.tensor(num_frames, dtype=torch.int32)

for param in vae.parameters():
    param.requires_grad = False

# Define a wrapper function that includes both inputs
def decode_wrapper(latents, num_frames):
    with torch.no_grad():
        return vae.decode(z=latents, num_frames=num_frames)

# Script the wrapper function
scripted_vae = torch.jit.trace(decode_wrapper, example_inputs=[tmp_input, num_frames])

# converted_vae = ct.converters.convert(scripted_vae, 
#                                 convert_to='mlprogram',
#                                 inputs=[ct.TensorType(name='latents', shape=vae_input_shape), 
#                                         ct.TensorType(name='num_frames', shape=[1])],
#                                 minimum_deployment_target=ct.target.iOS17
#                                 )

# converted_vae.save('vae.mlpackage')