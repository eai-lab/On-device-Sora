import coremltools as ct 
import torch.nn as nn 
import torch 
# from t5 import T5Encoder, T5EncoderModel , T5Embedder 
from transformers.models.t5.modeling_t5 import T5Stack, T5Block
from transformers.models.t5.configuration_t5 import T5Config

# config = T5Config(d_ff=10240,d_kv=64,
#                   d_model=512,decoder_start_token_id=0,
#                   eos_token_id=1,
#                   dense_act_fn='gelu_new',dropout_rate=0.1,
#                   feed_forward_proj='gated-gelu',
#                     initializer_factor=1.0,is_encoder_decoder=True,
#                     layer_norm_epsilon=1e-6,relative_attention_max_distance=128,
#                     model_type='t5', num_decoder_layers=24,num_heads=64,
#                     output_past=True, use_cache=False,pad_token_id=0,
#                     relative_attention_num_buckets=32,
#                     tie_word_embeddings=False,
#                     torch_dtype=torch.float16,
#                     vocab_size=32128,num_layers=24)
config = T5Config(use_cache=False)
model = T5Stack(config,nn.Embedding(32128,512))
model.eval()
x = torch.randint_like(torch.rand(2,512),10).long()
traced_model = torch.jit.trace(model,x)
# cvted_model = ct.convert(traced_model, minimum_deployment_target=ct.target.iOS17, inputs=[ct.TensorType(name='inp', shape=x.shape)])
# cvted_model.save('/Users/embeddedailab/workspace/On-Device-IOS/On-Device-OpenSora/t5_try_trace.mlpackage')


# config = T5Config()
# embedding = nn.Embedding(200,4096).to(dtype=torch.float16)
# model = T5Stack(config,embedding)
# model.eval()
# breakpoint()
# # tracing
# x = torch.randint_like(torch.rand(2,4,16,3,3),10).long()

# timestep = torch.Tensor([999,999])
# # scripted_model = torch.jit.script(architecture)

# traced_model = torch.jit.trace(model,x)