from transformers.models.t5.modeling_t5 import T5Block, T5Config, T5LayerNorm
import torch
import coremltools as ct 
import pickle
import torch.nn as nn

num_layers = 24
t5_block_config_path = "/{PATH}"
with open(t5_block_config_path, 'rb') as f:
    t5_block_config = pickle.load(f)

"""
T5Config {
  "_name_or_path": "DeepFloyd/t5-v1_1-xxl",
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": false,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.36.2",
  "use_cache": false,
  "vocab_size": 32128
}
"""

# t5 stack
blocks = nn.ModuleList(
    [T5Block(t5_block_config, has_relative_attention_bias=bool(i == 0)) for i in range(num_layers)]
)
blocks.eval()

for i, block in enumerate(blocks):
    save_path = "/{PATH}"
    _save_path = f"{save_path}/layer_{i}"

    with open(f"{_save_path}/hidden_states.pkl", "rb") as f:
        hidden_states = pickle.load(f)
    with open(f"{_save_path}/extended_attention_mask.pkl", "rb") as f:
        extended_attention_mask = pickle.load(f)
    with open(f"{_save_path}/position_bias.pkl", "rb") as f:
        position_bias = pickle.load(f)
    with open(f"{_save_path}/encoder_hidden_states.pkl", "rb") as f:
        encoder_hidden_states = pickle.load(f)
    with open(f"{_save_path}/encoder_extended_attention_mask.pkl", "rb") as f:
        encoder_extended_attention_mask = pickle.load(f)
    with open(f"{_save_path}/encoder_decoder_position_bias.pkl", "rb") as f:
        encoder_decoder_position_bias = pickle.load(f)
    with open(f"{_save_path}/layer_head_mask.pkl", "rb") as f:
        layer_head_mask = pickle.load(f)
    with open(f"{_save_path}/cross_attn_layer_head_mask.pkl", "rb") as f:
        cross_attn_layer_head_mask = pickle.load(f)
    with open(f"{_save_path}/past_key_value.pkl", "rb") as f:
        pask_key_value = pickle.load(f)
    with open(f"{_save_path}/use_cache.pkl", "rb") as f:
        use_cache = pickle.load(f)
    with open(f"{_save_path}/output_attentions.pkl", "rb") as f:
        output_attentions = pickle.load(f)
        
    _inputs = {
        "hidden_states": hidden_states,
        "attention_mask": extended_attention_mask,
        "position_bias": position_bias,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_extended_attention_mask": encoder_extended_attention_mask,
        "encoder_decoder_position_bias": encoder_decoder_position_bias,
        "layer_head_mask": layer_head_mask,
        "cross_attn_layer_head_mask": cross_attn_layer_head_mask,
        "past_key_value": pask_key_value,
        "use_cache": use_cache,
        "output_attentions": output_attentions
    }
    _inputs = {k: v for k, v in _inputs.items() if v is not None}
    # print(_inputs)
    
    # print("layer: ", i)
    if i == 0:
        scripted_block = torch.jit.trace(block, (
            hidden_states.to('cpu'),
            extended_attention_mask.to('cpu'),
        ))
        # converted_model = ct.converters.convert(scripted_block,
        #                                 convert_to='mlprogram',
        #                                 inputs=[ct.TensorType(name='hidden_states', shape=hidden_states.shape),
        #                                         ct.TensorType(name='attention_mask', shape=extended_attention_mask.shape)],
        #                                 minimum_deployment_target=ct.target.iOS17)
    else:
        scripted_block = torch.jit.trace(block, (
            hidden_states.to('cpu'),
            extended_attention_mask.to('cpu'),
            position_bias.detach().to('cpu'),
        ))
        # converted_model = ct.converters.convert(scripted_block,
        #                                 convert_to='mlprogram',
        #                                 inputs=[ct.TensorType(name='hidden_states', shape=hidden_states.shape),
        #                                         ct.TensorType(name='attention_mask', shape=extended_attention_mask.shape),
        #                                         ct.TensorType(name='position_bias', shape=position_bias.shape)],
        #                                 minimum_deployment_target=ct.target.iOS17)
    # converted_model.save(f't5block-layer{i}.mlpackage')