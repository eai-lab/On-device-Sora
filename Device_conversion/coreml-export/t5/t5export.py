import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Block, T5Config, T5LayerNorm
from typing import Tuple
from torch import Tensor
import coremltools as ct
import pickle
from coremltools.models.neural_network import quantization_utils
import warnings
warnings.filterwarnings("ignore")
            
class customT5Block_0(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        with open('y_embedding.pkl', 'rb') as f:
            y = pickle.load(f)
        self.y_embedding = nn.Parameter(y)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        y_embedding=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clip(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clip(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clip(hidden_states, min=-clamp_value, max=clamp_value)
        
        outputs = (hidden_states,)
        y_embedding *= self.y_embedding
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs + (y_embedding,)

        return outputs   # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)        

class customT5Block_n0(T5Block):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clip(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clip(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clip(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)        

def get_extended_attention_mask(
        attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
       
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
        # extended_attention_mask = (1.0 - extended_attention_mask)
        return extended_attention_mask


class EmbedTokensWrapper(torch.nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)
    
class CustomT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32
        eps = 1e-06
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)

        return self.weight * hidden_states

def export_t5_components(model_name):
    # Load the T5 model
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell",
                       max_length=300,
                       padding="max_length",
                       truncation=True,
                       return_attention_mask=True,
                       add_special_tokens=True,
                       return_tensors="pt",)
    
    attention_mask = inputs["attention_mask"]
    inputs = inputs["input_ids"]
    model.eval()
    minimum_deployment_target = ct.target.iOS17
    compute_precision = ct.precision.FLOAT32

    # Export embed_tokens
    embed_tokens = model.encoder.embed_tokens
    embed_tokens = EmbedTokensWrapper(embed_tokens)
    embed_tokens.eval()
    embed_tokens_traced = torch.jit.script(embed_tokens)
    converted_model = ct.converters.convert(embed_tokens_traced,
                                            convert_to='mlprogram',
                                            inputs=[ct.TensorType(name='input_ids', shape=inputs.shape)],
                                            outputs=[ct.TensorType(name='output')],
                                            compute_precision=compute_precision,
                                            compute_units=ct.ComputeUnit.CPU_AND_GPU,                                            
                                            minimum_deployment_target=minimum_deployment_target)
    converted_model.save('mlpackage/t5embed-tokens.mlpackage')

    embed_tokens_output = model.encoder.embed_tokens(inputs)
    input_ids = inputs
    input_shape = input_ids.size()
    # Get the embeddings from the embedding layer
    embed_tokens_output = model.encoder.embed_tokens(input_ids)
    device = input_ids.device
    extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, device)

    # Export encoder layers
    position_bias = None
    hidden_states = embed_tokens_output
    t5_block_config_path = "./configs/T5BlockConfig.pkl"
    with open(t5_block_config_path, 'rb') as f:
        t5_block_config = pickle.load(f)

    for i, b in enumerate(model.encoder.block):
        print("layer: ", i)
        if i == 0:
            block = customT5Block_0(t5_block_config, has_relative_attention_bias=bool(i == 0))
        else:
            block = customT5Block_n0(t5_block_config, has_relative_attention_bias=bool(i == 0))
        block.load_state_dict(b.state_dict(), strict= False)
        block.eval()
        if i == 0:
            with open('y_embedding.pkl', 'rb') as f:
                y_input = pickle.load(f)
            y_embedding = torch.ones_like(y_input)
            scripted_block = torch.jit.trace(block, (
                hidden_states,
                extended_attention_mask,
                y_embedding
            ))
            converted_model = ct.converters.convert(scripted_block,
                                            convert_to='mlprogram',
                                            inputs=[ct.TensorType(name='hidden_states', shape=hidden_states.shape),
                                                    ct.TensorType(name='attention_mask', shape=extended_attention_mask.shape),
                                                    ct.TensorType(name='y_embedding', shape=y_embedding.shape)],
                                            outputs=[ct.TensorType(name='output_hidden_states'), ct.TensorType(name='output_position_bias'), ct.TensorType(name='yNull')],
                                            compute_precision=compute_precision,
                                            compute_units=ct.ComputeUnit.CPU_AND_GPU,
                                            minimum_deployment_target=minimum_deployment_target
                                            )
            # converted_model = quantization_utils.quantize_weights(converted_model, nbits=16)
            result = block(hidden_states, extended_attention_mask, y_embedding)
            hidden_states = result[0]
            position_bias = result[1]
            y = result[2]
            converted_model.save(f'mlpackage/t5block-layer{i}.mlpackage')
        else:
            scripted_block = torch.jit.trace(block, (
                hidden_states,
                extended_attention_mask,
                position_bias,
            ))
            converted_model = ct.converters.convert(scripted_block,
                                            convert_to='mlprogram',
                                            inputs=[ct.TensorType(name='hidden_states', shape=hidden_states.shape),
                                                    ct.TensorType(name='attention_mask', shape=extended_attention_mask.shape),
                                                    ct.TensorType(name='position_bias', shape=position_bias.shape)],
                                            outputs=[ct.TensorType(name='output_hidden_states')],
                                            compute_units=ct.ComputeUnit.CPU_AND_GPU,
                                            compute_precision=compute_precision,
                                            minimum_deployment_target=minimum_deployment_target
                                            )            
            result = block(hidden_states, extended_attention_mask, position_bias)
            hidden_states = result[0]
            converted_model.save(f'mlpackage/t5block-layer{i}.mlpackage')

    # Export final_layer_norm
    custom_layer_norm = CustomT5LayerNorm(model.config.d_model, model.config.layer_norm_epsilon)
    custom_layer_norm.load_state_dict(model.encoder.final_layer_norm.state_dict())
    custom_layer_norm.eval()
    
    # converting layer norm...
    scripted_final_layer_norm = torch.jit.script(custom_layer_norm)
    converted_model = ct.converters.convert(scripted_final_layer_norm,
                                            convert_to='mlprogram',
                                            inputs=[ct.TensorType(name='input', shape=hidden_states.shape)],
                                            outputs=[ct.TensorType(name='output')],
                                            compute_precision=compute_precision,
                                            compute_units=ct.ComputeUnit.CPU_AND_GPU,
                                            minimum_deployment_target=minimum_deployment_target)
    # print the base layer norm model
    converted_model.save('mlpackage/t5final-layer-norm.mlpackage')

if __name__ == "__main__":
    export_t5_components("DeepFloyd/t5-v1_1-xxl")



