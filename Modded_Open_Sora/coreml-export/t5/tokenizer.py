import torch
import coremltools as ct 
import torch.nn as nn

import torch


# def string(x:str):
#     x += "world"
#     return x

# # traced = torch.jit.trace(string, ["hello"])
# script = torch.jit.script(string)
# out = script("hello")
# print(out)

from transformers import AutoTokenizer, T5EncoderModel
from transformers import T5TokenizerFast, T5Tokenizer

# tokenizer = T5Tokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl", torchscript=True)
tokenizer = T5TokenizerFast.from_pretrained("DeepFloyd/t5-v1_1-xxl", torchscript=True)

texts = ["Hello, my dog is cute"]
max_length = 300
padding="max_length"
truncation=True
return_attention_mask=True
add_special_tokens=True
return_tensors="pt"

text_tokens_and_mask = tokenizer(
    texts,
    max_length=max_length,
    padding=padding,
    truncation=truncation,
    return_attention_mask=return_attention_mask,
    add_special_tokens=add_special_tokens,
    return_tensors=return_tensors
)
script = torch.jit.script(tokenizer)

# scripted_tokenizer = torch.jit.trace(tokenizer.tokenize, (
#     texts,
#     max_length,
#     padding,
#     truncation,
#     return_attention_mask,
#     add_special_tokens,
#     return_tensors
# ))

# text_tokens_and_mask = self.tokenizer(
#     texts,
#     max_length=self.model_max_length,
#     padding="max_length",
#     truncation=True,
#     return_attention_mask=True,
#     add_special_tokens=True,
#     return_tensors="pt",
# )

# # converted_model = ct.converters.convert(torchscript_model,
# #                                     convert_to='mlprogram',
# #                                     inputs=[ct.TensorType(name='input', shape=x.shape),
# #                                             ct.TensorType(name='timestep', shape=timestep.shape),
# #                                             ct.TensorType(name='y', shape=y.shape)],
# #                                     minimum_deployment_target=ct.target.iOS17)
# # converted_model.save('stdit3.mlpackage')