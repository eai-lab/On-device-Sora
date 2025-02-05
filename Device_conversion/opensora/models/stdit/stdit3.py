import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from .rotary_embeddings import RotaryEmbedding

from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    # SeqParallelAttention,
    # SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint


class STDiT3Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            drop_path=0.0,
            rope=None,
            qk_norm=False,
            temporal=False,
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False,
    ):
        super().__init__()

        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        # if self.enable_sequence_parallelism and not temporal:
        #     attn_cls = SeqParallelAttention
        #     mha_cls = SeqParallelMultiHeadCrossAttention
        # else:
        attn_cls = Attention
        mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
            self,
            x,
            y,
            t,
            mask=None,  # text mask
            T=None,  # number of frames
    ):  
        # prepare modulate parameters
        B, N, _ = x.shape
        S = N // T

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            

        # attention
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        
        # apply token merging Case
        x = x + self.cross_attn(x, y, mask)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m


        # residual
        x = x + self.drop_path(x_m_s)

        return x
    
    def forwardTDTM(
            self,
            x,
            y,
            t,
            mask=None,  # text mask
            T=None,  # number of frames
    ):  
        # prepare modulate parameters
        B, N, _ = x.shape
        S = N // T

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            

        # attention
        # if self.temporal:
        #     x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        #     x_m = self.attn(x_m)
        #     x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        # else:
        #     x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        #     x_m = self.attn(x_m)
        #     x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
        if self.temporal:
            # Time mean
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = (x_m[:, 0::2, :] + x_m[:, 1::2, :]) / 2
                x_m = self.attn(x_m)
                BS, _, C = x_m.shape
                x_m = x_m.unsqueeze(2).expand(-1, -1, 2, -1)  # (BS, T//2, 2, C)
                x_m = x_m.contiguous().view(BS, T, C)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
                # origin
                # x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                # x_m = self.attn(x_m)
                # x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            # Time mean
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = (x_m[:, 0::2, :] + x_m[:, 1::2, :]) / 2
                new_T = x_m.shape[1]
                x_m = rearrange(x_m, "(B S) T C -> (B T) S C", T=new_T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> (B S) T C", T=new_T, S=S)
                BS, _, C = x_m.shape
                x_m = x_m.unsqueeze(2).expand(-1, -1, 2, -1)  # (BS, T//2, 2, C)
                x_m = x_m.contiguous().view(BS, T, C)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            # origin
                # x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                # x_m = self.attn(x_m)
                # x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
        # modulate (attention)
        x_m_s = gate_msa * x_m

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        
        # apply token merging Case
        x_m = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=S)
        x_m = (x_m[:, 0::2, :] + x_m[:, 1::2, :]) / 2
        new_T = x_m.shape[1]
        x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=new_T, S=S)
        x_m = self.cross_attn(x_m, y, mask)
        x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T= new_T, S=S) 
        BS, _, C = x_m.shape
        x_m = x_m.unsqueeze(2).expand(-1, -1, 2, -1)  # (BS, T//2, 2, C)
        x_m = x_m.contiguous().view(BS, T, C)
        x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        x = x + x_m

        # Original Case
        # x = x + self.cross_attn(x, y, mask)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m


        # residual
        x = x + self.drop_path(x_m_s)

        return x

class STDiT3Config(PretrainedConfig):
    model_type = "STDiT3"

    def __init__(
            self,
            input_size=(None, None, None),
            input_sq_size=512,
            in_channels=4,
            patch_size=(1, 2, 2),
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            pred_sigma=True,
            drop_path=0.0,
            caption_channels=4096,
            model_max_length=300,
            qk_norm=True,
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False,
            only_train_temporal=False,
            freeze_y_embedder=False,
            skip_y_embedder=False,
            **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.only_train_temporal = only_train_temporal
        self.freeze_y_embedder = freeze_y_embedder
        self.skip_y_embedder = skip_y_embedder
        super().__init__(**kwargs)


class STDiT3(PreTrainedModel):
    config_class = STDiT3Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                )
                for i in range(config.depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, int(np.prod(self.patch_size)), self.out_channels)

        self.initialize_weights()
        if config.only_train_temporal:
            for param in self.parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False
        for param in self.y_embedder.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize temporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()

        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            # mask = mask.squeeze(1).squeeze(1)
            # Before:
            # y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)

            # After:
            y = y.squeeze(1)
            mask_expanded = mask.unsqueeze(-1).expand_as(y)
            y = y[mask_expanded != 0].view(1, -1, self.hidden_size)
            # y_lens = mask.sum(dim=1).tolist()
            # y_lens = torch.tensor([mask.sum(dim=1)[0]])
            return y
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
            return y
        # return y, y_lens

    # def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None):
        dtype = self.x_embedder.proj.weight.dtype
        height = torch.Tensor([224]).to(dtype) if height is None else height
        width = torch.Tensor([224]).to(dtype) if width is None else width
        fps = torch.Tensor([24]).to(dtype) if fps is None else fps
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        base_size = torch.round(torch.Tensor([S ** 0.5]))
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size

        x = x.reshape(x.shape[0], N_t * N_h * N_w, H_p * W_p, self.out_channels)
        x = x.permute(0, 3, 1, 2)  # [B, C_out, N, P]
        B, _, _, _ = x.shape
        # 2. 공간 차원 복원
        x = x.reshape(x.shape[0], self.out_channels, N_t, N_h * N_w, H_p * W_p)
        x = x.reshape(x.shape[0] * self.out_channels, N_t, N_h, N_w, H_p * W_p)
        x = x.permute(0, 1, 2, 4, 3)  # [B * C_out, T, P, H, W]
        x = x.reshape(x.shape[0], N_t, N_h *H_p, W_p, N_w )
        x = x.permute(0, 1, 2, 4, 3)  # [B * C_out, T, H, W, P]
        x = x.reshape(x.shape[0], N_t, N_h *H_p, N_w * W_p)
        x = x.reshape(B , self.out_channels, N_t, N_h *H_p, N_w * W_p)

        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x
    
@MODELS.register_module("STDiT3-XL/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = STDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = STDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model

class custom_1_STDit3(STDiT3):
    def forward(self, x, timestep, y, mask, fps, height, width, pad_H, pad_W):
        dtype = self.x_embedder.proj.weight.dtype
        # B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        

        # === get pos embed ===
        B, _, T, Hx, Wx = x.size()
        pad_H = pad_H.item()
        pad_W = pad_W.item()
        H = (Hx + pad_H) // self.patch_size[1]
        W = (Wx + pad_W) // self.patch_size[2]
        S = H * W
        # T, H, W = self.get_dynamic_size(x)
        # S = H * W
        base_size = torch.round(torch.Tensor([S ** 0.5]))
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        # t0 = t0_mlp = None
        t0 = None
        # if x_mask is not None:
        t0_timestep = torch.zeros_like(timestep)
        t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
        t0 = t0 + fps
        # t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x, pad_H, pad_W)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        return (x, y, t_mlp, T, t)
    
@MODELS.register_module("STDiT3-XL-Custom-1/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = custom_1_STDit3.from_pretrained(from_pretrained, **kwargs)
        print("Pretrained model loaded")
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = custom_1_STDit3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model

class custom_st_STDit3(STDiT3):
    def __init__(self, config):
        super().__init__(config)
        self.spatial_block = self.spatial_blocks[0]
        self.temporal_block = self.temporal_blocks[0]

    def forward(self, x, y, t_mlp, mask, T):
        T = T.to(torch.int32).item()

        x = self.spatial_block(x, y, t_mlp,  mask, T)
        x = self.temporal_block(x, y, t_mlp, mask, T)
        return x
    
    

@MODELS.register_module("STDiT3-XL-Custom-ST/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = custom_st_STDit3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = custom_st_STDit3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model

class custom_st_STDit3TDTM(STDiT3):
    def __init__(self, config):
        super().__init__(config)
        self.spatial_block = self.spatial_blocks[0]
        self.temporal_block = self.temporal_blocks[0]

    def forward(self, x, y, t_mlp, mask, T):
        T = T.to(torch.int32).item()

        x = self.spatial_block.forwardTDTM(x, y, t_mlp,  mask, T)
        x = self.temporal_block.forwardTDTM(x, y, t_mlp, mask, T)
        return x
    
    

@MODELS.register_module("STDiT3-XL-Custom-TDTM/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = custom_st_STDit3TDTM.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = custom_st_STDit3TDTM(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model


class custom_2_STDit3(STDiT3):
    def forward(self, z_in, x, t, pad_H, pad_W):

        _, _, T, Hx, Wx = z_in.size()
        pad_H = pad_H.item()
        pad_W = pad_W.item()
        H = (Hx + pad_H) // self.patch_size[1]
        W = (Wx + pad_W) // self.patch_size[2]

        # === final layer ===
        x = self.final_layer(x, t)
        x = self.unpatchify(x, T, H, W, T, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

@MODELS.register_module("STDiT3-XL-Custom-2/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = custom_2_STDit3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = custom_2_STDit3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
