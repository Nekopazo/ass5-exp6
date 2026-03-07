from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)
from diffusers.utils import BaseOutput, logging

from .embeddings import TimestepEmbedding, Timesteps
from .attention import CrossAttention
from .unet_blocks import (DownBlock2D,
                          MCADownBlock2D,
                          UNetMidMCABlock2D,
                          UpBlock2D,
                          StyleUpBlock2D,
                          get_down_block,
                          get_up_block)


logger = logging.get_logger(__name__)

FIXED_STYLE_TRANSFORMER_SCALES = (16, 32, 64)
FIXED_STYLE_LOCAL_MOD_SCALES = ()
FIXED_DISABLED_DOWN_SELF_ATTN_SCALES = (128,)
FIXED_DISABLED_UP_SELF_ATTN_SCALES = (128,)


@dataclass
class UNetOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = None,
        up_block_types: Tuple[str] = None,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 1,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: int = 8,
        channel_attn: bool = False,
        content_encoder_downsample_size: int = 4,
        content_start_channel: int = 16,
        reduction: int = 32,
        mid_enable_content_attn: bool = True,
    ):
        super().__init__()

        self.content_encoder_downsample_size = content_encoder_downsample_size

        self.sample_size = sample_size
        self._style_scales = set(int(x) for x in FIXED_STYLE_TRANSFORMER_SCALES)
        self._local_style_scales = set(int(x) for x in FIXED_STYLE_LOCAL_MOD_SCALES)
        self._disabled_down_self_attn_scales = set(int(x) for x in FIXED_DISABLED_DOWN_SELF_ATTN_SCALES)
        self._disabled_up_self_attn_scales = set(int(x) for x in FIXED_DISABLED_UP_SELF_ATTN_SCALES)
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        def _style_attn_enabled(resolution: Optional[int]) -> bool:
            if resolution is None:
                return False
            return int(resolution) in self._style_scales

        def _style_condition_enabled(resolution: Optional[int]) -> bool:
            if resolution is None:
                return False
            r = int(resolution)
            return (r in self._style_scales) or (r in self._local_style_scales)

        # down
        output_channel = block_out_channels[0]
        current_resolution = int(self.sample_size) if isinstance(self.sample_size, int) else None
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            content_channel = content_start_channel * (2 ** i)
            enable_self_attn = current_resolution not in self._disabled_down_self_attn_scales

            print("Load the down block ", down_block_type)
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                content_channel=content_channel,
                reduction=reduction,
                channel_attn=channel_attn,
                enable_style_attn=False,
                enable_self_attn=enable_self_attn,
            )
            self.down_blocks.append(down_block)
            if current_resolution is not None and not is_final_block:
                current_resolution //= 2

        # mid
        mid_resolution = current_resolution
        self.mid_block = UNetMidMCABlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            channel_attn=channel_attn,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            content_channel=content_start_channel*(2**(content_encoder_downsample_size - 1)),
            reduction=reduction,
            enable_style_attn=_style_attn_enabled(mid_resolution),
            enable_content_attn=bool(mid_enable_content_attn),
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_resolution = mid_resolution
        self.up_block_style_keys: list[str | None] = []
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            content_channel = content_start_channel * (2 ** (content_encoder_downsample_size - i - 1))
            use_local_style_modulation = up_resolution in self._local_style_scales
            enable_self_attn = up_resolution not in self._disabled_up_self_attn_scales

            print("Load the up block ", up_block_type)
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,  # larger 1 than the down block
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                upblock_index=i + 1,
                enable_style_attn=_style_attn_enabled(up_resolution),
                use_local_style_modulation=use_local_style_modulation,
                enable_self_attn=enable_self_attn,
            )
            self.up_blocks.append(up_block)
            self.up_block_style_keys.append(
                f"up_{int(up_resolution)}" if _style_condition_enabled(up_resolution) else None
            )
            prev_output_channel = output_channel
            if up_resolution is not None and add_upsample:
                up_resolution *= 2

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )
        if slice_size is not None and slice_size > self.config.attention_head_dim:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )

        for block in self.down_blocks:
            if (
                hasattr(block, "set_attention_slice")
                and hasattr(block, "self_attentions")
                and hasattr(block, "style_cross_attentions")
                and block.self_attentions is not None
                and block.style_cross_attentions is not None
            ):
                block.set_attention_slice(slice_size)

        if hasattr(self.mid_block, "set_attention_slice"):
            self.mid_block.set_attention_slice(slice_size)

        for block in self.up_blocks:
            if (
                hasattr(block, "set_attention_slice")
                and hasattr(block, "self_attentions")
                and hasattr(block, "style_cross_attentions")
                and block.self_attentions is not None
                and block.style_cross_attentions is not None
            ):
                block.set_attention_slice(slice_size)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func=None,
        value: Optional[bool] = None,
    ):
        # diffusers versions call this hook with either `enable=` or `value=`.
        if value is not None:
            enable = bool(value)
        for module in self.modules():
            if isinstance(module, (DownBlock2D, UpBlock2D, MCADownBlock2D, StyleUpBlock2D)):
                module.gradient_checkpointing = enable

    def set_attention_logging(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.set_attention_logging(bool(enabled))

    def reset_attention_logging(self) -> None:
        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.reset_attention_logging()

    def collect_attention_logging(self) -> dict[int, list[float]]:
        # Aggregate per-context-length token masses over all cross-attn modules.
        total_sum: dict[int, torch.Tensor] = {}
        total_count: dict[int, int] = {}
        for module in self.modules():
            if not isinstance(module, CrossAttention):
                continue
            stats = module.get_attention_logging()
            for klen, payload in stats.items():
                mass_sum, count = payload
                if count <= 0:
                    continue
                if klen not in total_sum:
                    total_sum[klen] = mass_sum.clone()
                    total_count[klen] = int(count)
                else:
                    total_sum[klen] = total_sum[klen] + mass_sum
                    total_count[klen] = int(total_count[klen]) + int(count)

        out: dict[int, list[float]] = {}
        for klen, mass_sum in total_sum.items():
            cnt = max(1, int(total_count.get(klen, 1)))
            avg = (mass_sum / float(cnt)).to(dtype=torch.float64)
            out[int(klen)] = [float(x) for x in avg.tolist()]
        return out

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        content_encoder_downsample_size: int = 4,
        return_dict: bool = False,
    ) -> Union[UNetOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep   # only one time
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)  # projection

        # 2. pre-process
        sample = self.conv_in(sample)

        style_contexts = None
        if encoder_hidden_states is not None and len(encoder_hidden_states) > 2:
            style_contexts = encoder_hidden_states[2]

        # 3. down
        down_block_res_samples = (sample,)
        for index, downsample_block in enumerate(self.down_blocks):
            if (hasattr(downsample_block, "attentions") and downsample_block.attentions is not None) or hasattr(downsample_block, "content_attentions"):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    index=index,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)   

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample, 
                emb, 
                index=content_encoder_downsample_size,
                encoder_hidden_states=encoder_hidden_states,
                style_hidden_states=(
                    style_contexts.get("mid") if isinstance(style_contexts, dict) else style_contexts
                ),
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if isinstance(upsample_block, StyleUpBlock2D):
                style_tokens = None
                if isinstance(style_contexts, dict):
                    key = self.up_block_style_keys[i]
                    if key is not None:
                        style_tokens = style_contexts.get(key)
                else:
                    style_tokens = style_contexts
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=style_tokens,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNetOutput(sample=sample)
