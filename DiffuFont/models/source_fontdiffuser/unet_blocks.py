import torch
from torch import nn
import torch.nn.functional as F

from .attention import (SpatialTransformer,
                        ChannelAttnBlock)
from .resnet import (Downsample2D,
                     ResnetBlock2D,
                     Upsample2D)


class PassthroughSpatialTransformer(nn.Module):
    def forward(self, hidden_states, context=None):
        return hidden_states


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, n_heads: int, num_groups: int):
        super().__init__()
        self.attn = SpatialTransformer(
            channels,
            n_heads,
            channels // n_heads,
            depth=1,
            context_dim=None,
            num_groups=num_groups,
            use_self_attn=False,
        )

    def _set_attention_slice(self, slice_size):
        self.attn._set_attention_slice(slice_size)

    def forward(self, hidden_states):
        return self.attn(hidden_states, context=None)


class StyleCrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, n_heads: int, context_dim: int, num_groups: int):
        super().__init__()
        self.attn = SpatialTransformer(
            channels,
            n_heads,
            channels // n_heads,
            depth=1,
            context_dim=context_dim,
            num_groups=num_groups,
            use_self_attn=False,
        )

    def _set_attention_slice(self, slice_size):
        self.attn._set_attention_slice(slice_size)

    def forward(self, hidden_states, context):
        return self.attn(hidden_states, context=context)


class PassthroughChannelAttn(nn.Module):
    def forward(self, hidden_states, content_states=None):
        return hidden_states


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    channel_attn=False,
    content_channel=32,
    reduction=32,
    enable_style_attn=True):

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding)
    elif down_block_type == "MCADownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return MCADownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            channel_attn=channel_attn,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel,
            reduction=reduction,
            enable_style_attn=enable_style_attn)
    else:
        raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    enable_style_attn=True,
    **kwargs,
):

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups)
    elif up_block_type == "StyleUpBlock2D":
        return StyleUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            enable_style_attn=enable_style_attn)
    else:
        raise ValueError(f"{up_block_type} does not exist.")


class UNetMidMCABlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        channel_attn: bool = False,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        content_channel=256,
        reduction=32,
        enable_style_attn: bool = True,
        enable_content_attn: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.enable_style_attn = bool(enable_style_attn)
        self.enable_content_attn = bool(enable_content_attn)
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        content_attentions = []
        self_attentions = []
        style_cross_attentions = []
        post_attn_resnets = []

        for _ in range(num_layers):
            if self.enable_content_attn:
                content_attentions.append(
                    ChannelAttnBlock(
                        in_channels=in_channels + content_channel,
                        out_channels=in_channels,
                        non_linearity=resnet_act_fn,
                        channel_attn=channel_attn,
                        reduction=reduction,
                    )
                )
            else:
                content_attentions.append(PassthroughChannelAttn())
            if self.enable_style_attn:
                self_attentions.append(SelfAttentionBlock(in_channels, attn_num_head_channels, resnet_groups))
                style_cross_attentions.append(
                    StyleCrossAttentionBlock(
                        in_channels,
                        attn_num_head_channels,
                        cross_attention_dim,
                        resnet_groups,
                    )
                )
            else:
                self_attentions.append(PassthroughSpatialTransformer())
                style_cross_attentions.append(PassthroughSpatialTransformer())
            post_attn_resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.content_attentions = nn.ModuleList(content_attentions)
        self.self_attentions = nn.ModuleList(self_attentions)
        self.style_cross_attentions = nn.ModuleList(style_cross_attentions)
        self.post_attn_resnets = nn.ModuleList(post_attn_resnets)
        self.resnets = nn.ModuleList(resnets)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        for attn in self.self_attentions:
            if hasattr(attn, "_set_attention_slice"):
                attn._set_attention_slice(slice_size)
        for attn in self.style_cross_attentions:
            if hasattr(attn, "_set_attention_slice"):
                attn._set_attention_slice(slice_size)

    def forward(
        self, 
        hidden_states, 
        temb=None, 
        encoder_hidden_states=None,
        index=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        for content_attn, self_attn, style_cross_attn, post_resnet in zip(
            self.content_attentions,
            self.self_attentions,
            self.style_cross_attentions,
            self.post_attn_resnets,
        ):
            
            # content
            if self.enable_content_attn:
                current_content_feature = encoder_hidden_states[1][index]
                hidden_states = content_attn(hidden_states, current_content_feature)
            else:
                hidden_states = content_attn(hidden_states, None)
            
            # ResBlock -> Self-Attn -> Cross-Attn(style) -> ResBlock
            hidden_states = self_attn(hidden_states)

            # parts_vector condition is optional; when disabled we skip style cross-attention.
            current_style_feature = None
            if encoder_hidden_states is not None and len(encoder_hidden_states) > 2:
                current_style_feature = encoder_hidden_states[2]
            if current_style_feature is not None:
                hidden_states = style_cross_attn(hidden_states, context=current_style_feature)
            hidden_states = post_resnet(hidden_states, temb)

        return hidden_states


class MCADownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        channel_attn: bool = False,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        content_channel=16,
        reduction=32,
        enable_style_attn: bool = True,
    ):
        super().__init__()
        content_attentions = []
        resnets = []
        self_attentions = []
        style_cross_attentions = []
        post_attn_resnets = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.enable_style_attn = bool(enable_style_attn)

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels+content_channel,
                    out_channels=in_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if self.enable_style_attn:
                print("The style_attention cross attention dim in Down Block {} layer is {}".format(i + 1, cross_attention_dim))
                self_attentions.append(SelfAttentionBlock(out_channels, attn_num_head_channels, resnet_groups))
                style_cross_attentions.append(
                    StyleCrossAttentionBlock(
                        out_channels,
                        attn_num_head_channels,
                        cross_attention_dim,
                        resnet_groups,
                    )
                )
            else:
                self_attentions.append(PassthroughSpatialTransformer())
                style_cross_attentions.append(PassthroughSpatialTransformer())
            post_attn_resnets.append(
                ResnetBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.content_attentions = nn.ModuleList(content_attentions)
        self.self_attentions = nn.ModuleList(self_attentions)
        self.style_cross_attentions = nn.ModuleList(style_cross_attentions)
        self.post_attn_resnets = nn.ModuleList(post_attn_resnets)
        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        for attn in self.self_attentions:
            if hasattr(attn, "_set_attention_slice"):
                attn._set_attention_slice(slice_size)
        for attn in self.style_cross_attentions:
            if hasattr(attn, "_set_attention_slice"):
                attn._set_attention_slice(slice_size)

    def forward(
        self, 
        hidden_states, 
        index,
        temb=None, 
        encoder_hidden_states=None
    ):
        output_states = ()

        for content_attn, resnet, self_attn, style_cross_attn, post_resnet in zip(
            self.content_attentions,
            self.resnets,
            self.self_attentions,
            self.style_cross_attentions,
            self.post_attn_resnets,
        ):
            
            # content — None means skip injection for this block (e.g. Down-0 when encoder runs at 128)
            current_content_feature = encoder_hidden_states[1][index]

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                # Content attention (conditional)
                if current_content_feature is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(content_attn), hidden_states, current_content_feature, use_reentrant=False
                    )
                # First ResBlock
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                # Self-Attn
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self_attn), hidden_states, use_reentrant=False)
                # Style cross-attention (optional)
                current_style_feature = None
                if encoder_hidden_states is not None and len(encoder_hidden_states) > 2:
                    current_style_feature = encoder_hidden_states[2]
                if current_style_feature is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(style_cross_attn), hidden_states, current_style_feature, use_reentrant=False
                    )
                # Second ResBlock
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(post_resnet), hidden_states, temb, use_reentrant=False)
            else:
                # Content attention (conditional)
                if current_content_feature is not None:
                    hidden_states = content_attn(hidden_states, current_content_feature)
                # First ResBlock
                hidden_states = resnet(hidden_states, temb)
                # Self-Attn
                hidden_states = self_attn(hidden_states)

                # parts_vector condition is optional; when disabled we skip style cross-attention.
                current_style_feature = None
                if encoder_hidden_states is not None and len(encoder_hidden_states) > 2:
                    current_style_feature = encoder_hidden_states[2]
                if current_style_feature is not None:
                    hidden_states = style_cross_attn(hidden_states, context=current_style_feature)
                # Second ResBlock
                hidden_states = post_resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class StyleUpBlock2D(nn.Module):
    """Up block with optional style cross-attention (no RSI / deformable conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        add_upsample=True,
        enable_style_attn: bool = True,
    ):
        super().__init__()
        resnets = []
        self_attentions = []
        style_cross_attentions = []
        post_attn_resnets = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.enable_style_attn = bool(enable_style_attn)

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if self.enable_style_attn:
                self_attentions.append(SelfAttentionBlock(out_channels, attn_num_head_channels, resnet_groups))
                style_cross_attentions.append(
                    StyleCrossAttentionBlock(
                        out_channels,
                        attn_num_head_channels,
                        cross_attention_dim,
                        resnet_groups,
                    )
                )
            else:
                self_attentions.append(PassthroughSpatialTransformer())
                style_cross_attentions.append(PassthroughSpatialTransformer())
            post_attn_resnets.append(
                ResnetBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.self_attentions = nn.ModuleList(self_attentions)
        self.style_cross_attentions = nn.ModuleList(style_cross_attentions)
        self.post_attn_resnets = nn.ModuleList(post_attn_resnets)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.self_attentions:
            if hasattr(attn, "_set_attention_slice"):
                attn._set_attention_slice(slice_size)
        for attn in self.style_cross_attentions:
            if hasattr(attn, "_set_attention_slice"):
                attn._set_attention_slice(slice_size)

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        for resnet, self_attn, style_cross_attn, post_resnet in zip(
            self.resnets,
            self.self_attentions,
            self.style_cross_attentions,
            self.post_attn_resnets,
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self_attn), hidden_states, use_reentrant=False)
                if encoder_hidden_states is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(style_cross_attn), hidden_states, encoder_hidden_states, use_reentrant=False
                    )
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(post_resnet), hidden_states, temb, use_reentrant=False)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = self_attn(hidden_states)
                if encoder_hidden_states is not None:
                    hidden_states = style_cross_attn(hidden_states, context=encoder_hidden_states)
                hidden_states = post_resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
