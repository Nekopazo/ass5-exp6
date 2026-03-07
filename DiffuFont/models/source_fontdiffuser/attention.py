from typing import Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    import xformers.ops as xops
except Exception:
    xops = None


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        num_groups: int = 32,
        context_dim: Optional[int] = None,
        use_self_attn: bool = True,
        use_ffn: bool = True,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_self_attn=use_self_attn,
                    use_ffn=use_ffn,
                    ff_mult=ff_mult,
                )
                for d in range(depth)
            ]
        )

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)  # here change the shape torch.Size([1, 4096, 128])
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)  # hidden_states: torch.Size([1, 4096, 128])
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)   # torch.Size([1, 128, 64, 64])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        use_self_attn: bool = True,
        use_ffn: bool = True,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.use_self_attn = bool(use_self_attn)
        self.use_ffn = bool(use_ffn)
        self.attn1 = (
            CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
            if self.use_self_attn
            else None
        )  # self-attention (optional)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout, glu=gated_ff) if self.use_ffn else None
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim) if self.use_ffn else None
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        if self.attn1 is not None:
            self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        if self.attn1 is not None:
            hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        if self.ff is not None and self.norm3 is not None:
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        return self.net(hidden_states)


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class CrossAttention(nn.Module):
    _MAX_EXPLICIT_LOG_SCORE_ELEMENTS = 4_194_304

    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: int = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self._log_attention_enabled = False
        self._attn_mass_sum_by_klen: dict[int, torch.Tensor] = {}
        self._attn_mass_count_by_klen: dict[int, int] = {}
        self._use_memory_efficient_attention_xformers = False
        self._attention_op: Optional[Callable] = None

    def set_attention_logging(self, enabled: bool) -> None:
        self._log_attention_enabled = bool(enabled)

    def set_use_memory_efficient_attention_xformers(
        self,
        valid: bool,
        attention_op: Optional[Callable] = None,
    ) -> None:
        if not valid:
            self._use_memory_efficient_attention_xformers = False
            self._attention_op = None
            return
        if xops is None:
            raise ImportError("xformers is not installed; cannot enable memory efficient attention.")
        self._use_memory_efficient_attention_xformers = True
        self._attention_op = attention_op

    def reset_attention_logging(self) -> None:
        self._attn_mass_sum_by_klen.clear()
        self._attn_mass_count_by_klen.clear()

    def get_attention_logging(self) -> dict[int, tuple[torch.Tensor, int]]:
        out: dict[int, tuple[torch.Tensor, int]] = {}
        for klen, mass_sum in self._attn_mass_sum_by_klen.items():
            out[int(klen)] = (mass_sum.clone(), int(self._attn_mass_count_by_klen.get(klen, 0)))
        return out

    def _record_attention_probs(self, attention_probs: torch.Tensor) -> None:
        # attention_probs: (B*H, Q, K), average over heads and query positions -> (K,)
        klen = int(attention_probs.shape[-1])
        mass = attention_probs.mean(dim=(0, 1)).detach().to(device="cpu", dtype=torch.float64)
        if klen not in self._attn_mass_sum_by_klen:
            self._attn_mass_sum_by_klen[klen] = mass
            self._attn_mass_count_by_klen[klen] = 1
        else:
            self._attn_mass_sum_by_klen[klen] = self._attn_mass_sum_by_klen[klen] + mass
            self._attn_mass_count_by_klen[klen] = int(self._attn_mass_count_by_klen[klen]) + 1

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._log_attention_enabled:
            hidden_states = self._attention(query, key, value)
        elif (
            self._use_memory_efficient_attention_xformers
            or self._slice_size is None
            or query.shape[0] // self._slice_size == 1
        ):
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        return self.to_out(hidden_states)

    def _attention(self, query, key, value):
        # Use explicit attention path when logging is enabled so probabilities are available.
        can_log_explicit = (
            self._log_attention_enabled
            and (int(query.shape[1]) * int(key.shape[1])) <= self._MAX_EXPLICIT_LOG_SCORE_ELEMENTS
        )
        if can_log_explicit:
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attention_probs = attention_scores.softmax(dim=-1)
            self._record_attention_probs(attention_probs)
            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        if self._use_memory_efficient_attention_xformers and xops is not None and query.device.type == "cuda":
            try:
                batch_heads, sequence_length, head_dim = query.shape
                if batch_heads % self.heads != 0:
                    raise ValueError(f"Invalid attention shape: batch_heads={batch_heads}, heads={self.heads}")
                batch_size = batch_heads // self.heads

                q = query.view(batch_size, self.heads, sequence_length, head_dim).permute(0, 2, 1, 3).contiguous()
                k = key.view(batch_size, self.heads, key.shape[1], head_dim).permute(0, 2, 1, 3).contiguous()
                v = value.view(batch_size, self.heads, value.shape[1], head_dim).permute(0, 2, 1, 3).contiguous()

                hidden_states = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=None,
                    p=0.0,
                    op=self._attention_op,
                )
                hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_heads, sequence_length, head_dim)
                hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
                return hidden_states
            except Exception:
                pass

        # Use PyTorch SDPA to avoid explicitly materializing large attention score tensors.
        try:
            batch_heads, sequence_length, head_dim = query.shape
            if batch_heads % self.heads != 0:
                raise ValueError(f"Invalid attention shape: batch_heads={batch_heads}, heads={self.heads}")
            batch_size = batch_heads // self.heads

            q = query.view(batch_size, self.heads, sequence_length, head_dim)
            k = key.view(batch_size, self.heads, key.shape[1], head_dim)
            v = value.view(batch_size, self.heads, value.shape[1], head_dim)

            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.reshape(batch_heads, sequence_length, head_dim)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states
        except Exception:
            # Fallback path for environments where SDPA kernel is unavailable.
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attention_probs = attention_scores.softmax(dim=-1)
            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


class ChannelAttnBlock(nn.Module):
    """This is the Channel Attention in MCA.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=32,
        groups_out=None,
        eps=1e-6,
        non_linearity="swish",
        channel_attn=False,
        reduction=32):
        super().__init__()

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        
        self.channel_attn = channel_attn
        if self.channel_attn:
            # SE Attention
            self.se_channel_attn = SELayer(channel=in_channels, reduction=reduction)

        # Down channel: Use the conv1*1 to down the channel wise
        self.norm3 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.down_channel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) # conv1*1

    def forward(self, input, content_feature):

        concat_feature = torch.cat([input, content_feature], dim=1)
        hidden_states = concat_feature

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.channel_attn:
            hidden_states = self.se_channel_attn(hidden_states)
            hidden_states = hidden_states + concat_feature

        # Down channel
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.down_channel(hidden_states)

        return hidden_states
