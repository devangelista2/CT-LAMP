from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from einops import rearrange
from torch import nn


def _norm_groups(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.to(dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Timesteps(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _timestep_embedding(timesteps, self.dim)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.linear_1.weight.dtype)
        return self.linear_2(self.act(self.linear_1(x)))


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_emb_proj(self.act(temb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.conv_shortcut(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, head_dim: int = 8):
        super().__init__()
        self.group_norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.head_dim = head_dim if channels % head_dim == 0 else channels
        self.num_heads = max(1, channels // self.head_dim)
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.Sequential(nn.Linear(channels, channels, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.group_norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.to_out(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x_in + out


class DepthAttention(nn.Module):
    def __init__(self, channels: int, head_dim: int = 8):
        super().__init__()
        self.group_norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.head_dim = head_dim if channels % head_dim == 0 else channels
        self.num_heads = max(1, channels // self.head_dim)
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.Sequential(nn.Linear(channels, channels, bias=True))
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        bxd, c, h, w = x.shape
        if depth <= 0 or bxd % depth != 0:
            raise ValueError(f"Depth {depth} must divide batch {bxd}.")
        b = bxd // depth
        x_in = x
        x = rearrange(x, "(b d) c h w -> b c d h w", b=b, d=depth)
        x = self.group_norm(x)
        x = rearrange(x, "b c d h w -> (b h w) d c")

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(-1, depth, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, depth, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, depth, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(-1, depth, c)
        out = self.to_out(out)
        out = rearrange(out, "(b h w) d c -> (b d) c h w", b=b, h=h, w=w, d=depth)
        return x_in + out


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_dim: int,
        add_downsample: bool,
        add_attention: bool,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if add_attention else None
        for i in range(num_layers):
            resnet = ResnetBlock2D(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
            )
            self.resnets.append(resnet)
            if self.attentions is not None:
                self.attentions.append(SpatialAttention(out_channels))
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels)])
            if add_downsample
            else nn.ModuleList()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        res_samples: List[torch.Tensor] = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x)
            res_samples.append(x)
        if len(self.downsamplers) > 0:
            x = self.downsamplers[0](x)
            res_samples.append(x)
        return x, res_samples


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: List[int],
        time_emb_dim: int,
        add_upsample: bool,
        add_attention: bool,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if add_attention else None

        for i, skip_ch in enumerate(skip_channels):
            resnet = ResnetBlock2D(
                in_channels + skip_ch if i == 0 else out_channels + skip_ch,
                out_channels,
                time_emb_dim,
            )
            self.resnets.append(resnet)
            if self.attentions is not None:
                self.attentions.append(SpatialAttention(out_channels))
            in_channels = out_channels

        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)])
            if add_upsample
            else nn.ModuleList()
        )

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor, res_samples: List[torch.Tensor]
    ):
        for i, resnet in enumerate(self.resnets):
            res_hidden = res_samples.pop()
            x = torch.cat([x, res_hidden], dim=1)
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x)
        if len(self.upsamplers) > 0:
            x = self.upsamplers[0](x)
        return x


class MidBlock2D(nn.Module):
    def __init__(self, channels: int, time_emb_dim: int, add_attention: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(channels, channels, time_emb_dim),
                ResnetBlock2D(channels, channels, time_emb_dim),
            ]
        )
        self.attentions = nn.ModuleList(
            [SpatialAttention(channels)] if add_attention else []
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x, temb)
        if len(self.attentions) > 0:
            x = self.attentions[0](x)
        x = self.resnets[1](x, temb)
        return x


@dataclass
class UNet2DOutput:
    sample: torch.Tensor


class UNet2DModel(nn.Module):
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int = 2,
        block_out_channels: Tuple[int, ...] | List[int] = (128, 256, 512, 512),
        down_block_types: Tuple[str, ...] | List[str] = None,
        up_block_types: Tuple[str, ...] | List[str] = None,
    ):
        super().__init__()
        block_out_channels = list(block_out_channels)
        down_block_types = list(
            down_block_types or ["DownBlock2D"] * len(block_out_channels)
        )
        up_block_types = list(up_block_types or ["UpBlock2D"] * len(block_out_channels))

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0])
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            add_attn = "Attn" in down_block_types[i]
            add_downsample = i != len(block_out_channels) - 1
            down_block = DownBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                num_layers=layers_per_block,
                time_emb_dim=time_embed_dim,
                add_downsample=add_downsample,
                add_attention=add_attn,
            )
            self.down_blocks.append(down_block)
            in_ch = out_ch

        self.mid_block = MidBlock2D(in_ch, time_embed_dim, add_attention=True)

        skip_channels: List[int] = [block_out_channels[0]]
        for i, out_ch in enumerate(block_out_channels):
            skip_channels.extend([out_ch] * layers_per_block)
            if i != len(block_out_channels) - 1:
                skip_channels.append(out_ch)

        self.up_blocks = nn.ModuleList()
        skip_channels = list(reversed(skip_channels))
        in_ch = block_out_channels[-1]
        for i, out_ch in enumerate(reversed(block_out_channels)):
            add_attn = "Attn" in up_block_types[i]
            add_upsample = i != len(block_out_channels) - 1
            num_resnets = layers_per_block + 1
            skip_slice = skip_channels[i * num_resnets : (i + 1) * num_resnets]
            up_block = UpBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                skip_channels=skip_slice,
                time_emb_dim=time_embed_dim,
                add_upsample=add_upsample,
                add_attention=add_attn,
            )
            self.up_blocks.append(up_block)
            in_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.block_out_channels = tuple(block_out_channels)
        self.down_block_types = tuple(down_block_types)
        self.up_block_types = tuple(up_block_types)
        self.config = {
            "sample_size": sample_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "layers_per_block": layers_per_block,
            "block_out_channels": list(block_out_channels),
            "down_block_types": list(down_block_types),
            "up_block_types": list(up_block_types),
        }

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> UNet2DOutput:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps.expand(sample.shape[0])

        temb = self.time_proj(timesteps)
        temb = self.time_embedding(temb)

        x = self.conv_in(sample)
        down_block_res_samples: List[torch.Tensor] = [x]

        for down_block in self.down_blocks:
            x, res_samples = down_block(x, temb)
            down_block_res_samples.extend(res_samples)

        x = self.mid_block(x, temb)

        for up_block in self.up_blocks:
            res_count = len(up_block.resnets)
            res_samples = down_block_res_samples[-res_count:]
            down_block_res_samples = down_block_res_samples[:-res_count]
            x = up_block(x, temb, res_samples)

        x = self.conv_out(torch.nn.functional.silu(self.conv_norm_out(x)))
        return UNet2DOutput(sample=x)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "UNet2DModel":
        path = Path(path)
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(**config)
        state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state, strict=True)
        return model


class UNet2DTemporalAdapter(UNet2DModel):
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int = 2,
        block_out_channels: Tuple[int, ...] | List[int] = (128, 256, 512, 512),
        down_block_types: Tuple[str, ...] | List[str] = None,
        up_block_types: Tuple[str, ...] | List[str] = None,
        depth_attn_head_dim: int = 8,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        self.depth_attn_head_dim = depth_attn_head_dim
        self.config["depth_attn_head_dim"] = depth_attn_head_dim

        mid_channels = self.block_out_channels[-1]
        self.mid_depth_attn = nn.ModuleList()
        self.mid_depth_attn.append(
            DepthAttention(mid_channels, head_dim=depth_attn_head_dim)
        )
        if len(self.mid_block.attentions) > 0:
            self.mid_depth_attn.append(
                DepthAttention(mid_channels, head_dim=depth_attn_head_dim)
            )
        self.mid_depth_attn.append(
            DepthAttention(mid_channels, head_dim=depth_attn_head_dim)
        )

        self.up_depth_attn = nn.ModuleList()
        for up_block in self.up_blocks:
            out_channels = up_block.resnets[0].conv2.out_channels
            blocks = nn.ModuleList()
            for _ in range(len(up_block.resnets)):
                blocks.append(
                    DepthAttention(out_channels, head_dim=depth_attn_head_dim)
                )
                if up_block.attentions is not None:
                    blocks.append(
                        DepthAttention(out_channels, head_dim=depth_attn_head_dim)
                    )
            self.up_depth_attn.append(blocks)

    def forward(
        self, sample: torch.Tensor, timesteps: torch.Tensor, depth: int
    ) -> UNet2DOutput:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps.expand(sample.shape[0])

        temb = self.time_proj(timesteps)
        temb = self.time_embedding(temb)

        x = self.conv_in(sample)
        down_block_res_samples: List[torch.Tensor] = [x]

        for down_block in self.down_blocks:
            x, res_samples = down_block(x, temb)
            down_block_res_samples.extend(res_samples)

        mid_idx = 0
        x = self.mid_block.resnets[0](x, temb)
        x = self.mid_depth_attn[mid_idx](x, depth)
        mid_idx += 1
        if len(self.mid_block.attentions) > 0:
            x = self.mid_block.attentions[0](x)
            x = self.mid_depth_attn[mid_idx](x, depth)
            mid_idx += 1
        x = self.mid_block.resnets[1](x, temb)
        x = self.mid_depth_attn[mid_idx](x, depth)

        for up_idx, up_block in enumerate(self.up_blocks):
            res_count = len(up_block.resnets)
            res_samples = down_block_res_samples[-res_count:]
            down_block_res_samples = down_block_res_samples[:-res_count]
            depth_blocks = self.up_depth_attn[up_idx]
            depth_i = 0
            for i, resnet in enumerate(up_block.resnets):
                res_hidden = res_samples.pop()
                x = torch.cat([x, res_hidden], dim=1)
                x = resnet(x, temb)
                x = depth_blocks[depth_i](x, depth)
                depth_i += 1
                if up_block.attentions is not None:
                    x = up_block.attentions[i](x)
                    x = depth_blocks[depth_i](x, depth)
                    depth_i += 1
            if len(up_block.upsamplers) > 0:
                x = up_block.upsamplers[0](x)

        x = self.conv_out(torch.nn.functional.silu(self.conv_norm_out(x)))
        return UNet2DOutput(sample=x)
