"""
Implementation of ConvNextV1 for CIFAR-10 from PyTorch.
Based off pytorch impl: https://pytorch.org/vision/stable/_modules/torchvision/models/convnext.html
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List
from mlx.utils import tree_flatten
from collections.abc import Callable
from functools import partial
from typing import List, Optional, Callable, Tuple, Any, Union

class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=dim, bias=True),
            nn.LayerNorm(dim),
            nn.Linear(input_dims=dim, output_dims=4*dim, bias=True),
            nn.GELU(),
            nn.Linear(input_dims=4*dim, output_dims=dim, bias=True)
        )
        self.layer_scale = mx.ones((1, 1, dim)) * layer_scale
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def __call__(self, input: mx.array) -> mx.array:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result
    
    
class ConvNeXt(nn.Module):
    def __init__(
        self,
        block: Optional[Callable[..., nn.Module]] = CNBlock,
        channels_config: Tuple[int] = (96, 192, 384, 768),
        blocks_config: Tuple[int] = (3, 3, 9, 3),
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 10,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if block is None:
            block = CNBlock
            
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            
        layers: List[nn.Module] = []

        ### Stem - beginning block
        firstconv_output_channels = channels_config[0]
        # 2D convolution layer first
        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=firstconv_output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        )
        # Then layernorm
        layers.append(norm_layer(firstconv_output_channels))

        ## CN Blocks
        total_stage_blocks = sum(blocks_config)
        stage_block_id = 0
        for bc_index in range(len(blocks_config)):
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(blocks_config[bc_index]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(dim=channels_config[bc_index], layer_scale=layer_scale, stochastic_depth_prob=sd_prob, norm_layer=norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if bc_index<len(blocks_config)-1:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(channels_config[bc_index]),
                        nn.Conv2d(
                            in_channels=channels_config[bc_index], 
                            out_channels=channels_config[bc_index+1], 
                            kernel_size=2, 
                            stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        
        ## Final layer
        self.avgpool = AdaptiveAveragePool2D(1)
        lastconv_output_channels = channels_config[-1]
        self.classifier = nn.Sequential(
            Flatten(1), norm_layer(lastconv_output_channels), nn.Linear(lastconv_output_channels, num_classes)
        )
    
    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

## Utility functions

class Flatten(nn.Module):
    def __init__(self, start_axis: int = 1, end_axis: int = -1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis
    
    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, self.start_axis, self.end_axis)

class StochasticDepth(nn.Module):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    
    Args:
        input (mx.array[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    
    Returns:
        mx.array[N, ...]: The randomly zeroed tensor.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def __call__(self, input: mx.array) -> mx.array:
        if self.p < 0.0 or self.p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        if self.mode not in ["batch", "row"]:
            raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
        if not self.training or self.p == 0.0:
            return input
        
        survival_rate = 1.0 - self.p
        if self.mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = mx.random.bernoulli(survival_rate, shape=input.shape)
        if survival_rate > 0.0:
            noise = mx.divide(noise, survival_rate)
        return input * noise


    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s
        
def adaptive_average_pool2d(x: mx.array, output_size: tuple, ) -> mx.array:
    B, H, W, C = x.shape
    x = x.reshape(
        B, H // output_size[0], output_size[0], W // output_size[1], output_size[1], C
    )
    x = mx.mean(x, axis=(1, 3))
    return x


class AdaptiveAveragePool2D(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.output_size = (
            output_size
            if isinstance(output_size, tuple)
            else (output_size, output_size)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_average_pool2d(x, self.output_size)

## Model configurations go here
def ConvNeXt_Smol(**kwargs):
    return ConvNeXt(block=CNBlock, channels_config=(32, 64, 128, 256), blocks_config=(2, 2, 2, 2), layer_scale=1e-6, stochastic_depth_prob=0.2, **kwargs)
