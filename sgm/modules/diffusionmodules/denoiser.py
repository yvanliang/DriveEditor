from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        input_3d: torch.Tensor,
        sigma: torch.Tensor,
        sigma_3d: torch.Tensor,
        cond: Dict,
        cond_3d: Dict,
        **additional_model_inputs,
    ) -> (torch.Tensor, torch.Tensor):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        sigma_3d = self.possibly_quantize_sigma(sigma_3d)
        sigma_3d_shape = sigma_3d.shape
        sigma_3d = append_dims(sigma_3d, input_3d.ndim)
        c_skip_3d, c_out_3d, c_in_3d, c_noise_3d = self.scaling(sigma_3d)
        c_noise_3d = self.possibly_quantize_c_noise(c_noise_3d.reshape(sigma_3d_shape))
        model_output, model_output_3d = network(input * c_in, input_3d * c_in_3d, c_noise, c_noise_3d, cond, cond_3d,
                                                **additional_model_inputs)
        return (
            model_output * c_out + input * c_skip,
            model_output_3d * c_out_3d + input_3d * c_skip_3d,
        )


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
