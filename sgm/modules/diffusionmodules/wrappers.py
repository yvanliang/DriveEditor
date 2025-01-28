import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, x_3d: torch.Tensor, t: torch.Tensor, t_3d: torch.Tensor, c: dict, c_3d: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        x_3d = torch.cat((x_3d, c_3d.get("concat", torch.Tensor([]).type_as(x_3d))), dim=1)
        return self.diffusion_model(
            x,
            x_3d,
            timesteps=t,
            timesteps_3d=t_3d,
            context=c.get("crossattn", None),
            context_3d=c_3d.get("crossattn", None),
            y=c.get("vector", None),
            y_3d=c_3d.get("vector", None),
            res_fuse=c.get("depth", None),
            **kwargs,
        )
