import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        conditioner_3d_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        exclude_wrap_input: Optional[list] = None,
        init_model: Optional[bool] = False,
    ):
        super().__init__()
        self.init_param = {
            "network_config": network_config,
            "network_wrapper": network_wrapper,
            "compile_model": compile_model,
            "denoiser_config": denoiser_config,
            "sampler_config": sampler_config,
            "conditioner_config": conditioner_config,
            "conditioner_3d_config": conditioner_3d_config,
            "first_stage_config": first_stage_config,
            "loss_fn_config": loss_fn_config,
            "ema_decay_rate": ema_decay_rate,
        }

        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.scheduler_config = scheduler_config
        self.use_ema = use_ema
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.exclude_wrap_input = exclude_wrap_input
        self.bsz = None

        if ckpt_path.endswith("ckpt"):
            self.sd = torch.load(ckpt_path, map_location="cpu")
        elif ckpt_path.endswith("safetensors"):
            self.sd = load_safetensors(ckpt_path, device="cpu")
        else:
            raise NotImplementedError
        if init_model:
            self.configure_model()

    def configure_model(self):
        model = instantiate_from_config(self.init_param["network_config"])
        self.model = get_obj_from_str(default(self.init_param["network_wrapper"], OPENAIUNETWRAPPER))(
            model, compile_model=self.init_param["compile_model"]
        )
        self.denoiser = instantiate_from_config(self.init_param["denoiser_config"])
        self.sampler = (
            instantiate_from_config(self.init_param["sampler_config"])
            if self.init_param["sampler_config"] is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(self.init_param["conditioner_config"], UNCONDITIONAL_CONFIG)
        )
        self.conditioner_3d = instantiate_from_config(
            default(self.init_param["conditioner_3d_config"], UNCONDITIONAL_CONFIG)
        )
        self._init_first_stage(self.init_param["first_stage_config"])
        self.loss_fn = (
            instantiate_from_config(self.init_param["loss_fn_config"])
            if self.init_param["loss_fn_config"] is not None
            else None
        )
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=self.init_param["ema_decay_rate"])
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if self.sd is not None:
            self.init_from_ckpt()

        self.freeze_parameters_name = []
        freeze_keys = ['time_embed.', 'label_emb.']
        for k, v in self.model.named_parameters():
            if any([key in k for key in freeze_keys]) or '_3d' in k:
                v.requires_grad = False
                self.freeze_parameters_name.append(k)

        self.conditioner_3d.embedders[0].model = self.conditioner.embedders[0].model
        self.conditioner_3d.embedders[1].encoder = self.conditioner.embedders[3].encoder

    def init_from_ckpt(self) -> None:
        missing, unexpected = self.load_state_dict(self.sd, strict=False)
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        del self.sd

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # process batch channel
        self.bsz = batch[self.input_key].shape[0]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                v = rearrange(v, "b f ... -> (b f) ...")
            if k in self.exclude_wrap_input:
                v = v.to(torch.int)
            batch[k] = v
        if 'num_video_frames' in batch:
            batch['num_video_frames'] = batch['num_video_frames'].item()
        if 'num_video_frames_3d' in batch:
            batch['num_video_frames_3d'] = batch['num_video_frames_3d'].item()
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, x_3d, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, self.conditioner_3d, x, x_3d, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        x_3d = self.encode_first_stage(batch['jpg_3d'])
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, x_3d, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        # params = list(self.model.parameters())
        params = [v for k, v in self.model.named_parameters() if k not in self.freeze_parameters_name]
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        cond_3d: Dict,
        uc: Union[Dict, None] = None,
        uc_3d: Union[Dict, None] = None,
        shape: Union[None, Tuple, List] = None,
        shape_3d: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(shape).to(self.device)
        randn_3d = torch.randn(shape_3d).to(self.device)

        denoiser = lambda input, sigma, c, input_3d, sigma_3d, c_3d: self.denoiser(
            self.model, input, input_3d, sigma, sigma_3d, c, c_3d, **kwargs
        )
        samples, samples_3d = self.sampler(denoiser, randn, randn_3d, cond, cond_3d, uc=uc, uc_3d=uc_3d)
        return samples, samples_3d

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        sampling_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders] +\
                                 [e.input_key for e in self.conditioner_3d.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log, sampling_kwargs = dict(), dict()

        x = batch["jpg"]
        batch['cond_frames'] = batch['cond_frames_eval']
        batch['cond_frames_3d'] = batch['cond_frames_3d_eval']

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )
        c_3d, uc_3d = self.conditioner_3d.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        for key in batch:
            if key in sampling_keys:
                sampling_kwargs[key] = batch[key]
        sampling_kwargs['image_only_indicator'] = sampling_kwargs['image_only_indicator'].repeat(2, 1)
        sampling_kwargs['image_only_indicator_3d'] = sampling_kwargs['image_only_indicator_3d'].repeat(2, 1)
        sampling_kwargs['mask_fuse'] = torch.concat([sampling_kwargs['mask_fuse']] * 2)
        sampling_kwargs['obj_pos'] = sampling_kwargs['obj_pos'] * 2
        sampling_kwargs['obj_ratio'] = torch.concat([sampling_kwargs['obj_ratio']] * 2)
        sampling_kwargs['valid_mask'] = torch.concat([sampling_kwargs['valid_mask']] * 2)
        sampling_kwargs['indices_3d'] = torch.concat(
            [
                sampling_kwargs['indices_3d'],
                sampling_kwargs['indices_3d'] + batch['num_video_frames_3d'],
             ]
        )

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        log["cond_frames"] = batch['cond_frames'].to(self.device)[:N]
        # log["obj_im"] = batch['cond_frames_without_noise'][1].to(self.device)[:N]
        # log["mask"] = batch['mask'].repeat(1, 3, 1, 1).to(self.device)[:N] * 2. - 1.

        depth_im = batch['depth'].to(self.device)[:N]
        cm = plt.get_cmap('plasma')
        depth_im_vis = (cm((depth_im[:, 0].cpu().numpy() + 1.) / 2.)[..., :3] * 2. - 1.).astype(np.float32)
        log["depth_cube"] = torch.tensor(depth_im_vis).permute(0, 3, 1, 2)

        en_and_decode_n_samples_a_time = self.en_and_decode_n_samples_a_time
        self.en_and_decode_n_samples_a_time = 6
        z = self.encode_first_stage(x)
        z_3d = self.encode_first_stage(batch['jpg_3d'])
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
                if c[k].shape[0] != batch['num_video_frames']:
                    v = repeat(c[k], "b ... -> b t ...", t=batch['num_video_frames'])
                    v = rearrange(v, "b t ... -> (b t) ...", t=batch['num_video_frames'])
                    c[k] = v
                if uc[k].shape[0] != batch['num_video_frames']:
                    v = repeat(uc[k], "b ... -> b t ...", t=batch['num_video_frames'])
                    v = rearrange(v, "b t ... -> (b t) ...", t=batch['num_video_frames'])
                    uc[k] = v
        for k in c_3d:
            if isinstance(c_3d[k], torch.Tensor):
                if c_3d[k].shape[0] != batch['num_video_frames_3d']:
                    v = repeat(c_3d[k], "b ... -> b t ...", t=batch['num_video_frames_3d'])
                    v = rearrange(v, "b t ... -> (b t) ...", t=batch['num_video_frames_3d'])
                    c_3d[k] = v
                if uc_3d[k].shape[0] != batch['num_video_frames_3d']:
                    v = repeat(uc_3d[k], "b ... -> b t ...", t=batch['num_video_frames_3d'])
                    v = rearrange(v, "b t ... -> (b t) ...", t=batch['num_video_frames_3d'])
                    uc_3d[k] = v

        if sample:
            with self.ema_scope("Plotting"):
                samples, samples_3d = self.sample(
                    c, c_3d, shape=z.shape, shape_3d=z_3d.shape, uc=uc, uc_3d=uc_3d, **sampling_kwargs,
                )

            self.en_and_decode_n_samples_a_time = 2  # 3090
            samples = self.decode_first_stage(samples)
            samples_3d = self.decode_first_stage(samples_3d)
            self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

            log["samples"] = samples
            log["samples_3d"] = samples_3d

        return log
