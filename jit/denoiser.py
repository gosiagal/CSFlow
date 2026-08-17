from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import torch
from torch import nn

try:
    from .model_jit import JiT_models
except ImportError:  # pragma: no cover - fallback for direct script execution
    from model_jit import JiT_models


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


@dataclass
class WeightScheduleConfig:
    """Explicit runtime configuration for CSF/JiT weighting files."""

    inference_weights_path: str = ""
    train_weights_path: str = ""
    default_weights_filename: str = "training_weights_256.pkl"


def _load_pickle_numpy_compat(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if e.name != "numpy._core":
                raise

    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.weight_config = self._build_weight_config(args)

        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.weighted = getattr(
            args, "inference_weighted", getattr(args, "weighted", False)
        )
        self.interpolated = getattr(
            args, "inference_interpolated", getattr(args, "interpolated", False)
        )
        self.interp_alpha = getattr(
            args, "inference_interp_alpha", getattr(args, "interp_alpha", 1.0)
        )
        self.gen_timeshift = getattr(
            args, "inference_timeshift", getattr(args, "gen_timeshift", 1.0)
        )
        self.inference_weights_path = self.weight_config.inference_weights_path
        self.train_weighted = getattr(args, "train_weighted", False)
        self.train_interpolated = getattr(args, "train_interpolated", False)
        self.train_interp_alpha = getattr(args, "train_interp_alpha", 1.0)
        self.train_timeshift = getattr(args, "train_timeshift", 1.0)
        self.train_weights_path = self.weight_config.train_weights_path

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    @staticmethod
    def _build_weight_config(args):
        """Normalize explicit paths and environment fallbacks into a single config."""
        explicit = getattr(args, "weight_config", None)
        if explicit is not None:
            return explicit

        config = WeightScheduleConfig()
        config.inference_weights_path = Denoiser._resolve_weight_path(
            getattr(args, "inference_weights_path", ""),
            default_filename=config.default_weights_filename,
            env_names=("JIT_INFERENCE_WEIGHTS_PATH",),
        )
        config.train_weights_path = Denoiser._resolve_weight_path(
            getattr(args, "train_weights_path", ""),
            default_filename=config.default_weights_filename,
            env_names=("JIT_TRAIN_WEIGHTS_PATH",),
        )
        return config

    @staticmethod
    def _resolve_weight_path(
        path: str | None, *, default_filename: str, env_names: tuple[str, ...]
    ) -> str:
        candidates = []
        if path and str(path).strip():
            candidates.append(str(path).strip())
        for env_name in env_names:
            env_path = os.environ.get(env_name, "")
            if env_path and env_path.strip():
                candidates.append(env_path.strip())

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
            if candidate and not os.path.exists(candidate):
                return candidate

        local_default = os.path.join(os.path.dirname(__file__), default_filename)
        if os.path.exists(local_default):
            return local_default

        return ""

    @staticmethod
    def _interp1d_torch(
        x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
    ) -> torch.Tensor:
        """Torch equivalent of np.interp for 1D monotonic xp."""
        x = x.clamp(min=xp[0], max=xp[-1])
        idx = torch.searchsorted(xp, x, right=False)
        idx = idx.clamp(min=1, max=xp.numel() - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]

        denom = (x1 - x0).clamp_min(1e-12)
        w = (x - x0) / denom
        return y0 + w * (y1 - y0)

    @staticmethod
    def _inverse_shift_respace(t: torch.Tensor, shift: float) -> torch.Tensor:
        shift = float(shift)
        if shift <= 0.0:
            raise ValueError(f"timeshift must be > 0, got {shift}")
        return (shift * t) / (1.0 + (shift - 1.0) * t)

    @staticmethod
    def _shift_respace(t: torch.Tensor, shift: float) -> torch.Tensor:
        shift = float(shift)
        if shift <= 0.0:
            raise ValueError(f"timeshift must be > 0, got {shift}")
        return t / (t + (1.0 - t) * shift)

    def _build_general_weighted_timesteps(
        self, device, interp_alpha: float = 1.0, timeshift: float = 1.0
    ):
        """Resample an arbitrary-length CSF interval schedule to the current generation steps."""
        base_weights = self._load_interval_weights(device=device, for_training=False)

        base_grid = torch.linspace(
            0.0, 1.0, base_weights.numel() + 1, device=device, dtype=torch.float32
        )
        shifted_uniform_cdf = self._inverse_shift_respace(base_grid, timeshift).clamp(
            0.0, 1.0
        )
        shifted_uniform_cdf[0] = 0.0
        shifted_uniform_cdf[-1] = 1.0

        base_cdf = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float32),
                torch.cumsum(base_weights, dim=0),
            ]
        )
        base_cdf = torch.cummax(base_cdf, dim=0)[0]
        base_cdf = base_cdf.clamp(0.0, 1.0)
        base_cdf[0] = 0.0
        base_cdf[-1] = 1.0

        mix_alpha = float(interp_alpha)
        mixed_cdf = mix_alpha * base_cdf + (1.0 - mix_alpha) * shifted_uniform_cdf
        mixed_cdf = torch.cummax(mixed_cdf, dim=0)[0]
        mixed_cdf = mixed_cdf.clamp(0.0, 1.0)
        mixed_cdf[0] = 0.0
        mixed_cdf[-1] = 1.0

        base_cdf[-1] = 1.0

        target_q = torch.linspace(
            0.0, 1.0, self.steps + 1, device=device, dtype=torch.float32
        )
        timesteps = self._interp1d_torch(target_q, mixed_cdf, base_grid).clamp(0.0, 1.0)
        return timesteps

    def _build_weighted_timesteps(
        self, device, interp_alpha: float = 1.0, timeshift: float = 1.0
    ):
        """Backward-compatible alias for the general weighted timestep builder."""
        return self._build_general_weighted_timesteps(
            device=device, interp_alpha=interp_alpha, timeshift=timeshift
        )

    @staticmethod
    def _extract_interval_weights(raw_obj):
        """Accept arbitrary-length pickles loaded from the CSF pipeline.

        The exported files may contain either:
        - a single 1D weight array, or
        - a tuple/list such as (weights, timesteps), where the second item is metadata.
        In both cases we keep the interval weights and ignore any trailing endpoint value.
        """
        if isinstance(raw_obj, (tuple, list)):
            if len(raw_obj) == 0:
                raise ValueError("Weight pickle is empty")
            weights = raw_obj[0]
        else:
            weights = raw_obj

        arr = torch.as_tensor(weights, dtype=torch.float32)
        if arr.ndim == 0:
            raise ValueError("Weight tensor must be at least 1D")
        arr = arr.reshape(-1)

        # CSF schedules may include a trailing endpoint value (for example, a padded final zero).
        # For interval masses, the final value is not a separate mass and should be dropped.
        if arr.numel() >= 2 and float(arr[-1]) <= 1e-12:
            arr = arr[:-1]

        if arr.numel() < 2:
            raise ValueError("Weights must contain at least two interval values")

        arr = arr.clamp_min(0.0)
        total = arr.sum()
        if torch.isclose(total, torch.tensor(0.0, device=arr.device, dtype=arr.dtype)):
            raise ValueError(
                "Weights sum to zero; cannot normalize a timestep schedule"
            )
        return arr / total

    def _load_interval_weights(self, device, for_training: bool = False):
        """Load a CSF/JiT interval weight schedule from an arbitrary-length pickle."""
        if for_training and self.train_weights_path:
            pkl_path = self.train_weights_path
        elif self.inference_weights_path:
            pkl_path = self.inference_weights_path
        else:
            general_weights_file = self.weight_config.default_weights_filename
            pkl_path = os.path.join(os.path.dirname(__file__), general_weights_file)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Weights file not found at {pkl_path}")

        weights_raw = _load_pickle_numpy_compat(pkl_path)
        base_weights = self._extract_interval_weights(weights_raw)
        return base_weights.to(device=device, dtype=torch.float32)

    def _build_original_train_interval_weights(self, device, num_bins: int):
        """Approximate the default JiT sigmoid-normal train distribution on a fixed grid."""
        grid = torch.linspace(
            0.0, 1.0, num_bins + 1, device=device, dtype=torch.float32
        )
        logits = torch.logit(grid[1:-1].clamp(1e-6, 1.0 - 1e-6))
        normal = torch.distributions.Normal(loc=self.P_mean, scale=self.P_std)
        cdf = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float32),
                normal.cdf(logits),
                torch.ones(1, device=device, dtype=torch.float32),
            ]
        )
        interval_weights = (cdf[1:] - cdf[:-1]).clamp_min(0.0)
        return interval_weights / interval_weights.sum().clamp_min(1e-12)

    def _build_train_interval_weights(self, device, interp_alpha: float):
        weighted_base = self._load_interval_weights(device=device, for_training=True)
        original_base = self._build_original_train_interval_weights(
            device=device, num_bins=weighted_base.numel()
        )
        mix_alpha = float(interp_alpha)
        mixed = mix_alpha * weighted_base + (1.0 - mix_alpha) * original_base
        return mixed / mixed.sum().clamp_min(1e-12)

    def _sample_from_interval_weights(
        self, interval_weights: torch.Tensor, n: int, device=None
    ):
        base_cdf = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float32),
                torch.cumsum(interval_weights, dim=0),
            ]
        )
        base_cdf[-1] = 1.0
        base_grid = torch.linspace(
            0.0, 1.0, interval_weights.numel() + 1, device=device, dtype=torch.float32
        )
        q = torch.rand(n, device=device, dtype=torch.float32)
        return self._interp1d_torch(q, base_cdf, base_grid).clamp(0.0, 1.0)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        if self.train_weighted:
            interp_alpha = self.train_interp_alpha if self.train_interpolated else 1.0
            interval_weights = self._build_train_interval_weights(
                device=device, interp_alpha=interp_alpha
            )
            t = self._sample_from_interval_weights(
                interval_weights=interval_weights, n=n, device=device
            )
            if float(self.train_timeshift) != 1.0:
                t = self._shift_respace(t, self.train_timeshift)
            return t

        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(
            bsz, 3, self.img_size, self.img_size, device=device
        )

        if self.weighted:
            interp_alpha = self.interp_alpha if self.interpolated else 1.0
            timesteps = self._build_general_weighted_timesteps(
                device=device,
                interp_alpha=interp_alpha,
                timeshift=self.gen_timeshift,
            )
            if isinstance(timesteps, torch.Tensor):
                timesteps = timesteps.to(device)
            else:
                timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
            timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        else:
            timesteps = (
                torch.linspace(0.0, 1.0, self.steps + 1, device=device)
                .view(-1, *([1] * z.ndim))
                .expand(-1, bsz, -1, -1, -1)
            )

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unsupported sampling method: {self.method}")

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
