import logging
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class PerturbationConfig(BaseModel):
    target_size: List[int] = Field(
        default=...,
        description="Shape of target object that has to be perturbed."
    )

    apply_mode: Optional[Literal["sum", "replace"] | None] = Field(
        default=None,
        description="How the perturbation is applied to the input, i.e. replaced a certain part or add on a certain input's part."
    )

    patch_size: Optional[tuple[int, int] | tuple[int, int, int, int] | None] = Field(
        default=None,
        description="Patch size height and width."
    )

    mask: Optional[torch.Tensor | None] = Field(
        default=None,
        description=""
    )

    random_init: bool | None = Field(default=None, description="")

    device: torch.device = Field(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="Device where the operations will be do.",
    )

    model_config = {"arbitrary_types_allowed": True}


class Perturbation(nn.Module):
    """
    This class has the role to represent the Adversarial Perturbation.

    The perturbation object has:
        * internal parameters to optimize
        * methods to apply to a given type of input.
    """

    def __init__(self, config: PerturbationConfig = None):
        super().__init__()
        self._config = config
        self.pert_parameters = nn.Parameter(torch.zeros(self._config.target_size))
        if self._config.random_init:
            nn.init.kaiming_uniform_(self.pert_parameters, a=0)

        if self._config.patch_size is not None and self._config.mask is None:
            self.perturbation_type = "patch"
            self._config.apply_mode = "replace" if self._config.apply_mode is None else self._config.apply_mode
            self._config.mask = torch.zeros(self._config.target_size)

            h, w = self._config.target_size[-2:]
            if len(self._config.patch_size) == 2:
                ph, pw = self._config.patch_size
                wwf = max(w - pw + 1, 1)
                pph = max(h - ph + 1, 1)
                top = torch.randint(0, pph, (1,)).item()
                left = torch.randint(0, wwf, (1,)).item()
            elif len(self._config.patch_size) == 4:
                top, left, ph, pw = self._config.patch_size
            else:
                raise ValueError(f"Invalid size for patch_size: {self._config.patch_size}")

            if ph > h - top:
                logging.warning(f"Path height too large {ph}, reduced to {h - top}")
                ph = h - top
            if pw > w - left:
                logging.warning(f"Path width too large {pw}, reduced to {w - left}")
                pw = w - left
            self._config.mask[..., top: top + ph, left: left + pw] = 1.0

        elif self._config.mask is not None and self._config.patch_size is None:
            self.perturbation_type = "mask"
            self._config.apply_mode = "replace" if self._config.apply_mode is None else self._config.apply_mode
            mask_size = list(self._config.mask.size())
            assert self._config.target_size == mask_size, (
                f"Mismatch target and mask shapes: {self._config.target_size} vs {mask_size}"
            )

        elif self._config.mask is None and self._config.patch_size is None:
            self.perturbation_type = "bg"
            self._config.apply_mode = "sum" if self._config.apply_mode is None else self._config.apply_mode
            self._config.mask = torch.ones(self._config.target_size, dtype=torch.float32)

        else:
            raise ValueError("invalid combination of parameters for perturbation object.")

        self._config.mask = self._config.mask.to(self._config.device)
        self.pert_parameters = nn.Parameter(self.pert_parameters.to(self._config.device))

    def _render_perturbation(self):
        return torch.where(self._config.mask != 0.0, self.pert_parameters, 0.0)

    @torch.no_grad()
    def render_perturbation(self):
        """
        Render perturbation for logging purposes
        """
        return self._render_perturbation().clone().detach()

    @torch.no_grad()
    def parameter_op_(self, func):
        """Function used for inplace operation on perturbation parameters, like normalization.
        Everything is applied on the active parameter portion, so mask is applied before running function.
        """
        self.pert_parameters.copy_(func(self._render_perturbation()))

    def forward(self, x):
        """Apply perturbation conditional to inferred perturbation type"""
        rendered_perturbation = self._render_perturbation()
        match self._config.apply_mode:
            case "sum":
                return x + rendered_perturbation
            case "replace":
                return torch.where(self._config.mask != 0.0, rendered_perturbation, x)
            case _:
                raise ValueError(f"Unrecognised application model value: {self._config.apply_mode}")

    def __repr__(self):
        return f"Perturbation(perturbation_type={self.perturbation_type}, apply_mode={self._config.apply_mode}, size={self._config.target_size}"
