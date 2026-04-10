from typing import Optional, Type, Literal

import torch
import torch.optim as optim
from pydantic import Field, field_validator

from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.adv_generator import Perturbation, PerturbationConfig
from nn_trust.attack.normalization import LpNormalization
from nn_trust.core import Task
from nn_trust.loss.loss_composer import LossComposer


class _NumericalMethodsAttackConfig(EvasionAttackConfig):
    ##### attack
    image_smoothing: bool = Field(
        default=True,
        description="If the perturbation should be created as a patch of the original domain of the images."
    )

    patch: Optional[tuple[int, int] | tuple[int, int, int, int]] = Field(
        default=None,
        description="Get the dimension of the patch. If it is `None` then there would be no patch."
    )

    epsilon: float = Field(
        default=30.0,
        description="Attack strength.",
        ge=0.0,
        lt=float('inf')
    )

    universal: bool = Field(
        default=False,
        description="It tells weather the adversarial perturbation is universal or not."
    )

    ########## loss ##########
    loss: LossComposer = Field(default_factory=lambda: LossComposer(losses={"misclassification": {}}),
                               description="The objective loss that has to be reduce.")

    optimizer: Type[optim.Optimizer] = Field(default=optim.SGD, description="Optimizer to use.")

    bound_parameters: Optional[dict[str, float]] = Field(
        default={"c_max": 5.0, "c_min": -5.0},
        description="Parameters for the input domain."
    )
    ##### optimizer
    optimizer_params: dict[str, float | int] = Field(
        default_factory=lambda: dict(lr=0.1, weight_decay=5e-4), description="Parameters of the optimizer."
    )

    scheduler: Type[optim.lr_scheduler.LRScheduler] = Field(
        default_factory=lambda: None,
        description="Scheduler for the learning rate."
    )

    scheduler_params: dict[str, float | int] = Field(
        default_factory=lambda: dict(),
        description="Parameters of the learning rate scheduler."
    )

    gradient_normalizer: Optional[LpNormalization] = Field(
        default_factory=lambda: None,
        description="Gradient normalization at each optimization step."
    )

    optim_lr: float = Field(default=0.1, ge=1e-5, le=1.00, title="Learning Rate", description="SGD Optimizer learning rate.")

    optim_momentum: float = Field(default=0.0, ge=0.0, le=1.00, title="SGD Momentum", description="SGD Optimizer momentum.")

    optim_nesterov: bool = Field(default=False, title="Nesterov Momentum", description="SGD Nesterov momentum.")

    normalization_strategy: Literal["perturbation", "result", "none"] = Field(default="perturbation", description="Normalization strategy. Decide if project on epsilon-radius sphere only perturbation, adversarial result image or do nothing.")

    ########### VALIDATORS ###########
    @field_validator("patch", mode="before")
    def valid_patch(cls, v):
        if v is not None:
            if len(v) != 2 and len(v) != 4:
                raise ValueError(f"The dimension of the patch must be 2 or 4. Given {len(v)}")
            else:
                if (not isinstance(v[0], int)) or (not isinstance(v[1], int)):
                    raise ValueError("The dimensions of the patch must be integers.")
        return v

    @field_validator("optimizer", mode="before")
    def valid_optimizer(cls, v):
        if not issubclass(v, optim.Optimizer):
            raise ValueError("optimizer must be a subclass of optim.Optimizer.")
        return v

    @field_validator("scheduler", mode="before")
    def valid_scheduler(cls, v):
        if v is not None and not issubclass(v, optim.lr_scheduler.LRScheduler):
            raise ValueError("scheduler must be a subclass of optim.lr_scheduler.LRScheduler or None.")
        return v

    @field_validator("gradient_normalizer", mode="before")
    def valid_gradient_normalizer(cls, v):
        if v is not None and not isinstance(v, LpNormalization):
            raise ValueError("gradient_normalizer must be an instance of Normalization or None.")
        return v


class _NumericalMethodsAttack(EvasionAttack):
    CONFIG_T = _NumericalMethodsAttackConfig

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")
        self.add_variable_to_track("loss", "scalar")
        if self._config.task == Task.Detection:
            self.add_variable_to_track("preds", "tensor")
            self.add_variable_to_track("bboxes", "tensor")
        elif self._config.task == Task.Classification:
            self.add_variable_to_track("preds", "tensor")

    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            patch_mask: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs,
    ) -> tuple[torch.Tensor, bool]:
        ######## Adversarial generator Initialization ########
        input_shape = x[0].size() if self._config.universal else x.size()
        if not hasattr(self, "adv_generator"):
            self.adv_generator = Perturbation(
                PerturbationConfig(
                    target_size=list(x.size()),
                    patch_size=self._config.patch,
                    mask=patch_mask,
                    device=self._config.device,
                )
            )
        # Optimizer
        if not hasattr(self, "optimizer"):
            self.optimizer = optim.SGD(
                params=self.adv_generator.parameters(),
                lr=self._config.optim_lr,
                momentum=self._config.optim_momentum,
                nesterov=self._config.optim_nesterov
            )

            #self.optimizer = self._config.optimizer(self.adv_generator.parameters())
            ### setting the parameters that was chosen in the configuration file
            # for key, _ in self.optimizer.param_groups[0].items():
            #     if key in self._config.optimizer_params.keys():
            #        self.optimizer.param_groups[0][key] = self._config.optimizer_params[key]

        # Scheduler
        if not hasattr(self, "scheduler"):
            if self._config.scheduler is not None:
                self.scheduler = self._config.scheduler(optimizer=self.optimizer, **self._config.scheduler_params)
        # Perturbation Normalizer
        if not hasattr(self, "perturbation_normalizer"):
            self.perturbation_normalizer = LpNormalization(p=self._config.p, radius=self._config.epsilon)

        def closure():
            self.optimizer.zero_grad()
            if self.config.normalization_strategy == "perturbation":
                self.adv_generator.parameter_op_(self.perturbation_normalizer)
                x_adv = self.adv_generator(x)
            elif self.config.normalization_strategy == "result":
                x_adv_tmp = self.adv_generator(x)
                x_adv = self._config.epsilon * x_adv_tmp / x_adv_tmp.norm(p=self._config.p, dim=tuple(range(1, x_adv_tmp.dim())), keepdim=True)
                #x_adv = self.perturbation_normalizer(x_adv_tmp)
            else:
                x_adv = self.adv_generator(x)
            if self._config.task == Task.Detection:
                self.bboxes, self.preds = self._config.model(x_adv)
            elif self._config.task == Task.Classification:
                self.preds = self._config.model(x_adv)
            input_loss = {"x_adv": x_adv, "x": x, "out_adv": self.preds, "target": y}

            if len(kwargs) > 0:
                input_loss.update(kwargs)

            out = self._config.loss(**input_loss)
            out.backward()
            # Additional gradient normalization
            if self._config.gradient_normalizer is not None:
                with torch.no_grad():
                    for param in self.adv_generator.parameters():
                        param.grad.data = self._config.gradient_normalizer(param.grad).data
            return out

        # actual optimization procedure
        self.loss = self.optimizer.step(closure)
        if isinstance(self.loss, torch.Tensor):
            self.loss = self.loss.item()

        if self._config.scheduler is not None:
            self.scheduler.step()
        # tqdm and verbose logging
        loop = kwargs.get("loop")
        log_on_tqdm = self._config.verbose and loop is not None and hasattr(loop, "set_postfix")
        if log_on_tqdm:
            loop.set_postfix({"loss": self.loss})

        # generate the adversary sample
        self.perturbation = self.adv_generator.render_perturbation()
        x_adv = self.adv_generator(x).clone().detach()
        return x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "adv_generator",
            "optimizer",
            "scheduler",
            "perturbation_normalizer",
            "loss",
            "preds",
            "bboxes"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
