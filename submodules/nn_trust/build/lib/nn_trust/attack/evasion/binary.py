from contextlib import suppress
from typing import Optional

import torch
from pydantic import Field
from tqdm.auto import tqdm

from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.utils._utils import to_device
from nn_trust.attack.utils.logger import Logger
from nn_trust.core import AttackType, Knowledge, Task


class BinaryAttackConfig(EvasionAttackConfig):
    epsilon: float = Field(
        default=1e-6,
        description="Minimum update tolerance before stopping.",
        ge=0.0,
        title="Update's tolerance"
    )
    eta: float = Field(
        default=1e-6,
        description="Percentage increase in adverse disturbance.",
        ge=0.0,
        lt=1.0,
        title="Step size increment"
    )
    toll: float = Field(
        default=1e-9,
        description="Tolerance for vector norms.",
        ge=0.0,
        title="Tolerance"
    )
    early_stop: bool = Field(
        default=True,
        description="Whether the procedure stops if all classes satisfy the evasion conditions."
    )


@EvasionAttackFactory.register(
    name="Binary",
    description="A white-box adversarial attack that finds the optimal perturbation for each class and returns the most realistic modification.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class BinaryAttack(EvasionAttack):
    CONFIG_T = BinaryAttackConfig

    def _log(
            self,
            i: int,
            x: torch.Tensor,
            x_adv: torch.Tensor,
            y: torch.Tensor
    ) -> None:
        """
        Logs the most_confident class, confidence in the original prediction,
        adversary perturbation, perturbation and model_adv_classification for each
        element of a batch, separately.

        Args:
            x: original input
            x_adv: adversarial input
            y: original input
            i: iteration
        """
        err = (x_adv - x).norm(p=2, dim=list(range(1, x.dim())))
        probs = self.config.model(x_adv).softmax(-1)
        adv_predictions = self.config.model(x_adv).argmax(-1)
        self.logger.set_step(state="generate", step=i)

        for j in range(x_adv.shape[0]):
            self.most_confident[j, i] = max(probs[j].max(), self.most_confident[j, i])
            self.logger.log(
                tag=f"most_confident{j}" if x_adv.shape[0] != 1 else "most_confident",
                data=self.most_confident[j, i].unsqueeze(0),
                metadata="tensor",
                state="generate",
            )
            if self.best_error[j, i] > err[j]:
                # In this case the error associated with the j-th element of
                # a batch at the i-th iteration is better than the old one.
                self.logger.log(
                    tag=f"confidence{j}" if x_adv.shape[0] != 1 else "confidence",
                    data=probs[j, y[j]].unsqueeze(0),
                    metadata="tensor",
                    state="generate",
                )

                self.logger.log(
                    state="generate",
                    data=x_adv[j].unsqueeze(0),
                    tag=f"res{j}" if x_adv.shape[0] != 1 else "res",
                    metadata="image",
                )
                self.logger.log(
                    state="generate",
                    data=(x_adv[j] - x[j]).unsqueeze(0),
                    tag=f"perturbation{j}" if x_adv.shape[0] != 1 else "perturbation",
                    metadata="image",
                )
                self.logger.log(
                    state="generate",
                    data=adv_predictions[j],
                    tag=f"model_adv_classification{j}" if x_adv.shape[0] != 1 else "model_adv_classification",
                    metadata="scalar",
                )

                # update the error
                self.best_error[j, i] = err[j]

    def step(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        """Generate the minimal perturbation for passing from a class c_1 to a class c_2. It is based on the DeepFool algorithm

        Args:
            i int: Iteration.
            x (torch.Tensor): a tensor of shape (B, C, W, H).
            y_pred (torch.Tensor): a tensor containing the original predictions.
            y (torch.Tensor): If the attack is targeted it is the target label else is the original label.

        Return:
            a perturbation tensor with shape (B, C, W, H), such that x + perturbation is mis-classified by the passed model.
        """
        ### Initialization

        dim = list(range(1, x.dim()))
        # It represents the fact that the input could be batched
        rows = torch.arange(start=0, end=x.shape[0]).to(self.config.device)
        x_adv = x.clone()

        for i in range(self.config.max_iters):
            x_adv.requires_grad_()
            out = self.config.model.forward(x_adv)
            # B x 1
            f = out[rows, y_pred] - out[rows, y]
            # creation of the jacobian BxCxHxW
            derivative = torch.autograd.grad(f.sum(), x_adv)[0]
            with torch.no_grad():
                normalized_derivative = derivative / derivative.norm(2, dim=dim, keepdim=True).clamp_min(
                    self.config.toll
                ).pow(2)
                r = -f.view(-1, 1, 1, 1) * normalized_derivative
                # update
                x_adv = x_adv + r
                if self.config.early_stop and (r.norm(2, dim=dim).max() < self.config.epsilon):
                    return (1 + self.config.eta) * x_adv - x, True
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            self._log(x=x, x_adv=x_adv, y=y, i=i)
        return (1 + self.config.eta) * (x_adv - x), False

    def generate(
            self, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None, **kwargs
    ) -> torch.Tensor:
        """Generate an adversarial sample x* starting from a sample x and a target or the original label.

        Args:
            x: A tensor representing the sample to attack
            y: If targeted, the label to reach. If un-targeted, the class to miss-classification. y is one-hot encoded.
            ext_results: A dictionary where further results information are saved.

        Return:
            The resulting tensor of the perturbation p that x* = x + p
        """
        # Checking and setting all the parameters.
        # Creates the logger
        self.logger = kwargs.get("logger", Logger())

        # automatically tracks the variables
        with suppress(NotImplementedError):
            self.track_variables()

        # Map to the correct device
        to_device(self.config, self._config.device)
        x = x.to(self._config.device)
        self.config.model.to(self.config.device)
        self.res = x

        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Get number of classes
        with torch.no_grad():
            self.num_classes = self._config.model(x).shape[-1]

        self.logger.log(tag="original_images", data=x.detach().cpu(), state="generate", metadata="images")

        if y is not None:
            self.logger.log(
                tag="original_classification", data=y.argmax(dim=-1).detach().cpu(), state="generate", metadata="tensor"
            )
            y = y.to(self._config.device)
        else:
            if self._config.targeted:
                raise ValueError("In case of a targeted attack a target must be assigned, however 'y' is None.")

        dim = list(range(1, x.dim()))
        y_pred = self.config.model(x.to(self.config.device)).argmax(-1)

        if self.config.targeted:
            self.res = x + self.step(x=x, y=y.argmax(-1), y_pred=y_pred)
        else:
            # dimensionalità B x max_iter
            self.best_error = (
                self.best_error
                if hasattr(self, "best_error")
                else torch.ones(size=(x.shape[0], self.config.max_iters), device=self.config.device) * float("inf")
            )
            self.most_confident = torch.zeros(size=(x.shape[0], self.config.max_iters), device=self.config.device)
            self.confidence = torch.zeros(size=(x.shape[0], self.config.max_iters), device=self.config.device)

            best_err = float("inf") * torch.ones(x.shape[0], device=self.config.device)
            loop = tqdm(range(self.num_classes), disable=not self._config.verbose)
            for i in loop:
                target_class = torch.tensor([i] * x.shape[0], device=self.config.device)
                pert_adv, success = self.step(x=x, y_pred=y_pred, y=target_class)
                x_adv = x + pert_adv
                adv_predicted_class = self.config.model(x_adv).argmax(-1)
                err_adv = torch.norm(pert_adv, p=2, dim=dim)
                # I update the perturbation only if two conditions are satisfy at the same time:
                # 1) a smaller perturbation has been found
                # 2) the targeted class is different from predicted one
                # 3) the perturbation has need to create a change
                positions = torch.logical_and(best_err >= err_adv, target_class != y_pred)
                positions = torch.logical_and(positions, adv_predicted_class != y_pred)
                if torch.any(positions):
                    self.res[positions] = (x_adv)[positions]
                    best_err[positions] = err_adv[positions]

                if success:
                    break

            self.reset()

        return self.res

    def reset(self):
        for atr in ["best_error", "most_confident", "confidence"]:
            if hasattr(self, atr):
                delattr(self, atr)

    def __repr__(self):
        return "Binary"
