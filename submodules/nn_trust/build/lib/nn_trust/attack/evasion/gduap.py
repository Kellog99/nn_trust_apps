from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import Field, field_validator
from tqdm.auto import tqdm

from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.utils._utils import _compare_misclassification, to_device
from nn_trust.core import AttackType, Knowledge, Task


class GDUAPAttackConfig(EvasionAttackConfig):
    theta: float = Field(
        default=0.2,
        description="Saturation rate: represents the percentage of pixels reaching the threshold parameter 'epsilon'",
        ge=0.0,
        le=1.0,
    )
    saturation_rescaling: float = Field(
        default=2.0,
        description="How much do we need to divide delta, when the saturation reached the threshold",
        gt=0.0
    )
    H: int = Field(
        default=5,
        description="History of previous fooling ratios to store, used to determine the best model by fooling ratio. It works only when a proximity dataset is passed.",
        gt=0
    )
    optimizer_type: type[optim.Optimizer] = Field(default=optim.SGD, description="Type of the optimizer to use.")
    optimizer_args: dict[str, Any] = Field(
        default_factory=lambda: {"lr": 0.1},
        description="Parameter of the optimizer, the learning rate 'lr' is the minimum parameter required.",
    )
    layer_names: list[str] = Field(
        default_factory=lambda: [],
        description="Target neural network layer names to consider for the attack optimization procedure.",
    )

    @field_validator("optimizer_args")
    def valid_optimizer_args(cls, v):
        if not isinstance(v, dict) or "lr" not in v:
            raise ValueError('optimizer_args must be a dictionary containing at least the key "lr".')
        return v


@EvasionAttackFactory.register(
    name="Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations",
    description="A white-box universal adversarial attack that tries to 'overstimulate' the model's convolutional layers.",
    task={Task.Classification, Task.Segmentation},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class GDUAPAttack(EvasionAttack):
    r"""Implements the Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations
    [1]_. The method does not rely on data used for training or testing, but relies on the knowledge of
    model's architecture and parameters. They provide a few interesting improvements upon previous SOTA
    universal attack methods:
    - Data-Free Approach: Unlike existing methods, this technique doesn't rely on specific traini    ng
     data to craft UAPs.
    - Task Agnostic: The method is applicable to various vision tasks, including object recogniti    on,
        semantic segmentation, and depth estimation.
    - Black-Box Attack Effectiveness: The proposed method effectively fools black-box models,
        where the attacker has limited knowledge of the model's architecture and training data.
    - Prior Knowledge Enhancement: By leveraging simple priors about data distribution,
        the method further improves the effectiveness of UAPs.

    The idea is to optimize the function
    .. math::
     - \sum_{i=1}^K \log\left( \left\| l_i(x + \delta) \right\|_2\right)

    where the :math:`l_i` are the activation of the :math:`i`-th convolutional layer.

    For a more in-depth description see [1]_.

    .. [1] Konda Reddy Mopuri and Aditya Ganeshan and R. Venkatesh Babu,
        Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations,
        https://doi.org/10.48550/arXiv.1801.08092.
    """

    CONFIG_T = GDUAPAttackConfig

    def _is_conv_layer(self, name: str, layer: nn.Module) -> bool:
        # If config.layer_names is empty, check whether the class name corresponds to any of
        # the defined class in the torch.nn.modules.conv, which corresponds to validating that
        # the layer is a convolution layer.
        if not self._config.layer_names:
            return layer.__class__.__name__ in nn.modules.conv.__all__
        # If the name corresponds to the one in the config's layer_names, then validates it.
        else:
            return name in self._config.layer_names

    @staticmethod
    def _saturation_rate(delta: torch.Tensor, threshold: float) -> float:
        return torch.mean((delta.abs().view(-1) >= threshold) * 1.0).item()

    def _hooks_to_convs(self):
        """
        Adds a hook to each convolution layer with its relative buffer
        named log_loss. The hook stores to the buffer log_loss the value
        of log(|Relu(output)|_2).
        """
        cum_log_loss = [torch.zeros(1, device=self._config.device)]

        def module_hook(_model: nn.Module, _input: torch.Tensor, output: torch.Tensor):
            # print(f"Before computing log-loss for: {cum_log_loss[0]}")
            cum_log_loss[0] -= torch.log(torch.norm(F.relu(output), p=2.0))
            # print(f"After computing log-loss for: {cum_log_loss[0]}")

        self.hooks = []
        # Adds a buffer and a forward hook to each convolution layer
        for name, layer in self._config.model.named_modules():
            if self._is_conv_layer(name, layer):
                if self._config.verbose:
                    print(f"Adding the GD-UAP-attack hook to named layer: {name}, {layer}")

                hook_handle = layer.register_forward_hook(module_hook)
                self.hooks.append(hook_handle)
        return cum_log_loss

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "image")

    def fit(
            self,
            proxy_data: torch.utils.data.DataLoader,
            **kwargs: Any,
    ) -> None:
        if self._config.verbose:
            torch.autograd.set_detect_anomaly(True)

        to_device(self._config, self._config.device)
        self.cll = self._hooks_to_convs()
        # Store best value of delta
        min_loss = float("inf")
        epoch_loop = range(self._config.max_iters)
        if self._config.verbose:
            epoch_loop = tqdm(epoch_loop, initial=0, desc="Generating GDUAP Attack...", position=0)

        for iters in epoch_loop:
            avg_loss = 0.0
            n_batches = 0
            for x, _ in proxy_data:
                if not hasattr(self, "perturbation"):
                    self.perturbation = (
                            torch.randn((1, *x.shape[1:]), device=self._config.device) * self._config.epsilon
                    )
                    self.perturbation.requires_grad_(True)

                if not hasattr(self, "param_optim"):
                    param_optim = self._config.optimizer_type([self.perturbation], **self._config.optimizer_args)

                # Initialize to 0 gradient
                param_optim.zero_grad()
                # Forward pass
                x = x.to(self._config.device)
                self._config.model((x + self.perturbation).clamp(-1, 1))
                # optimization and avg_loss computation
                avg_loss += self.cll[0].item()
                n_batches += 1
                self.cll[0].backward(retain_graph=True)
                param_optim.step()

            avg_loss /= n_batches

            if self._config.verbose:
                print(f"At iteration: {iters}, the loss is: {avg_loss}.")

            if avg_loss < min_loss:
                self.best_modifier = self.perturbation.clone().detach()
                min_loss = avg_loss

            # Rescaling procedure of the parameters
            with torch.no_grad():
                saturation_rate = self._saturation_rate(self.perturbation, self._config.epsilon)
                if saturation_rate > self._config.theta:
                    if self._config.verbose:
                        print(f"Halving modifier parameters; The saturation rate is: {saturation_rate}")
                    self.perturbation /= self._config.saturation_rescaling
                # Reset the cumulative log loss.
                # NOTE: we can't use zero_, because it updates the
                # steps in the backward procedure, which is pointless in our
                # case.
                self.cll[0][:] = 0.0

        # Remove the model's handles
        for handle in self.hooks:
            handle.remove()

        if hasattr(self, "param_optim"):
            del self.param_optim

    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        r"""
        Implements a data-free universal attack as described in [1].

        The general idea is to optimize the function
        .. math::
         - \sum_{i=1}^K \log\left( \left\| l_i(x + \delta) \right\|_2\right)

        where the l_i are the activations of a convolutional layer.
        This allows for a possible data-free white-box attack on a neural-network
        by trying to fool the feature detection of each convolution layer.

        For a more in-depth description of the verschiedene details, see [1].

        :param x: it is a tensor of size (B, C, W, H) with B, C, W, H being respectively
            the number of elements in a batch, channels, width and height of an input image.
            If `with_range=True` and `B > 1` the tensor `x` should correspond to a gaussian sample from
            a distribution with mean and variance analogue to the targeted distribution dataset.
            If `with_range=False` and `B > 1` the tensor `x` should correspond to a sample form
            the targeted distribution.
        :param y: it is either None or a tensor of shape (B, NC) with B being the number of elements in a batch
            and NC the number of classes. It is required if and only if `with_range=False` and `B > 1`.

        [1] Konda Reddy Mopuri and Aditya Ganeshan and R. Venkatesh Babu,
            Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations,
            https://doi.org/10.48550/arXiv.1801.08092.

        Example::

        Example usage for image classification.


        >>> # Generate the attack.
        >>> cnf = EAF.get_config('gduap',
        >>>                      model=model,
        >>>                      targeted=False,
        >>>                      verbose=False,
        >>>                      optimizer_type=optim.Adam,
        >>>                      optimizer_args=dict(lr=0.1),
        >>>                      epsilon=0.03,
        >>>                      max_iters=400,
        >>>                      task=Task.Classification)
        >>> atk = EAF.create_attack(config=cnf)
        >>> # Generate a universal perturbation by giving the correct shape of the image, computes the
        >>> # prediction in case of a given image.
        >>> perturb = atk.generate(torch.zeros_like(img))
        >>> perturb_pred = torch.argmax(model(perturb + img))

        In a similar manner can be applied to ``Task.Segmentation``.
        """
        # If we have a true dataset onto validate the vector, compute the fooling rates.
        if not hasattr(self, "fooling_ratios"):
            self.fooling_ratios = torch.zeros(self._config.H, device=self._config.device)

        if not hasattr(self, "cll"):
            self.cll = self._hooks_to_convs()

        if not hasattr(self, "perturbation"):
            self.perturbation = torch.randn((1, *x.shape[1:]), device=self._config.device) * self._config.epsilon
            self.perturbation.requires_grad_(True)
        if not hasattr(self, "param_optim"):
            self.param_optim = self._config.optimizer_type([self.perturbation], **self._config.optimizer_args)
        # Store best perturbation value
        if not hasattr(self, "best_perturbation"):
            self.best_perturbation = self.perturbation.clone().detach()

        self.param_optim.zero_grad()

        # Computes the perturbation wrt a known dataset
        self._config.model(x + self.perturbation)
        self.cll[0].backward(retain_graph=True)
        # optimization + halving when the parameters are too saturated
        self.param_optim.step()
        with torch.no_grad():
            if self._config.verbose:
                print(f"Saturation: {self._saturation_rate(self.perturbation, self._config.epsilon)}")
            if self._saturation_rate(self.perturbation, self._config.epsilon) > self._config.theta:
                if self._config.verbose:
                    print("Halving perturbation")
                self.perturbation.div_(self._config.saturation_rescaling)
            # Reset the cumulative log loss.
            # NOTE: we can't use zero_, because it updates the
            # steps in the backward procedure, which is pointless in our
            # case.
            self.cll[0][:] = 0.0

        # If data of some kind is passed, we compute the validation fooling rate
        # to know which iteration of perturbation should be the best.
        new_fooling_ratio = torch.mean(
            _compare_misclassification(
                y_label=torch.argmax(y, dim=1),
                y_pred=torch.argmax(self._config.model(x + self.perturbation), dim=1)
            )
            * 1.0
        )
        if self._config.verbose:
            print(f"Stored fooling ratios: {self.fooling_ratios}")
            print(f"The newly evaluated fooling ratio: {new_fooling_ratio}")

        # If the attack degrades significantly, stops
        if all(new_fooling_ratio < self.fooling_ratios):
            return self.perturbation + x, True
        else:
            self.fooling_ratios = self.fooling_ratios.roll(shifts=1)
            self.fooling_ratios[0] = new_fooling_ratio

        # updates the best perturbation
        if all(new_fooling_ratio >= self.fooling_ratios):
            self.best_perturbation = self.perturbation.clone().detach()
        # If we achieve more than 90% fooling ratio we are done
        if new_fooling_ratio >= 0.9:
            self.best_perturbation = torch.clip(self.best_perturbation, -self._config.epsilon, self._config.epsilon)
            return x + self.best_perturbation, True
        # In case we can't validate the fooling rate, assumes that the last iteration is the bes
        # Clip with respect to the maximum L^infinity norm, maybe add a projection function?
        self.best_perturbation = torch.clip(self.best_perturbation, -self._config.epsilon, self._config.epsilon)
        return self.best_perturbation + x, False

    def reset(self):
        super().reset()
        for atr in [
            "best_perturbation",
            "perturbation",
            "param_optim",
            "fooling_ratios",
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
        if hasattr(self, "hooks"):
            for hook in self.hooks:
                hook.remove()
            del self.hooks
