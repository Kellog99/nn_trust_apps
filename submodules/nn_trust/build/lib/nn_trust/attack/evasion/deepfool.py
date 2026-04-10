from typing import Optional

import torch

from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.attack_factory import EvasionAttackFactory
from nn_trust.core import AttackType, Knowledge, Task


class DeepFoolAttackConfig(EvasionAttackConfig):
    pass


@EvasionAttackFactory.register(
    name="DeepFool",
    description="A white-box adversarial attack computing the optimal adversarial perturbation under the assumption of a linear classifier.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class DeepFoolAttack(EvasionAttack):
    CONFIG_T = DeepFoolAttackConfig

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")
        self.add_variable_to_track("max_change", "scalar")

    @torch.no_grad()
    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        # tqdm and verbose logging
        loop = kwargs.get("loop")
        log_on_tqdm = self._config.verbose and loop is not None and hasattr(loop, "set_postfix")

        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone()
        if not hasattr(self, "perturbation"):
            self.perturbation = self.x_adv - x
        if not hasattr(self, "n_classes"):
            self.n_classes = self._config.model(x).shape[-1]
        if not hasattr(self, "I"):
            ## Initialization
            self.I = self._config.model(self.x_adv).argmax(dim=-1)

        if not hasattr(self, "rows"):
            self.rows = torch.arange(start=0, end=x.shape[0], device=self._config.device)
        if not hasattr(self, "q"):
            if self._config.p == 1:
                self.q = float("inf")
            elif self._config.p == float("inf"):
                self.q = 1.0
            else:
                self.q = self._config.p / (self._config.p - 1)

        if not hasattr(self, "jac_model"):
            self.jac_model = torch.func.jacrev(self._config.model.forward, chunk_size=1)
        if not hasattr(self, "derivative"):
            self.derivative = torch.zeros(
                (x.shape[0], self.n_classes, *x.shape[1:]), device=self._config.device, dtype=torch.float
            )

        # early stopping
        if hasattr(self, "err") and self.err < self._config.epsilon:
            return self.x_adv, True

        # change from one-hot to tensor of indexes
        targeted = torch.amin(y, dim=tuple(range(1, y.dim())))
        negative_rows = targeted < 0
        # targeted_mask = negative_rows.view(y.shape[0], *[1] * (y.dim() - 1)).expand_as(y)
        ids_y = y.abs().argmax(dim=-1)

        f = self._config.model(self.x_adv)
        # Run in-place operations that are equivalent to
        # f = f[rows, I].unsqueeze(1) - f
        # To avoid tensor cloning as they are quite memory-heavy
        dd = f[self.rows, self.I].unsqueeze(1)
        f.sub_(dd)
        f.neg_()

        del dd

        # Populate the derivative tensor
        # NOTE: If more memory might be available, change chunk_size=N with N being
        # the number of classes onto we want to vectorize the gradient computation.
        for b in self.rows.unbind():
            dy_dx = self.jac_model(self.x_adv[b].unsqueeze(0)).squeeze(2)
            self.derivative[b] = dy_dx
        # Run in-place operations that are equivalent to
        # derivative = derivative[rows, I].unsqueeze(1) - derivative
        # To avoid tensor cloning as they are quite memory-heavy
        dd = self.derivative[self.rows, self.I].unsqueeze(1)
        self.derivative.sub_(dd)
        self.derivative.neg_()

        del dd

        ###  r = -f * (|w|^p * sign(w)) / ||w||^p_p
        l = torch.zeros_like(self.rows)
        # Here we cover the targeted case
        # if the attack is targeted then there is only one way to go
        if (~negative_rows).any():
            l[~negative_rows] = ids_y[~negative_rows]

        # Here we cover the untargeted case:
        # In general `l` is computed as follows
        # l = argmin_{i\ne I} -|f_i| / ||w_i||^p
        # however, in this case it is `B x C`
        if negative_rows.any():
            new_l = torch.zeros_like(f[negative_rows])
            new_l = f[negative_rows].abs() / self.derivative[negative_rows].norm(
                p=self.q, dim=list(range(2, self.derivative.dim()))
            ).clamp_min(self._config.toll)
            # In this case, it avoids choosing the starting label.
            new_l[negative_rows, self.I] = float("inf")
            # In this case y is a class to avoid
            new_l[negative_rows, ids_y] = float("inf")
            l[negative_rows] = new_l.argmin(dim=-1)

        # This allows to reduce the memory in the gpu
        sel_derivative = self.derivative[self.rows, l]
        sel_f = f[self.rows, l].view(self.x_adv.shape[0], *[1] * (self.x_adv.dim() - 1))
        ###  r = -f_l * (|w_l|^p * sign(w_l)) / ||w_l||^p_p
        self.max_change = (
            (
                    torch.abs(sel_f)
                    / sel_derivative.norm(p=self.q, dim=list(range(1, sel_derivative.dim()))).clamp_min(
                self._config.toll)
            )
            .max()
            .item()
        )

        sel_f = -sel_f * sel_derivative.abs().pow(self.q - 1) * torch.sign(sel_derivative)
        # update the input and the counter
        self.x_adv.add_(sel_f)
        self.perturbation = self.x_adv - x

        # additional logging if required
        if log_on_tqdm:
            loop.set_postfix({"max_change": self.max_change})

        del sel_derivative, l
        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "perturbation",
            "x_adv",
            "n_classes",
            "I",
            "rows",
            "q",
            "jac_model",
            "derivative",
            "max_change"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
