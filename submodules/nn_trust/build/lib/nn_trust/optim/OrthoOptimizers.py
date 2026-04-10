import torch
import torch.optim as optim


def ortho_grad_(params: torch.Tensor) -> None:
    p = params.data.view(-1)
    q = params.grad.data.view(-1)
    # Here we compute:
    # \grad_\perp L(theta) = \nabla L(\theta) - (\theta \nabla L(\theta))/(\theta.T\theta) \theta
    q_norm = q.dot(q)
    scaling = torch.dot(p, q).div(p.dot(p).clamp_min(1e-12))
    params.grad.data.sub_(params.data * scaling)
    # Rescale back the norm of the gradient to the original magnitude.
    q = params.grad.data.view(-1)
    # note: with this trick we can reduce the number of sqrt operations.
    qq_norm = (q_norm / q.dot(q).clamp_min(1e-12)).sqrt()
    params.grad.data.mul_(qq_norm)


class OrthoSGD(optim.SGD):
    r"""Implements the orthogonal projection of teh gradient combined with the SGD optimizer as in [1]_.
    The optimizer is the same as the :class:`torch.optim.SGD`, but before each optimization step an orthogonalization
    procedure of the gradient is performed. The parameters :math:`\theta`'s gradient are changed to :math:`\nabla_\perp L(\theta)`
    for some loss function :math:`L`. Namely,

    .. math::
        \nabla_\perp L(\theta) \coloneqq \nabla L(\theta) - \left( \frac{\theta^\top \nabla L(\theta)}{\theta^\top \theta} \right) \theta

    See :class:`torch.optim.SGD` for further documentation.

    .. Note:: it is about 1.5 times slower than the original SGD optimizer. However, the memory usage is the same as
        :class:`torch.optim.SGD` because all operations on the gradient are performed in-place.

    .. [1] Prieto, Lucas, Melih Barsbey, Pedro A. M. Mediano and Tolga Birdal. “Grokking at the Edge of Numerical Stability.” (2025).
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                for params in group["params"]:
                    ortho_grad_(params)

        super().step(closure=None)


class OrthoAdam(optim.Adam):
    r"""Implements the orthogonal projection of teh gradient combined with the Adam optimizer as in [1]_.
    The optimizer is the same as the :class:`torch.optim.Adam`, but before each optimization step an orthogonalization
    procedure of the gradient is performed. The parameters :math:`\theta`'s gradient are changed to :math:`\nabla_\perp L(\theta)`
    for some loss function :math:`L`. Namely,

    .. math::
        \nabla_\perp L(\theta) \coloneqq \nabla L(\theta) - \left( \frac{\theta^\top \nabla L(\theta)}{\theta^\top \theta} \right) \theta

    See :class:`torch.optim.Adam` for further documentation.

    .. Note:: it is about 1.5 times slower than the original Adam optimizer. However, the memory usage is the same as
        :class:`torch.optim.Adam` because all operations on the gradient are performed in-place.

    .. [1] Prieto, Lucas, Melih Barsbey, Pedro A. M. Mediano and Tolga Birdal. “Grokking at the Edge of Numerical Stability.” (2025).
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                for params in group["params"]:
                    ortho_grad_(params)

        super().step(closure=None)


class OrthoAdamW(optim.AdamW):
    r"""Implements the orthogonal projection of teh gradient combined with the AdamW optimizer as in [1]_.
    The optimizer is the same as the :class:`torch.optim.AdamW`, but before each optimization step an orthogonalization
    procedure of the gradient is performed. The parameters :math:`\theta`'s gradient are changed to :math:`\nabla_\perp L(\theta)`
    for some loss function :math:`L`. Namely,

    .. math::
        \nabla_\perp L(\theta) \coloneqq \nabla L(\theta) - \left( \frac{\theta^\top \nabla L(\theta)}{\theta^\top \theta} \right) \theta

    See :class:`torch.optim.AdamW` for further documentation.

    .. Note:: it is about 1.5 times slower than the original AdamW optimizer. However, the memory usage is the same as
        :class:`torch.optim.AdamW` because all operations on the gradient are performed in-place.

    .. [1] Prieto, Lucas, Melih Barsbey, Pedro A. M. Mediano and Tolga Birdal. “Grokking at the Edge of Numerical Stability.” (2025).
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                for params in group["params"]:
                    ortho_grad_(params)

        super().step(closure=None)
