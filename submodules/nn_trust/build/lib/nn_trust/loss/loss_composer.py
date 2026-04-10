from typing import Annotated, Optional, List

import torch
from annotated_types import Ge
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from nn_trust.loss.loss_factory import LossFactory as LF

class LossComposer(torch.nn.Module):
    r"""A module that helps to compose various loss with a cumulative.
    Basically the modules passed maps the input and then a 'reduction step'
    is performed to align the input to a single :class:`torch.Tensor`.
    """

    def __init__(self,
                 losses: dict[str, dict] | list[str],
                 weights: Optional[list[float]] = None) -> None:
        """
        :param losses: This dictionary represent the loss that are needed for a certain attack. The values could also be empty dict. In this case, the config the default one.
        :param weights: This represents the weights to associate to each loss.

        """
        super().__init__()


        # Using the Loss Composer to create each Composer's loss
        if weights:
            if len(weights) != len(losses):
                raise ValueError(
                    f"The weights' list, {len(weights)}, has different length compare to the losses' list, {len(losses)}."
                )
            self.weights = weights
        else:
            # by default if no weights are passed then it is a list of 1/len(classes)
            # hence, if len(losses)= 5, then [0.2, 0.2, 0.2, 0.2, 0.2]

            self.weights = [1 for _ in range(len(losses))]

        ####################### Filtering #######################
        ids = []
        if isinstance(losses, list):
            ids = losses
        if isinstance(losses, dict):
            ids = losses.keys()

        not_usable = [id_loss for id_loss in ids if id_loss not in LF.get_list_classes()]
        if len(not_usable) > 0:
            print(f"The following id are not usable: {not_usable}")
        #########################################################

        ####################### Loss Creation #######################
        if isinstance(losses, list):
            self.loss = [LF.create(class_id=loss_id) for loss_id in losses]
        elif isinstance(losses, dict):
            self.loss = [LF.create(class_id=loss_id, **config_loss) for loss_id, config_loss in losses.items()]
        else:
            raise ValueError(f"The type of losses, {type(losses)}, is not supported.")
        #############################################################

    def forward(
            self,
            x_adv: torch.Tensor,
            x: torch.Tensor,
            target: torch.Tensor,
            out_adv: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        #### checking the rightness of the input
        if (x_adv.dim() != 4) and (x_adv.shape != x.shape):
            raise ValueError("The adversarial input has not a proper shape.")
        # if out_adv.dim() != 2:
        # raise ValueError("The output of the network does not have a proper shape.")

        #### converting into a dictionary
        loss_kwargs = {
            "x_adv": x_adv,
            "x": x,
            "target": target.float(),
            "out_adv": out_adv.float()
        }

        out = 0.0
        for i in range(len(self.loss)):
            out += self.loss[i](**loss_kwargs) * self.weights[i]

        return out
