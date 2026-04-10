import heapq
import logging
import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from pydantic import Field, field_validator
from torchvision import transforms

from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.core import AttackType, Knowledge, Task


class ParsimoniousAttackConfig(EvasionAttackConfig):
    batch_size: int = Field(
        default=-1,
        description="Number of pixels to process in each batch.",
        ge=-1,
        title="Batch size"
    )
    block_size: int = Field(
        default=64,
        description="Block size for the attack. Must be a power of 2.",
        ge=1
    )
    loss: Literal["cw", "xent"] = Field(
        default="xent",
        description="Loss function to use for the attack."
    )

    max_queries: int = Field(
        default=50,
        description="Maximum number of queries before stopping.",
        ge=1,
        title="Max number of queries"
    )

    @field_validator("block_size", mode="before")
    def valid_block_size(cls, v):
        if v & (v - 1) != 0:
            raise ValueError("block_size must be a power of 2.")
        return v


@EvasionAttackFactory.register(
    name="Parsimonious",
    description="A parsimonious attack.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class ParsimoniousAttack(EvasionAttack):
    CONFIG_T = ParsimoniousAttackConfig

    def generate(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> torch.Tensor:
        x = x.to(self._config.device)
        if x.dim() == 4:
            x = x.squeeze(0)

        y = y.to(self._config.device)
        self._config.model.to(self._config.device)

        # Define a function to transform the perturbation to the sample dimension
        resize2sample = transforms.Resize(x.size()[1:])

        n_queries = 0
        curr_block_size = self._config.block_size
        max_block_size = min(x.shape[:-2])
        if curr_block_size > max_block_size:
            logging.warning("Block size is larger than the image size. Reducing to image size.")
            curr_block_size = max_block_size

        res_size = 2 ** math.floor(math.log2(min(x.size()[1:])))  # size of the perturbation
        n_channels = x.size()[0]

        # Initialize perturbation to -epsilon (found in the implementation). So nothing is in the working set
        perturbation = -self._config.epsilon * torch.ones((n_channels, res_size, res_size), device=self._config.device)

        # Loop on the attack algorithm on the current block size
        while n_queries < self._config.max_queries and curr_block_size >= 1:
            # Define a permutation over the blocks
            perturbation_blocks = perturbation.unfold(1, curr_block_size, curr_block_size).unfold(
                2, curr_block_size, curr_block_size
            )
            order = torch.randperm(
                torch.prod(torch.tensor(perturbation_blocks.size()[:3])).item(), device=self._config.device
            )
            order = (
                order // perturbation_blocks.size(1) // perturbation_blocks.size(2),
                order // perturbation_blocks.size(1) % perturbation_blocks.size(2),
                order % perturbation_blocks.size(2),
            )
            n_blocks = order[0].size(0)
            # Adapt the batch size
            batch_size = min(self._config.batch_size, n_blocks) if self._config.batch_size > 0 else n_blocks

            for i in range(math.ceil(n_blocks / batch_size)):
                # Calculate current loss
                adv_sample = resize2sample(perturbation) + x
                output = self._config.model(adv_sample.unsqueeze(0))
                n_queries += 1

                # Early stopping with a successfully attack
                if self._compare(output, y).size(0) > 0:
                    if self._config.verbose:
                        print("Early stopping on current sample.")
                    if ext_results:
                        ext_results[n_queries] = n_queries
                    return resize2sample(perturbation)

                loss = self._run_loss(output, y)

                # --- Local search ---
                # Working set on the batch
                si = 1 * torch.any(
                    perturbation_blocks[
                        order[0][i * batch_size: max((i + 1) * batch_size, n_blocks)],
                        order[1][i * batch_size: max((i + 1) * batch_size, n_blocks)],
                        order[2][i * batch_size: max((i + 1) * batch_size, n_blocks)],
                    ]
                    > 0,
                    dim=(1, 2),
                )
                for _ in range(self._config.max_iters):
                    # -- Lazy insertion --
                    # Elements out of the working set
                    indices = torch.where(si == 0)[0]

                    # Batch with perturbation, where each element have one block inserted in the working set
                    perturbation_batch = perturbation.repeat(
                        (indices.size(0), *[1 for _ in range(len(adv_sample.size()))])
                    )
                    perturbation_batch_blocks = perturbation_batch.unfold(2, curr_block_size, curr_block_size).unfold(
                        3, curr_block_size, curr_block_size
                    )
                    perturbation_batch_blocks[
                        torch.arange(indices.size(0)), order[0][indices], order[1][indices], order[2][indices]
                    ] = self._config.epsilon
                    adv_sample_batch = resize2sample(perturbation_batch) + x
                    outputs = self._config.model(adv_sample_batch)
                    n_queries += indices.size(0)

                    # Early stopping if one of those sample is already good enough
                    match_indices = self._compare(outputs, y)
                    if match_indices.size(0) != 0:
                        if self._config.verbose:
                            print("Early stopping on lazy insertion batch.")
                        if ext_results:
                            ext_results[n_queries] = n_queries
                        perturbation = perturbation_batch[match_indices[0]]
                        return resize2sample(perturbation)

                    losses = self._run_loss(outputs, y)
                    margin = losses - loss
                    priority_queue = [(m, int(indices[i].item())) for (i, m) in enumerate(margin)]
                    heapq.heapify(priority_queue)

                    # The first iteration is already computed
                    if priority_queue:
                        best_m, best_idx = heapq.heappop(priority_queue)
                        perturbation_blocks[
                            order[0][i * batch_size + best_idx],
                            order[1][i * batch_size + best_idx],
                            order[2][i * batch_size + best_idx],
                        ] = self._config.epsilon
                        loss += best_m
                        si[best_idx] = 1

                    while priority_queue:
                        cand_m, can_idx = heapq.heappop(priority_queue)
                        perturbation_blocks[
                            order[0][i * batch_size + can_idx],
                            order[1][i * batch_size + can_idx],
                            order[2][i * batch_size + can_idx],
                        ] = self._config.epsilon
                        adv_sample = resize2sample(perturbation) + x
                        output = self._config.model(adv_sample.unsqueeze(0))
                        n_queries += 1
                        new_loss = self._run_loss(output, y)
                        new_margin = new_loss[0] - loss

                        if not priority_queue or new_margin <= priority_queue[0][0]:
                            if new_margin > 0:
                                break
                            si[can_idx] = 1
                            loss = new_loss[0]
                            if self._compare(output, y).size(0) > 0:
                                if self._config.verbose:
                                    print("Early stopping on lazy insertion queue.")
                                if ext_results:
                                    ext_results[n_queries] = n_queries
                                return resize2sample(perturbation)
                        else:
                            perturbation_blocks[
                                order[0][i * batch_size + can_idx],
                                order[1][i * batch_size + can_idx],
                                order[2][i * batch_size + can_idx],
                            ] = -self._config.epsilon
                            heapq.heappush(priority_queue, (new_margin, can_idx))

                    # -- Lazy deletion --
                    # Elements in the working set
                    indices = torch.where(si == 1)[0]

                    # Batch with perturbation, where each element have one block removed from the working set
                    perturbation_batch = perturbation.repeat(
                        (indices.size(0), *[1 for _ in range(len(adv_sample.size()))])
                    )
                    perturbation_batch_blocks = perturbation_batch.unfold(2, curr_block_size, curr_block_size).unfold(
                        3, curr_block_size, curr_block_size
                    )
                    perturbation_batch_blocks[
                        torch.arange(indices.size(0)), order[0][indices], order[1][indices], order[2][indices]
                    ] = -self._config.epsilon
                    adv_sample_batch = resize2sample(perturbation_batch) + x
                    outputs = self._config.model(adv_sample_batch)
                    n_queries += indices.size(0)

                    # Early stopping if one of those sample is already good enough
                    match_indices = self._compare(outputs, y)
                    if match_indices.size(0) != 0:
                        if self._config.verbose:
                            print("Early stopping on lazy deletion batch.")
                        if ext_results:
                            ext_results[n_queries] = n_queries
                        perturbation = perturbation_batch[match_indices[0]]
                        return resize2sample(perturbation)

                    losses = self._run_loss(outputs, y)
                    margin = losses - loss
                    priority_queue = [(m, int(indices[i].item())) for (i, m) in enumerate(margin)]
                    heapq.heapify(priority_queue)

                    # The first iteration is already computed
                    if priority_queue:
                        best_m, best_idx = heapq.heappop(priority_queue)
                        perturbation_blocks[
                            order[0][i * batch_size + best_idx],
                            order[1][i * batch_size + best_idx],
                            order[2][i * batch_size + best_idx],
                        ] = -self._config.epsilon
                        loss += best_m
                        si[best_idx] = 0

                    while priority_queue:
                        cand_m, can_idx = heapq.heappop(priority_queue)
                        perturbation_blocks[
                            order[0][i * batch_size + can_idx],
                            order[1][i * batch_size + can_idx],
                            order[2][i * batch_size + can_idx],
                        ] = -self._config.epsilon
                        adv_sample = resize2sample(perturbation) + x
                        output = self._config.model(adv_sample.unsqueeze(0))
                        n_queries += 1
                        new_loss = self._run_loss(output, y)
                        new_margin = new_loss[0] - loss

                        if not priority_queue or new_margin <= priority_queue[0][0]:
                            if new_margin > 0:
                                break
                            si[can_idx] = 0
                            loss = new_loss[0]
                            if self._compare(output, y).size(0) > 0:
                                if self._config.verbose:
                                    print("Early stopping on lazy deletion queue.")
                                if ext_results:
                                    ext_results[n_queries] = n_queries
                                perturbation = resize2sample(perturbation)
                                return resize2sample(perturbation)
                        else:
                            perturbation_blocks[
                                order[0][i * batch_size + can_idx],
                                order[1][i * batch_size + can_idx],
                                order[2][i * batch_size + can_idx],
                            ] = self._config.epsilon
                            heapq.heappush(priority_queue, (new_margin, can_idx))

            # Split block before re-search for a solution
            if self._config.verbose:
                print("Splitting block.")
            curr_block_size //= 2

        # If the attack is not found, return the last perturbation found
        if self._config.verbose:
            print("Attack not found.")
        if ext_results:
            ext_results[n_queries] = n_queries
        return resize2sample(perturbation) + x

    def _run_loss(
            self,
            outputs: torch.Tensor,
            label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss or carlini-wagner loss.
        """
        if self._config.loss == "xent":
            xent = F.cross_entropy(outputs, label.repeat(outputs.size(0), 1).to(dtype=torch.float32), reduction="none")
            if self._config.targeted:
                return xent
            else:
                return -xent
        elif self._config.loss == "cw":
            outputs = F.softmax(outputs, dim=0)
            p_max = torch.log(torch.max(outputs, dim=1)[0] + 1e-10)
            p_label = torch.log(outputs[:, torch.argmax(label)] + 1e-10)
            if self._config.targeted:
                return p_max - p_label
            else:
                return p_label - p_max
        else:
            raise ValueError(f'Argument "loss" for {self.__class__.__name__} can not be {self._config.loss}')

    def _compare(
            self,
            outputs: torch.Tensor,
            label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given the outputs and one label, compute at what indexes the attack worked.
        """
        if self._config.targeted:
            indexes = torch.argmax(outputs, dim=1) == torch.argmax(label)
        else:
            indexes = torch.argmax(outputs, dim=1) != torch.argmax(label)
        return torch.nonzero(indexes).flatten()
