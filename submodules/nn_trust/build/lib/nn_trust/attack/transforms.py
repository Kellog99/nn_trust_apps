import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF


class CreaseTransform(nn.Module):
    r"""
    Transformation that implement the crease transformation described in DAP

    .. math::

        \text{multiplier}(x,y) = 1-\frac{\sin^2 \theta \cdot [(x-x_0)^2 + (y-y_0)^2]}{\text{width}^2 + \text{height}^2}

    source: A. Guesmi, et al., "DAP: A Dynamic Adversarial Patch for Evading Person Detectors," 2023.
    """

    def __init__(
        self,
        center_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
        angle_range: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        if center_range is None:
            self._center_range = (0.0, 1.0), (0.0, 1.0)
        else:
            self._center_range = center_range
        if angle_range is None:
            self._angle_range = (0.0, 360.0)
        else:
            self._angle_range = angle_range

    def forward(
        self,
        image: torch.Tensor,
        center: Optional[tuple[int, int]] = None,
        direction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply a single crease effect to an image.

        Args:
        :param image: Input image tensor of shape (C, H, W)
        :param center: (x, y) coordinates for the center of the crease. If None, randomly chosen.
        :param direction: 2D vector for the direction of the crease. If None, randomly chosen.

        :return: Transformed image tensor
        """
        assert len(image.size()) == 3, "Expected 3D tensor C x H x W."
        n_ch, height, width = image.size()
        device = image.device

        # Crease Initiation
        if center is None:
            cr = self._center_range
            x0 = random.randint(int(cr[0][0] * width), int(cr[0][1] * width - 1))
            y0 = random.randint(int(cr[1][0] * height), int(cr[1][1] * height - 1))
        else:
            x0, y0 = center

        # Direction of Crease
        if direction is None:
            angle = random.uniform(*self._angle_range)
            vector = torch.tensor([math.cos(math.radians(angle)), math.sin(math.radians(angle))], device=device)
        else:
            vector = direction

        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        coords = torch.stack((x, y), dim=-1).float()

        # Calculate displacement
        diff = coords - torch.tensor([x0, y0], device=device)
        dist_squared = torch.sum(diff**2, dim=-1)

        # Calculate angle between diff and vector
        product = torch.sum(diff * vector, dim=-1)
        magnitudes = torch.sqrt(dist_squared) * vector.pow(2).sum().sqrt()
        cos_theta = product / (magnitudes + 1e-8)
        sin_squared_theta = 1 - cos_theta**2

        # Calculate multiplier
        multiplier = 1 - sin_squared_theta * dist_squared / (width**2 + height**2)

        # Apply displacement
        displacement = vector.unsqueeze(0).unsqueeze(0) * multiplier.unsqueeze(-1) * (width + height) / 2
        grid = coords.unsqueeze(0) + displacement

        # Normalize grid coordinates to [-1, 1]
        min1, max1 = grid[:, :, :, 0].min(), grid[:, :, :, 0].max()
        min2, max2 = grid[:, :, :, 1].min(), grid[:, :, :, 1].max()
        grid[:, :, :, 0] = -2 * (grid[:, :, :, 0] - min1) / (max1 - min1) + 1.0
        grid[:, :, :, 1] = -2 * (grid[:, :, :, 1] - min2) / (max2 - min2) + 1.0

        # Apply transformation
        image = F.grid_sample(
            image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(0)
        return image


class AddUniformNoiseTransform(nn.Module):
    """
    Transformation that add a random noise on top of the image (simulates camera noise).
    """

    def __init__(self, noise_range: tuple[float, float] = (0.0, 0.1)):
        super().__init__()
        self._noise_range = noise_range

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r1, r2 = self._noise_range
        noise = (r2 - r1) * torch.rand_like(x) + r1
        return torch.clamp(x + noise, 0, 1)


class RandomRotation(nn.Module):
    r"""
    Transformation that implements a general rotation of a 2D image as a 3D sheet of paper.

    :param center: a torch tensor of shape (3, 1) representing the
        center of each rotation in pixel resolution.
    :param angle_range: range of any rotation in radians. If `angle_range = (value, value)`,
        then the rotates at the specified angle `value`. Default is (0, 2 * pi).
    :param rot_axis: available axis of rotations may be applied, dependently on the probability.
        Default is [0,1,2].
    :param p_axis: probability that each rotation is applied. Default is [1/3, 1/3, 1/3].
    :param camera_projection: boolean value, default is True. If true a pinhole-camera projection is performed,
        otherwise the image is transformed using an orthogonal projection.
    :param centering: boolean value, default is True. If true rescales the image in a way to fit the
        original size of the image, thus not cutting any part of the image. If false, under certain
        rotations, the image could be cut-off outside the original format.

    :heading level: 3
    Examples

    Let x be an image of shape (3, 28, 28). To randomly rotate it
    by an angle of (-pi/6, pi/6) around the center of the image,
    only on the z-axis with 100% probability, we can use the following code:

    .. code-block:: python

        zRot = RandomRotation(
            torch.tensor([14, 14, 0.0]).view(-1, 1), angle_range=(-3.14159 / 6, 3.14159 / 6), rot_axis=[2], p_axis=[0, 0, 1]
        )

        res = zRot(x)

        plt.imshow(res)

    To compose multiple rotations, we can use a similar approach:

    .. code-block:: python

        yzRot = RandomRotation(
            torch.tensor([14, 14, 0.0]).view(-1, 1), angle_range=(-3.14159 / 6, 3.14159 / 6), rot_axis=[1, 2], p_axis=[0, 1, 1]
        )

        res = yzRot(x)

        plt.imshow(res)

    which rotates with 100% probability on both the y and z-axis the image x.
    """

    def __init__(
        self,
        center: torch.Tensor,
        angle_range: Optional[Tuple[float, float]] = None,
        rot_axis: Optional[List[int]] = None,
        p_axis: Optional[List[float]] = None,
        camera_projection: bool = True,
        centering: bool = True,
    ):
        super().__init__()

        # Check initialization conditions.
        if center.numel() != 3 or list(center.shape) != [3, 1]:
            raise ValueError("The center must be a tensor of shape (3, 1).")

        if angle_range is not None:
            if len(angle_range) != 2:
                raise ValueError("Angle range must be a Tuple of two values.")

            if angle_range[0] > angle_range[1]:
                raise ValueError("Angle range must be a tuple (minimum value, maximum value).")

        # check if the axis list is empty and the values are in the correct range.
        if rot_axis is not None:
            if not rot_axis:
                raise ValueError("Provide either None or a non-empty list of axis.")
            if 0 > min(rot_axis) > 2 or 2 < max(rot_axis) < 0:
                raise ValueError("The available rotation axis are between 0 and 2.")

        # Set the protected variables.
        self._center = center
        if angle_range is None:
            self._angle_range = (0.0, 2 * torch.pi)
        else:
            self._angle_range = angle_range

        if rot_axis is None:
            self._rot_axis = [0, 1, 2]
        else:
            # remove duplicated axis if any.
            self._rot_axis = list(set(rot_axis))
        # If no distribution of the rotation axis to choose is provided
        # use a uniform distribution.
        if p_axis is None:
            p_axis = [1.0 / len(self._rot_axis)] * len(self._rot_axis)
        self._prob_axis = p_axis
        self._camera_projection = camera_projection
        self._centering = centering

    @staticmethod
    def generate_rotation_matrix(axis: int, angle: float, device: str):
        ca, sa = math.cos(angle), math.sin(angle)
        # z-axis rotation matrix
        if axis == 2:
            return torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], device=device)
        # y-axis rotation matrix
        elif axis == 1:
            return torch.tensor([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], device=device)
        # x-axis rotation matrix
        elif axis == 0:
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], device=device)
        else:
            raise ValueError("No rotation available on the specified axis.")

    def forward(self, image: torch.Tensor, center: Optional[torch.Tensor] = None):
        r"""
        Given an image of shape (C, H, W) returns an image of same shape, but randomly
        rotated on the specified axis according to the specified parameters.

        :raises ValueError: if the image tensor's shape is not (C, H, W).
        """

        if len(image.size()) != 3:
            raise ValueError("Expected 3D tensor of shape C x H x W.")

        n_ch, height, width = image.size()
        device = image.device

        if center is None:
            center = self._center
        center = center.to(device)

        # Sample the rotation to perform.
        ps = torch.rand(3, device=device)
        rotation_matrix = torch.eye(3, device=device)
        # Compose the rotations in a random order to compensate for the non-commutativity
        # of rotation matrices.
        for i in random.sample(self._rot_axis, len(self._rot_axis)):
            if ps[i] <= self._prob_axis[i]:
                # sample uniformly an angle in the given range.
                angle = random.random() * (self._angle_range[1] - self._angle_range[0]) + self._angle_range[0]
                # generate the rotation matrix and compose it with the previous rotations.
                tmp_rot = RandomRotation.generate_rotation_matrix(axis=i, angle=angle, device=device)
                rotation_matrix.data = (rotation_matrix @ tmp_rot).data

        z = 1.0
        vertices = torch.tensor([[0, 0, z], [0, height, z], [width, 0, z], [width, height, z]], device=device).T
        # apply rotation
        vertices.sub_(center)
        vertices.data = (rotation_matrix @ vertices).data
        # If the camera projection is chosen, normalize on the z-axis by noting
        # that the maximum depth is achieved on the diagonal of the image starting
        # from the center of the rotation.
        if self._camera_projection:
            max_dxyz = torch.norm(vertices - center, dim=0).max()
            vertices[-1, :] = vertices[-1, :] / max_dxyz + 1.00001
            vertices.data = vertices.data / vertices[-1, :]
        # Move back to center
        vertices.add_(center)

        # Resize inside the original image the transformed image.
        if self._centering:
            # Compute the polygonal bounding box
            vals, _ = vertices.max(dim=1)
            x_max, y_max, _ = vals
            vals, _ = vertices.min(dim=1)
            x_min, y_min, _ = vals
            # Compute the minimal scaling ratio to fit the polygonal into the original image
            vert_width, vert_height = abs(x_max - x_min), abs(y_max - y_min)
            ratio = min(width / vert_width, height / vert_height)

            # Center the coordinates and scale by the best ratio to fit the original bounding box.
            vertices_center = torch.tensor([vert_width / 2, vert_height / 2, 0.0], device=device).view(-1, 1)
            vertices.sub_(vertices_center)
            vertices.mul_(ratio)
            # Re-center the coordinate space to the original bounding box.
            vals, _ = vertices.min(dim=1)
            x_min, y_min, _ = vals
            vertices.sub_(torch.tensor([x_min, y_min, 0.0], device=device).view(-1, 1))
            vals, _ = vertices.max(dim=1)
            x_max, y_max, _ = vals
            vertices.add_(torch.tensor([(width - x_max) / 2, (height - y_max) / 2, 0.0], device=device).view(-1, 1))

        # convert from tensor notation to list again
        xs, ys, _ = vertices.tolist()
        endpoints = list(map(list, zip(xs, ys, strict=False)))
        startpoints = [[0, 0], [0, height], [width, 0], [width, height]]
        return TF.functional.perspective(image, startpoints=startpoints, endpoints=endpoints)
