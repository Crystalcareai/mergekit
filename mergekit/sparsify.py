# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from enum import Enum

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    rescaled_random = "rescaled_random"
    magnitude_outliers = "magnitude_outliers"


def magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.numel())

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask


def magnitude_outliers(
    tensor: torch.Tensor, density: float, outlier_fraction: float = 0.2
):
    """Masks out smallest values in addition to large outliers.

    Args:
        tensor (torch.Tensor): The tensor to sparsify.
        density (float): The proportion of weights to retain.
        outlier_fraction (float): The fraction of removed elements that are large-magnitude.
    """
    if density >= 1:
        return tensor

    num_elems = tensor.numel()
    n = int(density * num_elems)

    to_remove = num_elems - n
    r_top = int(outlier_fraction * to_remove)
    r_bot = to_remove - r_top

    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    indices = torch.sort(w, descending=False).indices
    mask = torch.zeros_like(tensor)

    mask.view(-1)[indices[r_bot:-r_top]] = 1

    return tensor * mask


def bernoulli(
    tensor: torch.Tensor, density: float, rescale: bool = True
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = tensor.to(work_dtype) * mask
    if rescale:
        res /= density
    return res.to(tensor.dtype)


def sparsify(
    tensor: torch.Tensor, density: float, method: SparsificationMethod, **kwargs
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density, **kwargs)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale=False, **kwargs)
    elif method == SparsificationMethod.rescaled_random:
        return bernoulli(tensor, density=density, rescale=True, **kwargs)
    elif method == SparsificationMethod.magnitude_outliers:
        return magnitude_outliers(
            tensor, density=density, **kwargs
        )  # TODO: plumb through
    else:
        raise NotImplementedError(method)
