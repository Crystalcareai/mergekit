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

import json
import logging
import os
from typing import Dict

import safetensors
import torch


class TensorWriter:
    def __init__(self, out_path: str, max_shard_size: int = 1000 * 1000 * 1000 * 5, safe_serialization: bool = True) -> None:
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0
        self.total_size = 0
        self.sharded_tensors = {}

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        tensor_size = tensor.numel() * tensor.element_size()
        if self.current_shard and self.current_shard_size + tensor_size > self.max_shard_size:
            self.flush_current_shard()

        if clone:
            tensor = tensor.clone()

        if tensor_size > self.max_shard_size:
            # Handle large tensors that exceed the maximum shard size
            num_shards = (tensor_size + self.max_shard_size - 1) // self.max_shard_size
            tensor_shards = tensor.chunk(num_shards)
            for i, tensor_shard in enumerate(tensor_shards):
                shard_name = f"{name}.shard_{i}"
                self.current_shard[shard_name] = tensor_shard
                self.current_shard_size += tensor_shard.numel() * tensor_shard.element_size()
                if shard_name not in self.sharded_tensors:
                    self.sharded_tensors[shard_name] = []
                self.sharded_tensors[shard_name].append(f"model-{self.shards_written+1:05d}-of-00007.safetensors")
        else:
            self.current_shard[name] = tensor
            self.current_shard_size += tensor_size

        self.total_size += tensor_size

    def flush_current_shard(self):
        if not self.current_shard:
            return

        logging.info(f"Writing shard #{self.shards_written+1} to disk")

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{self.shards_written+1:05d}-of-00007.{extension}"
        for key in self.current_shard:
            if key not in self.sharded_tensors:
                self.weight_map[key] = shard_name

        shard_path = os.path.join(self.out_path, shard_name)
        if self.safe_serialization:
            self._save_st(shard_path)
        else:
            torch.save(self.current_shard, shard_path)

        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written += 1

    def finalize(self):
        self.flush_current_shard()

        logging.info("Finalizing shard names")

        prefix, extension = self._get_name_components()

        for key, shard_names in self.sharded_tensors.items():
            self.weight_map[key] = shard_names

        with open(os.path.join(self.out_path, f"{prefix}.{extension}.index.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "metadata": {"mergekit_version": "0.0.4.1"},
                    "weight_map": self.weight_map,
                },
                file,
            )

    def _get_name_components(self):
        if self.safe_serialization:
            return "model", "safetensors"
        return "pytorch_model", "bin"

    def _save_st(self, shard_path: str):
        def _do_save():
            safetensors.torch.save_file(
                self.current_shard,
                shard_path,
                metadata={"format": "pt"},
            )

        try:
            _do_save()
        except RuntimeError as e:
            if (
                len(e.args) > 0
                and isinstance(e.args[0], str)
                and "share memory" in e.args[0]
            ):
                logging.warning(
                    "Your model has duplicated tensors but the --clone-tensors "
                    "flag is not set."
                )
                self.current_shard = {
                    key: self.current_shard[key].clone() for key in self.current_shard
                }
                _do_save()
            else:
                raise
