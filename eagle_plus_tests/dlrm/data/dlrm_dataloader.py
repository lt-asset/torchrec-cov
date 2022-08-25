#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DAYS,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset

STAGES = ["train", "val", "test"]


class TestDataset(Dataset):
    def __init__(self, inputs, device):
        self.inputs = [input.to(device) for input in inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def _get_random_dataloader(
    args: argparse.Namespace,
    pin_memory: bool,
) -> DataLoader:
    print("in here")
    randDataset = RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=args.batch_size,
        hash_size=args.num_embeddings,
        hash_sizes=args.num_embeddings_per_feature,
        manual_seed=args.seed,
        ids_per_feature=1,
        num_dense=len(DEFAULT_INT_NAMES),
    )
    rank = dist.get_rank()
    if args.backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    print(device)
    iterator = iter(randDataset)

    batch_list = []
    for i in range(100):
        batch = next(iterator)
        batch_list.append(batch)

    return DataLoader(
        TestDataset(batch_list, device),
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
    pin_memory: bool,
) -> DataLoader:
    files = os.listdir(args.in_memory_binary_criteo_path)

    def is_final_day(s: str) -> bool:
        return f"day_{DAYS - 1}" in s

    if stage == "train":
        # Train set gets all data except from the final day.
        files = list(filter(lambda s: not is_final_day(s), files))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = (
            dist.get_rank()
            if stage == "val"
            else dist.get_rank() + dist.get_world_size()
        )
        world_size = dist.get_world_size() * 2

    stage_files: List[List[str]] = [
        sorted(
            map(
                lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
                filter(lambda s: kind in s, files),
            )
        )
        for kind in ["dense", "sparse", "labels"]
    ]
    dataloader = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            *stage_files,  # pyre-ignore[6]
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=args.shuffle_batches,
            hashes=args.num_embeddings_per_feature
            if args.num_embeddings is None
            else ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    pin_memory = (backend == "nccl") if args.pin_memory is None else args.pin_memory

    if args.in_memory_binary_criteo_path is None:
        return _get_random_dataloader(args, pin_memory)
    else:
        return _get_in_memory_dataloader(args, stage, pin_memory)
