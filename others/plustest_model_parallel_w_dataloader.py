#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import unittest
from collections import defaultdict, OrderedDict
from typing import cast, Dict, List, Optional, Tuple, Type, Union

import hypothesis.strategies as st
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchrec.distributed as trec_dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, Verbosity

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.utils import Batch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    EmbeddingBagSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.fused_embeddingbag import ShardedFusedEmbeddingBagCollection

from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestDenseArch,
    TestOverArch,
    TestSparseArch,
    TestSparseNNBase,
)

from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    EmbeddingTableConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.test_utils import get_free_port, seed_and_log, skip_if_asan_class

from .models.simple_models import TestSparseNN

from .plustest_util_random import RandomRecDataset


class ModelParallelTestShared(MultiProcessTestBase):
    @seed_and_log
    def setUp(self) -> None:
        super().setUp()

        num_features = 4

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 2) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        # self.weighted_tables = [
        #     EmbeddingBagConfig(
        #         num_embeddings=(i + 1) * 10,
        #         embedding_dim=(i + 2) * 4,
        #         name="weighted_table_" + str(i),
        #         feature_names=["weighted_feature_" + str(i)],
        #     )
        #     for i in range(num_weighted_features)
        # ]
        self.weighted_tables = None

        self.embedding_groups = {
            "group_0": ["feature_" + str(i) for i in range(num_features)]
        }

    def _test_sharding(
        self,
        sharders: List[ModuleSharder[nn.Module]],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSparseNN,
        use_dataloader: bool = False,
        num_workers: int = 0,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            backend=backend,
            optim=EmbOptimType.EXACT_SGD,
            constraints=constraints,
            use_dataloader=use_dataloader,
            num_workers=num_workers,
        )


def generate_inputs(
    world_size: int,
    tables: List[EmbeddingTableConfig],
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    batch_size: int = 4,
    num_float_features: int = 16,
) -> Tuple[ModelInput, List[ModelInput]]:
    return ModelInput.generate(
        batch_size=batch_size,
        world_size=world_size,
        num_float_features=num_float_features,
        tables=tables,
        weighted_tables=weighted_tables or [],
    )


class TestDataset(Dataset):
    def __init__(self, inputs: List[Batch], device):
        self.inputs = [input.to(device) for input in inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def get_random_dataset(
    embedding_bag_configs: List[EmbeddingBagConfig],
    batch_size: int = 64,
    num_batches: int = 10,
    num_dense_features: int = 1024,
    pooling_factors: Optional[Dict[str, int]] = None,
    device: torch.device = torch.device("cpu"),
) -> IterableDataset[Batch]:
    # Generate a random dataset according to the embedding bag configs

    if pooling_factors is None:
        pooling_factors = {}

    keys = []
    ids_per_features = []
    hash_sizes = []

    for table in embedding_bag_configs:
        for feature_name in table.feature_names:
            keys.append(feature_name)
            # guess a pooling factor here
            ids_per_features.append(pooling_factors.get(feature_name, 64))
            hash_sizes.append(table.num_embeddings)
    print("in get random dataset", device)
    randDataset = RandomRecDataset(
        keys=keys,
        batch_size=batch_size,
        hash_sizes=hash_sizes,
        ids_per_features=ids_per_features,
        num_dense=num_dense_features,
        num_batches=num_batches,
        device=device,
    )
    return randDataset
    iterator = iter(randDataset)

    batch_list = []
    for i in range(num_batches):
        batch = next(iterator)
        batch_list.append(batch)

    return TestDataset(batch_list, device)


def _get_random_dataloader(
    tables,
    num_float_features,
    num_workers,
    backend,
    device,
    pin_memory=None,
) -> DataLoader:
    dataset = get_random_dataset(
        embedding_bag_configs=tables,
        batch_size=1,
        num_dense_features=num_float_features,
        device=device,
    )

    if pin_memory is None:
        pin_memory = backend == "nccl"

    return DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        # pin_memory=pin_memory,
        num_workers=num_workers,
    )


from torchrec.distributed.test_utils.test_sharding import generate_inputs


def gen_model_and_input(
    model_class: TestSparseNNBase,
    tables: List[EmbeddingTableConfig],
    embedding_groups: Dict[str, List[str]],
    world_size: int,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    num_float_features: int = 16,
    dense_device: Optional[torch.device] = None,
    sparse_device: Optional[torch.device] = None,
    use_dataloader: bool = False,
    num_workers: int = 0,
    backend: str = "gloo",
):
    torch.manual_seed(0)
    print(dense_device, sparse_device)
    model = model_class(
        tables=cast(List[BaseEmbeddingConfig], tables),
        num_float_features=num_float_features,
        weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
        embedding_groups=embedding_groups,
        dense_device=dense_device,
        sparse_device=sparse_device,
    )

    if use_dataloader:
        dataloader = _get_random_dataloader(
            tables=tables,
            num_float_features=num_float_features,
            num_workers=num_workers,
            backend=backend,
            device=dense_device,
        )
        return model, dataloader
    else:
        inputs = [
            generate_inputs(
                world_size=world_size,
                tables=tables,
                weighted_tables=weighted_tables,
                num_float_features=num_float_features,
            )
        ]
        return (model, inputs)


TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.


def sharding_single_rank_test(
    rank: int,
    world_size: int,
    model_class: TestSparseNNBase,
    embedding_groups: Dict[str, List[str]],
    tables: List[EmbeddingTableConfig],
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    optim: EmbOptimType,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    use_dataloader: bool = False,
    num_workers: int = 0,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Generate model & inputs.
        if backend == "gloo":
            dense_device = torch.device("cpu")
            sparse_device = torch.device("cpu")
        elif backend == "nccl":
            dense_device = torch.device(f"cuda:{rank}")
            sparse_device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(sparse_device)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        (global_model, inputs_data) = gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
            num_float_features=16,
            dense_device=dense_device,
            sparse_device=sparse_device,
            use_dataloader=use_dataloader,
            num_workers=num_workers,
            backend=backend,
        )
        print(inputs_data)
        if use_dataloader:
            train_iterator = iter(inputs_data)
            inputs = next(train_iterator)
        else:
            inputs = inputs_data
        dense_r = inputs.dense_features
        sparse_r = inputs.sparse_features
        print(dense_r)
        print(sparse_r)
        print(dense_r.device)
        print(sparse_r.device())
        # print(next(global_model.parameters()).device)

        print(global_model(inputs))

        (_, inputs_data_val) = gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
            num_float_features=16,
            use_dataloader=use_dataloader,
            num_workers=num_workers,
            backend=backend,
        )
        val_iterator = iter(inputs_data_val)
        # exit()
        # global_model = global_model.to(ctx.device)
        # global_input = inputs[0][0].to(ctx.device)
        # local_input = inputs[0][1][rank].to(ctx.device)

        # # Shard model.
        # local_model = model_class(
        #     tables=cast(List[BaseEmbeddingConfig], tables),
        #     weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
        #     embedding_groups=embedding_groups,
        #     dense_device=ctx.device,
        #     sparse_device=torch.device("meta"),
        #     num_float_features=16,
        # )

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.collective_plan(global_model, sharders, ctx.pg)
        # """
        # Simulating multiple nodes on a single node. However, metadata information and
        # tensor placement must still be consistent. Here we overwrite this to do so.

        # NOTE:
        #     inter/intra process groups should still behave as expected.

        # TODO: may need to add some checks that only does this if we're running on a
        # single GPU (which should be most cases).
        # """

        # for group in plan.plan:
        #     for _, parameter_sharding in plan.plan[group].items():
        #         if (
        #             parameter_sharding.sharding_type
        #             in {
        #                 ShardingType.TABLE_ROW_WISE.value,
        #                 ShardingType.TABLE_COLUMN_WISE.value,
        #             }
        #             and ctx.device.type != "cpu"
        #         ):
        #             sharding_spec = parameter_sharding.sharding_spec
        #             if sharding_spec is not None:
        #                 # pyre-ignore
        #                 for shard in sharding_spec.shards:
        #                     placement = shard.placement
        #                     rank: Optional[int] = placement.rank()
        #                     assert rank is not None
        #                     shard.placement = torch.distributed._remote_device(
        #                         f"rank:{rank}/cuda:{rank}"
        #                     )

        global_model = DistributedModelParallel(
            global_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=sharders,
            device=ctx.device,
        )
        # global_model(inputs)

        # dense_optim = KeyedOptimizerWrapper(
        #     dict(local_model.named_parameters()),
        #     lambda params: torch.optim.SGD(params, lr=0.1),
        # )
        # local_opt = CombinedOptimizer([local_model.fused_optimizer, dense_optim])

        # # Load model state from the global model.
        # copy_state_dict(local_model.state_dict(), global_model.state_dict())

        # # Run a single training step of the sharded model.
        # local_pred = gen_full_pred_after_one_step(local_model, local_opt, local_input)

        # all_local_pred = []
        # for _ in range(world_size):
        #     all_local_pred.append(torch.empty_like(local_pred))
        # dist.all_gather(all_local_pred, local_pred, group=ctx.pg)

        # # Run second training step of the unsharded model.
        assert optim == EmbOptimType.EXACT_SGD
        global_opt = torch.optim.SGD(global_model.parameters(), lr=0.1)

        train_pipeline = TrainPipelineSparseDist(
            global_model,
            global_opt,
            ctx.device,
        )

        train_pipeline._model.train()
        combined_iterator = itertools.chain(
            train_iterator,
            itertools.islice(val_iterator, TRAIN_PIPELINE_STAGES - 1),
        )
        result = train_pipeline.progress(combined_iterator)
        print(result)
        # model = train_pipeline._model
        # model.eval()
        # model = model.to(ctx.device)
        # model(inputs)

        # global_pred = gen_full_pred_after_one_step(
        #     global_model, global_opt, global_input
        # )

        # # Compare predictions of sharded vs unsharded models.
        # # torch.testing.assert_allclose(global_pred, torch.cat(all_local_pred))
        # # LT: remove deprecated API
        # # print('test_sharding.py: torch.testing.assert_close(global_pred, torch.cat(all_local_pred))')
        # torch.testing.assert_close(global_pred, torch.cat(all_local_pred))


def gen_full_pred_after_one_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    input: ModelInput,
) -> torch.Tensor:
    # Run a single training step of the global model.
    opt.zero_grad()
    model.train(True)
    loss, _ = model(input)
    loss.backward()
    opt.step()

    # Run a forward pass of the global model.
    with torch.no_grad():
        model.train(False)
        full_pred = model(input)
        return full_pred


@skip_if_asan_class
class ModelParallelTest(ModelParallelTestShared):
    # @unittest.skipIf(
    #     torch.cuda.device_count() <= 1,
    #     "Not enough GPUs, this test requires at least two GPUs",
    # )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                # SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                # EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_sharding_nccl_rw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        print(f"test_sharding_nccl_rw: {sharder_type}, {sharding_type}, {kernel_type}")
        self._test_sharding(
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(sharder_type, sharding_type, kernel_type),
                ),
            ],
            backend="nccl",
            # backend="gloo",
            use_dataloader=True,
            num_workers=2,
        )
