#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from collections import defaultdict, OrderedDict
from typing import cast, Dict, List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchrec.distributed as trec_dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, Verbosity
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
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec.test_utils import get_free_port, skip_if_asan_class


@skip_if_asan_class
class ModelParallelTest(ModelParallelTestShared):
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                # TODO: enable it with correct semantics, see T104397332
                # SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    # target
    def test_sharding_gloo_cw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        # world_size = 4
        world_size = 3  # test world size 3 instead of 4
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                ),
            ],
            backend="gloo",
            world_size=world_size,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
        )
