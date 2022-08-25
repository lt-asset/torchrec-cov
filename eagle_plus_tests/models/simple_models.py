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

from torchrec.datasets.random import RandomRecDataset
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


class TestSparseArch(nn.Module):
    """
    Basic nn.Module for testing

    Args:
        tables
        device

    Call Args:
        features

    Returns:
        KeyedTensor
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
            tables=tables,
            device=device,
        )

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        return self.ebc(features)


class TestSparseNN(TestSparseNNBase):
    """
    Simple version of a SparseNN model.

    Args:
        tables: List[EmbeddingBagConfig],
        weighted_tables: Optional[List[EmbeddingBagConfig]],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],

    Call Args:
        input: ModelInput,

    Returns:
        torch.Tensor

    Example::

        TestSparseNN()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        num_float_features: int = 10,
        weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
        if weighted_tables is None:
            weighted_tables = []

        self.dense = TestDenseArch(num_float_features, dense_device)
        self.sparse = TestSparseArch(tables, sparse_device)
        self.over = TestOverArch(tables, weighted_tables, dense_device)

    def forward(
        self,
        input: Batch,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.dense_features)
        sparse_r = self.sparse(input.sparse_features)
        over_r = self.over(dense_r, sparse_r)
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, (input.labels).to(torch.float)
                ),
                pred,
            )
        else:
            return pred
