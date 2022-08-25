import itertools
import os

import unittest
from collections import defaultdict, OrderedDict

from dataclasses import dataclass
from typing import cast, Dict, List, Optional, Tuple, Type, Union

import fbgemm_gpu  # nopycln: import

import hypothesis.strategies as st
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchrec.distributed as trec_dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, Verbosity

from models.simple_models import TestSparseNN
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

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
from torchrec.inference.modules import quantize_embeddings
from torchrec.models.dlrm import DLRM
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


def create_default_model_config():
    @dataclass
    class DLRMModelConfig:
        dense_arch_layer_sizes: List[int]
        dense_in_features: int
        embedding_dim: int
        id_list_features_keys: List[str]
        num_embeddings_per_feature: List[int]
        over_arch_layer_sizes: List[int]

    return DLRMModelConfig(
        dense_arch_layer_sizes=[32, 16, 8],
        dense_in_features=len(DEFAULT_INT_NAMES),
        embedding_dim=8,
        id_list_features_keys=DEFAULT_CAT_NAMES,
        num_embeddings_per_feature=len(DEFAULT_CAT_NAMES)
        * [
            3,
        ],
        over_arch_layer_sizes=[32, 32, 16, 1],
    )


class DLRMFactory(type):
    def __new__(cls, model_config=None):

        # If we do not provide a model config we use the default one compatible with the Criteo dataset
        if not model_config:
            model_config = create_default_model_config()

        default_cuda_rank = 0
        device = torch.device("cuda", default_cuda_rank)
        torch.cuda.set_device(device)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=model_config.embedding_dim,
                num_embeddings=model_config.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                model_config.id_list_features_keys
            )
        ]
        # Creates an EmbeddingBagCollection without allocating any memory
        ebc = EmbeddingBagCollection(tables=eb_configs, device=device)

        module = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=model_config.dense_in_features,
            dense_arch_layer_sizes=model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=model_config.over_arch_layer_sizes,
            dense_device=device,
        )

        module = quantize_embeddings(module, dtype=torch.qint8, inplace=True)

        return module


class TestSaveLoad(unittest.TestCase):
    def test_save_load_quant_simple(self):
        model = TestSparseNN(
            tables=[
                EmbeddingBagConfig(
                    num_embeddings=100,
                    embedding_dim=40,
                    name="table_1",
                    feature_names=["feature_1"],
                )
            ]
        )
        model = quantize_embeddings(model, dtype=torch.qint8, inplace=True)

        state_dict = model.state_dict()
        model.load_state_dict(state_dict)

    def test_save_load_quant_dlrm(self):
        model = DLRMFactory()
        state_dict = model.state_dict()
        model.load_state_dict(state_dict)
