#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile
import unittest
import uuid

from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec import test_utils
from torchrec.datasets.test_utils.criteo_test_utils import CriteoTest

from ..dlrm_main import main


class MainTorchxTest(unittest.TestCase):
    @test_utils.skip_if_asan
    def test_main_function_torchx(self) -> None:
        p = subprocess.run(
            [
                "torchx",
                "run",
                "-s",
                "local_cwd",
                "dist.ddp",
                "-j",
                "1x2",
                "--script",
                "eagle_plus_tests/dlrm/dlrm_main.py",
            ],
            check=True,
            cwd="/mnt/torchrec-private",
        )
        print("return code:", p)
