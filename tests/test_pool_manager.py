"""Tests for machine-wide pipeline resource sizing."""

import os
import unittest
from unittest import mock

from modules.pool_manager import PoolConfig


class PoolConfigTest(unittest.TestCase):
    def test_low_available_memory_limits_global_build_pool(self):
        gib = 1024 ** 3
        with mock.patch("modules.pool_manager.cpu_count", return_value=16), \
             mock.patch("modules.pool_manager._windows_memory", return_value=(16 * gib, 2 * gib)), \
             mock.patch.dict(os.environ, {"ASMR_ACTIVE_PIPELINES": "1"}):
            config = PoolConfig(tts_backend="minimax-tts")
        self.assertEqual(config.split_workers, 1)
        self.assertEqual(config.build_workers, 2)
        self.assertEqual(config.tts_concurrency, 8)
        self.assertEqual(config.chunk_workers, 1)

    def test_second_task_receives_divided_auto_budget(self):
        gib = 1024 ** 3
        with mock.patch("modules.pool_manager.cpu_count", return_value=16), \
             mock.patch("modules.pool_manager._windows_memory", return_value=(32 * gib, 16 * gib)), \
             mock.patch.dict(os.environ, {"ASMR_ACTIVE_PIPELINES": "2"}):
            config = PoolConfig(tts_backend="fish-audio")
        self.assertEqual(config.build_workers, 4)
        self.assertEqual(config.tts_concurrency, 3)

    def test_indextts_is_always_serial(self):
        with mock.patch("modules.pool_manager.cpu_count", return_value=16), \
             mock.patch("modules.pool_manager._windows_memory", return_value=(None, None)):
            config = PoolConfig(tts_backend="indextts-2.5", tts_concurrency=8)
        self.assertEqual(config.tts_concurrency, 1)


if __name__ == "__main__":
    unittest.main()
