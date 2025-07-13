import unittest
import threading
import time
from datamax.parser.core import DataMax


class TestDataMaxInit(unittest.TestCase):
    def test_ttl_validation(self):
        """测试TTL参数验证"""
        with self.assertRaises(ValueError):
            DataMax(ttl=-1)

        # 测试有效值
        self.assertIsInstance(DataMax(ttl=0), DataMax)
        self.assertIsInstance(DataMax(ttl=3600), DataMax)

    def test_cache_cleaner_thread(self):
        """测试缓存清理线程启动"""
        dm = DataMax(ttl=10)
        thread_names = [t.name for t in threading.enumerate()]
        self.assertTrue(any("cache_cleaner" in name for name in thread_names))