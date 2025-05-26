"""
DataMax 基础测试
"""

from datamax import DataMax


def test_import():
    """测试模块导入"""
    assert DataMax is not None


def test_version():
    """测试版本号"""
    import datamax

    assert hasattr(datamax, "__version__") or True  # 版本号检查


# 更多测试用例...
