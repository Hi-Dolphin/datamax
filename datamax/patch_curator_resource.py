import sys
import types

# 伪造 resource 模块，绕过 Windows 不支持的问题
resource = types.SimpleNamespace()

def fake_getrlimit(name):
    # name 就是 RLIMIT_NOFILE，返回软硬限制，示例随意
    return (1024, 4096)

def fake_setrlimit(name, limits):
    return None

# 添加缺失的常量
resource.RLIMIT_NOFILE = 7

resource.getrlimit = fake_getrlimit
resource.setrlimit = fake_setrlimit

sys.modules['resource'] = resource

