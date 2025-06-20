import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# 延迟导入标志
_uno_imported = False
_import_error = None

logger = logging.getLogger(__name__)


def _lazy_import_uno():
    """延迟导入 UNO 模块，避免与其他库冲突"""
    global _uno_imported, _import_error
    
    if _uno_imported:
        return True
        
    try:
        # 在这里导入所有 UNO 相关的模块
        global uno, PropertyValue, NoConnectException
        import uno
        from com.sun.star.beans import PropertyValue
        from com.sun.star.connection import NoConnectException
        _uno_imported = True
        return True
    except ImportError as e:
        _import_error = e
        return False


# 检查 UNO 是否可用（但不立即导入）
def check_uno_available():
    """检查 UNO 是否可用（不会真正导入）"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("uno")
        return spec is not None
    except:
        return False


HAS_UNO = check_uno_available()


class UnoManager:
    """
    UNO管理器，用于管理LibreOffice服务实例和文档转换
    支持高并发真并行处理
    """
    
    def __init__(self, host: str = "localhost", port: int = 2002, timeout: int = 30):
        """
        初始化UNO管理器
        
        Args:
            host: LibreOffice服务主机地址
            port: LibreOffice服务端口
            timeout: 连接超时时间（秒）
        """
        # 延迟导入 UNO
        if not _lazy_import_uno():
            raise ImportError(
                f"python-uno未安装或无法导入。错误: {_import_error}\n"
                "请安装LibreOffice并确保python-uno可用。\n"
                "Ubuntu/Debian: apt-get install libreoffice python3-uno\n"
                "其他系统请参考: https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
            )
            
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connection_string = f"socket,host={host},port={port};urp;StarOffice.ComponentContext"
        self._lock = threading.Lock()
        self._desktop = None
        self._ctx = None
        self._soffice_process = None
        logger.info(f"🚀 UnoManager初始化 - 主机: {host}, 端口: {port}")
    
    def _start_soffice_service(self):
        """启动LibreOffice服务"""
        logger.info(f"🌟 启动LibreOffice服务，监听端口 {self.port}...")
        
        # 检查是否已有服务在运行
        if self._check_soffice_running():
            logger.info("✅ LibreOffice服务已在运行")
            return
            
        # 启动新的服务实例
        cmd = [
            "soffice",
            "--headless",
            "--invisible",
            "--nocrashreport",
            "--nodefault",
            "--nofirststartwizard",
            "--nologo",
            "--norestore",
            f"--accept={self.connection_string}"
        ]
        
        try:
            self._soffice_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"⏳ 等待LibreOffice服务启动...")
            time.sleep(5)  # 给服务一些启动时间
            
            if self._check_soffice_running():
                logger.info("✅ LibreOffice服务启动成功")
            else:
                raise Exception("LibreOffice服务启动失败")
                
        except Exception as e:
            logger.error(f"❌ 启动LibreOffice服务失败: {str(e)}")
            raise
    
    def _check_soffice_running(self) -> bool:
        """检查LibreOffice服务是否在运行"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except:
            return False
    
    def connect(self):
        """连接到LibreOffice服务"""
        with self._lock:
            if self._desktop is not None:
                return  # 已连接
                
            self._start_soffice_service()
            
            logger.info(f"🔌 连接到LibreOffice服务...")
            start_time = time.time()
            
            while time.time() - start_time < self.timeout:
                try:
                    # 获取组件上下文
                    local_ctx = uno.getComponentContext()
                    resolver = local_ctx.ServiceManager.createInstanceWithContext(
                        "com.sun.star.bridge.UnoUrlResolver", local_ctx
                    )
                    
                    # 连接到LibreOffice
                    self._ctx = resolver.resolve(f"uno:{self.connection_string}")
                    self._desktop = self._ctx.ServiceManager.createInstanceWithContext(
                        "com.sun.star.frame.Desktop", self._ctx
                    )
                    
                    logger.info("✅ 成功连接到LibreOffice服务")
                    return
                    
                except NoConnectException:
                    logger.debug("⏳ 等待LibreOffice服务就绪...")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"❌ 连接失败: {str(e)}")
                    time.sleep(1)
            
            raise TimeoutError(f"连接LibreOffice服务超时（{self.timeout}秒）")
    
    def disconnect(self):
        """断开与LibreOffice服务的连接"""
        with self._lock:
            if self._desktop is not None:
                try:
                    self._desktop.terminate()
                except:
                    pass
                self._desktop = None
                self._ctx = None
                logger.info("🔌 已断开LibreOffice服务连接")
    
    def stop_service(self):
        """停止LibreOffice服务"""
        self.disconnect()
        if self._soffice_process:
            try:
                self._soffice_process.terminate()
                self._soffice_process.wait(timeout=10)
            except:
                self._soffice_process.kill()
            self._soffice_process = None
            logger.info("🛑 LibreOffice服务已停止")
    
    @contextmanager
    def get_document(self, file_path: str):
        """
        获取文档对象的上下文管理器
        
        Args:
            file_path: 文档路径
            
        Yields:
            文档对象
        """
        self.connect()
        
        # 将路径转换为URL格式
        file_url = uno.systemPathToFileUrl(os.path.abspath(file_path))
        
        # 打开文档
        properties = []
        properties.append(self._make_property("Hidden", True))
        properties.append(self._make_property("ReadOnly", True))
        
        document = None
        try:
            document = self._desktop.loadComponentFromURL(
                file_url, "_blank", 0, properties
            )
            logger.debug(f"📄 打开文档: {file_path}")
            yield document
        finally:
            if document:
                try:
                    document.dispose()
                    logger.debug(f"📄 关闭文档: {file_path}")
                except:
                    pass
    
    def convert_document(self, input_path: str, output_path: str, 
                        output_format: str, filter_name: Optional[str] = None):
        """
        转换文档格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            output_format: 输出格式（如'txt', 'pdf', 'docx'等）
            filter_name: 过滤器名称（可选）
        """
        logger.info(f"🔄 开始转换文档: {input_path} -> {output_path} ({output_format})")
        
        with self.get_document(input_path) as document:
            if document is None:
                raise Exception(f"无法打开文档: {input_path}")
            
            # 准备输出属性
            properties = []
            
            # 设置过滤器
            if filter_name:
                properties.append(self._make_property("FilterName", filter_name))
            else:
                # 根据格式自动选择过滤器
                filter_map = {
                    'txt': 'Text',
                    'pdf': 'writer_pdf_Export',
                    'docx': 'MS Word 2007 XML',
                    'pptx': 'Impress MS PowerPoint 2007 XML',
                    'xlsx': 'Calc MS Excel 2007 XML',
                }
                if output_format in filter_map:
                    properties.append(
                        self._make_property("FilterName", filter_map[output_format])
                    )
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 转换为URL格式
            output_url = uno.systemPathToFileUrl(os.path.abspath(output_path))
            
            # 执行转换
            document.storeToURL(output_url, properties)
            logger.info(f"✅ 文档转换成功: {output_path}")
    
    def _make_property(self, name: str, value):
        """创建属性对象"""
        prop = PropertyValue()
        prop.Name = name
        prop.Value = value
        return prop


# UNO管理器连接池
_uno_managers = {}
_managers_lock = threading.Lock()
_base_port = 2002
_max_managers = 16


class UnoManagerPool:
    """UNO管理器连接池，支持真正的并行处理"""
    
    def __init__(self, max_managers: int = 8):
        self.max_managers = max_managers
        self.managers = {}
        self.available_ports = list(range(_base_port, _base_port + max_managers))
        self.lock = threading.Lock()
        logger.info(f"🏊 UnoManagerPool初始化 - 最大管理器数: {max_managers}")
    
    def get_manager(self) -> UnoManager:
        """获取可用的UNO管理器"""
        thread_id = threading.current_thread().ident
        
        with self.lock:
            # 如果当前线程已有管理器，直接返回
            if thread_id in self.managers:
                return self.managers[thread_id]
            
            # 创建新的管理器
            if len(self.managers) < self.max_managers and self.available_ports:
                port = self.available_ports.pop(0)
                try:
                    manager = UnoManager(port=port)
                    manager.connect()
                    self.managers[thread_id] = manager
                    logger.info(f"🎯 为线程{thread_id}创建UnoManager (端口: {port})")
                    return manager
                except Exception as e:
                    # 如果创建失败，释放端口
                    self.available_ports.append(port)
                    logger.error(f"❌ 创建UnoManager失败: {str(e)}")
                    raise
            
            # 如果无法创建新管理器，复用现有的
            if self.managers:
                available_manager = next(iter(self.managers.values()))
                logger.warning(f"⚠️ 线程{thread_id}复用现有UnoManager")
                return available_manager
            
            raise Exception("无法获取UNO管理器")
    
    def release_manager(self, thread_id: int = None):
        """释放线程的UNO管理器"""
        if thread_id is None:
            thread_id = threading.current_thread().ident
            
        with self.lock:
            if thread_id in self.managers:
                manager = self.managers[thread_id]
                try:
                    manager.stop_service()
                    self.available_ports.append(manager.port)
                    del self.managers[thread_id]
                    logger.info(f"🔄 释放线程{thread_id}的UnoManager")
                except Exception as e:
                    logger.error(f"❌ 释放UnoManager失败: {str(e)}")
    
    def cleanup_all(self):
        """清理所有管理器"""
        with self.lock:
            for thread_id, manager in list(self.managers.items()):
                try:
                    manager.stop_service()
                except:
                    pass
            self.managers.clear()
            self.available_ports = list(range(_base_port, _base_port + self.max_managers))
            logger.info("🧹 清理所有UnoManager")


# 全局连接池
_uno_pool = None
_pool_lock = threading.Lock()


def get_uno_manager() -> UnoManager:
    """获取UNO管理器（从连接池）"""
    global _uno_pool
    
    with _pool_lock:
        if _uno_pool is None:
            _uno_pool = UnoManagerPool(max_managers=8)
    
    return _uno_pool.get_manager()


def cleanup_uno_managers():
    """清理所有UNO管理器"""
    global _uno_pool
    
    with _pool_lock:
        if _uno_pool is not None:
            _uno_pool.cleanup_all()


def convert_with_uno(input_path: str, output_format: str, 
                    output_dir: Optional[str] = None) -> str:
    """
    使用UNO转换文档格式（便捷函数）
    
    Args:
        input_path: 输入文件路径
        output_format: 输出格式
        output_dir: 输出目录（可选，默认为输入文件所在目录）
        
    Returns:
        输出文件路径
    """
    input_path = Path(input_path)
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        
    output_path = output_dir / f"{input_path.stem}.{output_format}"
    
    manager = get_uno_manager()
    manager.convert_document(str(input_path), str(output_path), output_format)
    
    return str(output_path) 