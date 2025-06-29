import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from loguru import logger

# delayed import of lock and flag
_uno_imported = False
_import_error = None
_import_lock = threading.Lock()


def _lazy_import_uno():
    """延迟导入 UNO 模块，避免与其他库冲突（线程安全）"""
    global _uno_imported, _import_error

    # quick check,avoiding unnacessary acquisition of lock
    if _uno_imported:
        return True

    with _import_lock:
        # double check lock mode
        if _uno_imported:
            return True

        try:
            # import module relate to UNO
            global uno, PropertyValue, NoConnectException
            import uno
            from com.sun.star.beans import PropertyValue
            from com.sun.star.connection import NoConnectException

            _uno_imported = True
            logger.info("✅ UNO模块导入成功")
            return True
        except ImportError as e:
            _import_error = e
            logger.error(f"❌ UNO模块导入失败: {str(e)}")
            return False


def ensure_uno_imported():
    """确保UNO已导入，适用于需要提前导入的场景"""
    if not _lazy_import_uno():
        raise ImportError(
            f"python-uno未安装或无法导入。错误: {_import_error}\n"
            "请安装LibreOffice并确保python-uno可用。\n"
            "Ubuntu/Debian: apt-get install libreoffice python3-uno\n"
            "其他系统请参考: https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
        )


# check if uno is available(not importing immediately）
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
    单线程版本，适合稳定高效的文档处理
    """

    def __init__(self, host: str = "localhost", port: int = 2002, timeout: int = 30):
        """
        初始化UNO管理器

        Args:
            host: LibreOffice服务主机地址
            port: LibreOffice服务端口
            timeout: 连接超时时间（秒）
        """
        # Ensure that UNO has been imported (in a thread-safe manner)
        ensure_uno_imported()

        self.host = host
        self.port = port
        self.timeout = timeout
        self.connection_string = (
            f"socket,host={host},port={port};urp;StarOffice.ComponentContext"
        )
        self._lock = threading.Lock()
        self._desktop = None
        self._ctx = None
        self._soffice_process = None
        self._connected = False
        logger.info(f"🚀 UnoManager初始化 - 主机: {host}, 端口: {port} (单线程模式)")

    def _start_soffice_service(self):
        """启动LibreOffice服务"""
        logger.info(f"🌟 启动LibreOffice服务，监听端口 {self.port}...")

        # check if soffice running
        if self._check_soffice_running():
            logger.info("✅ LibreOffice服务已在运行")
            return

        # new a soffice
        cmd = [
            "soffice",
            "--headless",
            "--invisible",
            "--nocrashreport",
            "--nodefault",
            "--nofirststartwizard",
            "--nologo",
            "--norestore",
            f"--accept={self.connection_string}",
        ]

        try:
            self._soffice_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"⏳ 等待LibreOffice服务启动...")

            # Intelligent waiting: Polling to check service status, providing flexible time for machines of different performance levels.
            start_time = time.time()
            check_interval = 1  # checking every sec
            max_wait_time = 30  # wait for max 30 sec

            while time.time() - start_time < max_wait_time:
                if self._check_soffice_running():
                    elapsed = time.time() - start_time
                    logger.info(f"✅ LibreOffice服务启动成功 (耗时 {elapsed:.1f}秒)")
                    return

                logger.debug(
                    f"🔄 服务未就绪，继续等待... (已等待 {time.time() - start_time:.1f}秒)"
                )
                time.sleep(check_interval)

            # overtime
            raise Exception(f"LibreOffice服务启动超时 (等待了{max_wait_time}秒)")

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

    def is_connected(self) -> bool:
        """检查是否已连接"""
        with self._lock:
            return self._connected and self._desktop is not None

    def connect(self):
        """连接到LibreOffice服务"""
        with self._lock:
            if self._connected and self._desktop is not None:
                return  # connected

            self._start_soffice_service()

            logger.info(f"🔌 连接到LibreOffice服务...")
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                try:
                    # get context
                    local_ctx = uno.getComponentContext()
                    resolver = local_ctx.ServiceManager.createInstanceWithContext(
                        "com.sun.star.bridge.UnoUrlResolver", local_ctx
                    )

                    # connect to LibreOffice
                    self._ctx = resolver.resolve(f"uno:{self.connection_string}")
                    self._desktop = self._ctx.ServiceManager.createInstanceWithContext(
                        "com.sun.star.frame.Desktop", self._ctx
                    )

                    self._connected = True
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
                self._connected = False
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

        # converse path to URL
        file_url = uno.systemPathToFileUrl(os.path.abspath(file_path))

        # open file
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

    def convert_document(
        self,
        input_path: str,
        output_path: str,
        output_format: str,
        filter_name: Optional[str] = None,
    ):
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

            # prepare to output properties
            properties = []

            # set filter
            if filter_name:
                properties.append(self._make_property("FilterName", filter_name))
            else:
                # choose filter by format
                if output_format == "txt":
                    # multi-filter for multi-files
                    filter_options = [
                        ("Text (encoded)", "UTF8"),
                        ("Text", None),
                        ("HTML (StarWriter)", None),
                    ]

                    success = False
                    for filter_name, filter_option in filter_options:
                        try:
                            properties = []
                            properties.append(
                                self._make_property("FilterName", filter_name)
                            )
                            if filter_option:
                                properties.append(
                                    self._make_property("FilterOptions", filter_option)
                                )

                            # ensuring that the output directory exists.
                            output_dir = os.path.dirname(output_path)
                            if output_dir and not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            # converse to URL
                            output_url = uno.systemPathToFileUrl(
                                os.path.abspath(output_path)
                            )

                            # conversing
                            document.storeToURL(output_url, properties)
                            logger.info(
                                f"✅ 文档转换成功 (使用过滤器: {filter_name}): {output_path}"
                            )
                            success = True
                            break
                        except Exception as e:
                            logger.debug(f"🔄 过滤器 {filter_name} 失败: {str(e)}")
                            continue

                    if not success:
                        raise Exception(
                            f"所有文本过滤器都失败，无法转换文档: {input_path}"
                        )

                    return  # converted,return
                else:
                    # Other formats use the default filter
                    filter_map = {
                        "pdf": "writer_pdf_Export",
                        "docx": "MS Word 2007 XML",
                        "pptx": "Impress MS PowerPoint 2007 XML",
                        "xlsx": "Calc MS Excel 2007 XML",
                    }
                    if output_format in filter_map:
                        properties.append(
                            self._make_property("FilterName", filter_map[output_format])
                        )

            # ensuring that the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # converse to URL
            output_url = uno.systemPathToFileUrl(os.path.abspath(output_path))

            # conversing
            document.storeToURL(output_url, properties)
            logger.info(f"✅ 文档转换成功: {output_path}")

    def _make_property(self, name: str, value):
        """创建属性对象"""
        prop = PropertyValue()
        prop.Name = name
        prop.Value = value
        return prop


# global Singleton UnoManager
_global_uno_manager: Optional[UnoManager] = None
_manager_lock = threading.Lock()


def get_uno_manager() -> UnoManager:
    """获取全局单例UNO管理器"""
    global _global_uno_manager

    if _global_uno_manager is None:
        with _manager_lock:
            if _global_uno_manager is None:
                _global_uno_manager = UnoManager()
                logger.info("🎯 创建全局单例UnoManager (单线程模式)")

    return _global_uno_manager


def cleanup_uno_manager():
    """清理全局UNO管理器"""
    global _global_uno_manager

    with _manager_lock:
        if _global_uno_manager is not None:
            try:
                _global_uno_manager.stop_service()
            except:
                pass
            _global_uno_manager = None
            logger.info("🧹 清理全局UnoManager")


@contextmanager
def uno_manager_context():
    """UNO管理器上下文管理器，自动获取和管理"""
    manager = get_uno_manager()
    try:
        yield manager
    finally:
        # Maintain connections to improve efficiency in single-threaded mode
        pass


def convert_with_uno(
    input_path: str, output_format: str, output_dir: Optional[str] = None
) -> str:
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

    with uno_manager_context() as manager:
        manager.convert_document(str(input_path), str(output_path), output_format)

    return str(output_path)
