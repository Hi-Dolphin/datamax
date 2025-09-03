import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger


# Delayed import flags and lock
_uno_imported = False
_import_error = None
_import_lock = threading.Lock()


def _lazy_import_uno():
    """Lazy import UNO modules to avoid conflicts with other libraries (thread-safe)"""
    global _uno_imported, _import_error

    # Quick check without acquiring lock if already imported
    if _uno_imported:
        return True

    with _import_lock:
        # Double-check lock pattern
        if _uno_imported:
            return True

        try:
            # Import UNO-related modules
            global uno, PropertyValue, NoConnectException
            import uno
            from com.sun.star.beans import PropertyValue
            from com.sun.star.connection import NoConnectException

            _uno_imported = True
            logger.info("UNO modules imported successfully")
            return True
        except ImportError as e:
            _import_error = e
            logger.error(f"UNO modules import failed: {e}")
            return False


def ensure_uno_imported():
    """Ensure UNO is imported for scenarios requiring pre-import"""
    if not _lazy_import_uno():
        raise ImportError(
            f"python-uno is not installed or cannot be imported. Error: {_import_error}\n"
            "Please install LibreOffice and ensure python-uno is available.\n"
            "Ubuntu/Debian: apt-get install libreoffice python3-uno\n"
            "For other systems, refer to: https://wiki.documentfoundation.org/Documentation/DevGuide/Installing_the_SDK"
        )


# Check if uno is available (without actually importing)
def check_uno_available():
    """Check if UNO is available (without actually importing)"""
    try:
        import importlib.util

        spec = importlib.util.find_spec("uno")
        return spec is not None
    except Exception:
        return False


HAS_UNO = check_uno_available()


class UnoManager:
    """
    UNO manager for handling LibreOffice service instances and document conversion.
    Single-threaded version, suitable for stable and efficient document processing.
    """

    def __init__(self, host: str = "localhost", port: int = 2002, timeout: int = 30):
        """
        Initialize UNO manager.

        Args:
            host: LibreOffice service host address
            port: LibreOffice service port
            timeout: Connection timeout (seconds)
        """
        # Ensure UNO has been imported (in a thread-safe manner)
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
        logger.info(f"UnoManager initialized - Host: {host}, Port: {port} (single-threaded mode)")

    def _start_soffice_service(self):
        """Start LibreOffice service"""
        logger.info(f"Starting LibreOffice service on port {self.port}...")

        # Check if soffice is already running
        if self._check_soffice_running():
            logger.info("LibreOffice service is already running")
            return

        # Start soffice process
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
            logger.info("Waiting for LibreOffice service to start...")

            # Intelligent waiting: Poll service status for flexibility across different machine performance levels
            start_time = time.time()
            check_interval = 1  # check every second
            max_wait_time = 30  # maximum wait time in seconds

            while time.time() - start_time < max_wait_time:
                if self._check_soffice_running():
                    elapsed = time.time() - start_time
                    logger.info(f"LibreOffice service started successfully ({elapsed:.1f}s)")
                    return

                logger.debug(
                    f"Service not ready, waiting... ({time.time() - start_time:.1f}s elapsed)"
                )
                time.sleep(check_interval)

            # Timeout
            raise Exception(f"LibreOffice service startup timed out (waited {max_wait_time}s)")

        except Exception as e:
            logger.error(f"Failed to start LibreOffice service: {e}")
            raise

    def _check_soffice_running(self) -> bool:
        """Check if LibreOffice service is running"""
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def is_connected(self) -> bool:
        """Check if already connected"""
        with self._lock:
            return self._connected and self._desktop is not None

    def connect(self):
        """Connect to LibreOffice service"""
        with self._lock:
            if self._connected and self._desktop is not None:
                return  # already connected

            self._start_soffice_service()

            logger.info("Connecting to LibreOffice service...")
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                try:
                    # Get component context
                    local_ctx = uno.getComponentContext()
                    resolver = local_ctx.ServiceManager.createInstanceWithContext(
                        "com.sun.star.bridge.UnoUrlResolver", local_ctx
                    )

                    # Connect to LibreOffice
                    self._ctx = resolver.resolve(f"uno:{self.connection_string}")
                    self._desktop = self._ctx.ServiceManager.createInstanceWithContext(
                        "com.sun.star.frame.Desktop", self._ctx
                    )

                    self._connected = True
                    logger.info("Successfully connected to LibreOffice service")
                    return

                except NoConnectException:
                    logger.debug("Waiting for LibreOffice service to be ready...")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Connection failed: {e}")
                    time.sleep(1)

            raise TimeoutError(f"Connection to LibreOffice service timed out ({self.timeout}s)")

    def disconnect(self):
        """Disconnect from LibreOffice service"""
        with self._lock:
            if self._desktop is not None:
                try:
                    self._desktop.terminate()
                except Exception:
                    pass
                self._desktop = None
                self._ctx = None
                self._connected = False
                logger.info("Disconnected from LibreOffice service")

    def stop_service(self):
        """Stop LibreOffice service"""
        self.disconnect()
        if self._soffice_process:
            try:
                self._soffice_process.terminate()
                self._soffice_process.wait(timeout=10)
            except Exception:
                self._soffice_process.kill()
            self._soffice_process = None
            logger.info("LibreOffice service stopped")

    @contextmanager
    def get_document(self, file_path: str):
        """
        Context manager for document object.

        Args:
            file_path: Path to the document

        Yields:
            Document object
        """
        self.connect()

        # Convert path to URL
        file_url = uno.systemPathToFileUrl(os.path.abspath(file_path))

        # Open file
        properties = []
        properties.append(self._make_property("Hidden", True))
        properties.append(self._make_property("ReadOnly", True))

        document = None
        try:
            document = self._desktop.loadComponentFromURL(
                file_url, "_blank", 0, properties
            )
            logger.debug(f"Document opened: {file_path}")
            yield document
        finally:
            if document:
                try:
                    document.dispose()
                    logger.debug(f"Document closed: {file_path}")
                except Exception:
                    pass

    def convert_document(
        self,
        input_path: str,
        output_path: str,
        output_format: str,
        filter_name: str | None = None,
    ):
        """
        Convert document format.

        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Output format (e.g., 'txt', 'pdf', 'docx', etc.)
            filter_name: Filter name (optional)
        """
        logger.info(f"Converting document: {input_path} -> {output_path} ({output_format})")

        with self.get_document(input_path) as document:
            if document is None:
                raise Exception(f"Unable to open document: {input_path}")

            # Prepare output properties
            properties = []

            # Set filter
            if filter_name:
                properties.append(self._make_property("FilterName", filter_name))
            else:
                # Select filter by format
                if output_format == "txt":
                    # Multiple filters for different text formats
                    filter_options = [
                        ("Text (encoded)", "UTF8"),
                        ("Text", None),
                        ("HTML (StarWriter)", None),
                    ]

                    # Ensure output directory exists before trying filters
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)

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

                            # Convert to URL
                            output_url = uno.systemPathToFileUrl(
                                os.path.abspath(output_path)
                            )

                            # Convert document
                            document.storeToURL(output_url, properties)
                            logger.info(
                                f"Document conversion successful (using filter: {filter_name}): {output_path}"
                            )
                            success = True
                            break
                        except Exception as e:
                            logger.debug(f"Filter {filter_name} failed: {e}")
                            continue

                    if not success:
                        raise Exception(
                            f"All text filters failed, unable to convert document: {input_path}"
                        )

                    return  # Conversion complete
                else:
                    # Other formats use default filters
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

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Convert to URL
            output_url = uno.systemPathToFileUrl(os.path.abspath(output_path))

            # Convert document
            document.storeToURL(output_url, properties)
            logger.info(f"Document conversion successful: {output_path}")

    def _make_property(self, name: str, value):
        """Create a property object"""
        prop = PropertyValue()
        prop.Name = name
        prop.Value = value
        return prop


# Global singleton UnoManager
_global_uno_manager: UnoManager | None = None
_manager_lock = threading.Lock()


def get_uno_manager() -> UnoManager:
    """Get global singleton UNO manager"""
    global _global_uno_manager

    if _global_uno_manager is None:
        with _manager_lock:
            if _global_uno_manager is None:
                _global_uno_manager = UnoManager()
                logger.info("Global singleton UnoManager created (single-threaded mode)")

    return _global_uno_manager


def cleanup_uno_manager():
    """Clean up global UNO manager"""
    global _global_uno_manager

    with _manager_lock:
        if _global_uno_manager is not None:
            try:
                _global_uno_manager.stop_service()
            except Exception:
                pass
            _global_uno_manager = None
            logger.info("Global UnoManager cleaned up")


@contextmanager
def uno_manager_context():
    """UNO manager context manager, auto acquires and manages"""
    manager = get_uno_manager()
    try:
        yield manager
    finally:
        # Maintain connections to improve efficiency in single-threaded mode
        pass


def convert_with_uno(
    input_path: str, output_format: str, output_dir: str | None = None
) -> str:
    """
    Convert document format using UNO (convenience function).

    Args:
        input_path: Input file path
        output_format: Output format
        output_dir: Output directory (optional, defaults to input file directory)

    Returns:
        Output file path
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
