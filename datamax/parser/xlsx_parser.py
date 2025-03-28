import multiprocessing
import time
from multiprocessing import Queue
from datamax.parser.base import MarkdownOutputVo
from datamax.parser.base import BaseLife
from openpyxl import load_workbook
import warnings
from markitdown import MarkItDown

warnings.filterwarnings("ignore")

class XlsxParser(BaseLife):
    # 实现单例 避免重复的初始化

    _markitdown_instance = None

    @classmethod
    def get_markitdown(cls):
        if cls._markitdown_instance is None:
            cls._markitdown_instance = MarkItDown()  # 第一次调用时初始化
        return cls._markitdown_instance

    def __init__(self, file_path, timeout):
        super().__init__()
        self.file_path = file_path
        self.timeout = timeout
        self.markitdown = self.get_markitdown()

    def _parse(self, file_path: str, result_queue: Queue) -> dict:
        # 将原始解析逻辑封装到这个方法中
        try:
            wb = load_workbook(
                filename=file_path,
                data_only=True,
                read_only=True
            )
            wb.close()
        except Exception as e:
            raise e

        mk_content = self.markitdown.convert(file_path).text_content
        lifecycle = self.generate_lifecycle(
            source_file=file_path,
            domain="Technology",
            usage_purpose="Documentation",
            life_type="LLM_ORIGIN"
        )
        output_vo = MarkdownOutputVo(self.get_file_extension(file_path), mk_content)
        output_vo.add_lifecycle(lifecycle)
        result_queue.put(output_vo.to_dict())
        time.sleep(0.5)
        return output_vo.to_dict()

    def parse(self, file_path: str) -> dict:
        import time
        result_queue = Queue()
        process = multiprocessing.Process(target=self._parse, args=(file_path, result_queue))
        process.start()
        # 记录当前任务开始的时间
        start_time = time.time()

        # 长度位timeout的定时器, 定义死循环每隔一秒判断是否有结果
        while time.time() - start_time < self.timeout:
            print(f"plz waiting...: {int(time.time() - start_time)}")
            if not process.is_alive():
                break
            if not result_queue.empty():
                # 有结果直接返回
                return result_queue.get()
            time.sleep(1)
        else:
            # 到了timeout没有结果进行进程杀死
            process.terminate()
            process.join()