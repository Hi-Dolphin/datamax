import sys
import pathlib
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datamax.parser.docx_parser import DocxParser
from datamax.parser.base import MarkdownOutputVo, LifeCycle


class TestDocxParser:
    """DOCX解析器测试类"""

    def test_docx_parser_initialization(self):
        """测试DocxParser初始化"""
        file_path = "test.docx"
        parser = DocxParser(file_path=file_path, to_markdown=False)
        assert parser.file_path == file_path
        assert parser.to_markdown == False

        # 测试markdown模式
        parser_md = DocxParser(file_path=file_path, to_markdown=True)
        assert parser_md.to_markdown == True

    @patch('datamax.parser.docx_parser.subprocess.Popen')
    @patch('datamax.parser.docx_parser.os.path.exists')
    def test_docx_to_txt_success(self, mock_exists, mock_popen):
        """测试DOCX到TXT转换成功的情况"""
        # 设置模拟
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"conversion complete", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_exists.return_value = True

        parser = DocxParser(file_path="test.docx")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            docx_path = os.path.join(temp_dir, "test.docx")
            result = parser.docx_to_txt(docx_path, temp_dir)
            expected_path = os.path.join(temp_dir, "test.txt")
            assert result == expected_path

    @patch('datamax.parser.docx_parser.subprocess.Popen')
    def test_docx_to_txt_failure(self, mock_popen):
        """测试DOCX到TXT转换失败的情况"""
        # 设置模拟失败的转换
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"conversion failed")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        parser = DocxParser(file_path="test.docx")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            docx_path = os.path.join(temp_dir, "test.docx")
            with pytest.raises(Exception, match="Error Output"):
                parser.docx_to_txt(docx_path, temp_dir)

    @patch('builtins.open', new_callable=mock_open, read_data="测试DOCX文档内容\n第二行内容")
    @patch('datamax.parser.docx_parser.chardet.detect')
    def test_read_txt_file_success(self, mock_detect, mock_file):
        """测试读取TXT文件成功"""
        mock_detect.return_value = {"encoding": "utf-8"}
        
        parser = DocxParser(file_path="test.docx")
        content = parser.read_txt_file("test.txt")
        
        assert content == "测试DOCX文档内容\n第二行内容"
        mock_detect.assert_called_once()

    def test_read_txt_file_not_found(self):
        """测试读取不存在的TXT文件"""
        parser = DocxParser(file_path="test.docx")
        
        with pytest.raises(Exception, match="文件未找到"):
            parser.read_txt_file("nonexistent.txt")

    @patch('datamax.parser.docx_parser.shutil.copy')
    @patch('datamax.parser.docx_parser.tempfile.TemporaryDirectory')
    def test_read_docx_file_success(self, mock_temp_dir, mock_copy):
        """测试读取DOCX文件成功"""
        # 设置临时目录模拟
        temp_path = "/tmp/test"
        mock_temp_dir.return_value.__enter__.return_value = temp_path
        
        parser = DocxParser(file_path="test.docx")
        
        # 模拟docx_to_txt和read_txt_file方法
        with patch.object(parser, 'docx_to_txt', return_value="/tmp/test/tmp.txt") as mock_docx_to_txt, \
             patch.object(parser, 'read_txt_file', return_value="DOCX文档内容") as mock_read_txt:
            
            content = parser.read_docx_file("test.docx")
            
            assert content == "DOCX文档内容"
            mock_copy.assert_called_once()
            mock_docx_to_txt.assert_called_once()
            mock_read_txt.assert_called_once()

    def test_read_docx_file_not_found(self):
        """测试读取不存在的DOCX文件"""
        parser = DocxParser(file_path="test.docx")
        
        with pytest.raises(Exception, match="文件未找到"):
            parser.read_docx_file("nonexistent.docx")

    @patch('datamax.parser.docx_parser.os.path.exists')
    @patch('datamax.parser.docx_parser.os.path.getsize')
    def test_parse_file_not_exists(self, mock_getsize, mock_exists):
        """测试解析不存在的文件"""
        mock_exists.return_value = False
        
        parser = DocxParser(file_path="test.docx")
        
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            parser.parse("nonexistent.docx")

    @patch('datamax.parser.docx_parser.os.path.exists')
    @patch('datamax.parser.docx_parser.os.path.getsize')
    def test_parse_success_without_markdown(self, mock_getsize, mock_exists):
        """测试成功解析DOCX文件（不转换为markdown）"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        parser = DocxParser(file_path="test.docx", to_markdown=False)
        
        with patch.object(parser, 'read_docx_file', return_value="DOCX文档内容测试") as mock_read, \
             patch.object(parser, 'get_file_extension', return_value="docx") as mock_ext, \
             patch.object(parser, 'generate_lifecycle') as mock_lifecycle:
            
            # 设置lifecycle模拟
            mock_lifecycle_obj = MagicMock()
            mock_lifecycle_obj.to_dict.return_value = {
                "update_time": "2024-01-01 12:00:00",
                "life_type": ["LLM_ORIGIN"],
                "life_metadata": {"source_file": "test.docx"}
            }
            mock_lifecycle.return_value = mock_lifecycle_obj
            
            result = parser.parse("test.docx")
            
            assert result["extension"] == "docx"
            assert result["content"] == "DOCX文档内容测试"
            assert "lifecycle" in result
            mock_read.assert_called_once_with(docx_path="test.docx")
            mock_ext.assert_called_once_with("test.docx")

    @patch('datamax.parser.docx_parser.os.path.exists')
    @patch('datamax.parser.docx_parser.os.path.getsize')
    def test_parse_success_with_markdown(self, mock_getsize, mock_exists):
        """测试成功解析DOCX文件（转换为markdown）"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        parser = DocxParser(file_path="test.docx", to_markdown=True)
        
        with patch.object(parser, 'read_docx_file', return_value="DOCX标题\n内容段落") as mock_read, \
             patch.object(parser, 'get_file_extension', return_value="docx") as mock_ext, \
             patch.object(parser, 'generate_lifecycle') as mock_lifecycle:
            
            # 设置lifecycle模拟
            mock_lifecycle_obj = MagicMock()
            mock_lifecycle_obj.to_dict.return_value = {
                "update_time": "2024-01-01 12:00:00",
                "life_type": ["LLM_ORIGIN"],
                "life_metadata": {"source_file": "test.docx"}
            }
            mock_lifecycle.return_value = mock_lifecycle_obj
            
            result = parser.parse("test.docx")
            
            assert result["extension"] == "docx"
            assert result["content"] == "DOCX标题\n内容段落"  # format_as_markdown在这种情况下保持原样
            assert "lifecycle" in result

    @patch('datamax.parser.docx_parser.os.path.exists')
    @patch('datamax.parser.docx_parser.os.path.getsize')
    def test_parse_with_file_extension_validation(self, mock_getsize, mock_exists):
        """测试解析时的文件扩展名验证"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        parser = DocxParser(file_path="test.txt", to_markdown=False)  # 使用错误的扩展名
        
        with patch.object(parser, 'read_docx_file', return_value="内容") as mock_read, \
             patch.object(parser, 'get_file_extension', return_value="txt") as mock_ext, \
             patch.object(parser, 'generate_lifecycle') as mock_lifecycle:
            
            # 设置lifecycle模拟
            mock_lifecycle_obj = MagicMock()
            mock_lifecycle_obj.to_dict.return_value = {
                "update_time": "2024-01-01 12:00:00",
                "life_type": ["LLM_ORIGIN"],
                "life_metadata": {"source_file": "test.txt"}
            }
            mock_lifecycle.return_value = mock_lifecycle_obj
            
            # 应该仍然能够解析，只是会有警告
            result = parser.parse("test.txt")
            assert result["extension"] == "txt"

    @patch('datamax.parser.docx_parser.os.path.exists')
    @patch('datamax.parser.docx_parser.os.path.getsize')
    def test_parse_empty_file(self, mock_getsize, mock_exists):
        """测试解析空文件"""
        mock_exists.return_value = True
        mock_getsize.return_value = 0  # 空文件
        
        parser = DocxParser(file_path="empty.docx", to_markdown=False)
        
        with patch.object(parser, 'read_docx_file', return_value="") as mock_read, \
             patch.object(parser, 'get_file_extension', return_value="docx") as mock_ext, \
             patch.object(parser, 'generate_lifecycle') as mock_lifecycle:
            
            # 设置lifecycle模拟
            mock_lifecycle_obj = MagicMock()
            mock_lifecycle_obj.to_dict.return_value = {
                "update_time": "2024-01-01 12:00:00",
                "life_type": ["LLM_ORIGIN"],
                "life_metadata": {"source_file": "empty.docx"}
            }
            mock_lifecycle.return_value = mock_lifecycle_obj
            
            result = parser.parse("empty.docx")
            assert result["extension"] == "docx"
            assert result["content"] == ""

    def test_format_as_markdown(self):
        """测试markdown格式化功能"""
        parser = DocxParser(file_path="test.docx")
        
        # 测试空内容
        assert parser.format_as_markdown("") == ""
        assert parser.format_as_markdown("   ") == "   "
        
        # 测试普通内容
        content = "DOCX标题1\n\n内容段落1\n内容段落2\n\n标题2\n内容段落3"
        result = parser.format_as_markdown(content)
        expected = "DOCX标题1\n\n内容段落1\n内容段落2\n\n标题2\n内容段落3"
        assert result == expected

    def test_integration_with_temp_file(self, tmp_path):
        """集成测试：使用临时文件测试基本功能（不依赖soffice）"""
        # 创建临时DOCX文件（实际上是txt文件，用于测试）
        docx_file = tmp_path / "test.docx"
        docx_file.write_text("这是一个测试DOCX文档\n包含多行内容", encoding="utf-8")
        
        parser = DocxParser(file_path=str(docx_file), to_markdown=False)
        
        # 模拟soffice转换过程
        with patch.object(parser, 'read_docx_file', return_value="这是一个测试DOCX文档\n包含多行内容"):
            result = parser.parse(str(docx_file))
            
            assert result["extension"] == "docx"
            assert "这是一个测试DOCX文档" in result["content"]
            assert result["lifecycle"][0]["life_metadata"]["source_file"] == str(docx_file)

    def test_error_handling_in_parse(self):
        """测试parse方法的错误处理"""
        parser = DocxParser(file_path="test.docx")
        
        with patch('datamax.parser.docx_parser.os.path.exists', return_value=True), \
             patch('datamax.parser.docx_parser.os.path.getsize', return_value=1024), \
             patch.object(parser, 'read_docx_file', side_effect=Exception("读取文件时出错")):
            
            with pytest.raises(Exception):
                parser.parse("test.docx")

    def test_permission_error_handling(self):
        """测试权限错误处理"""
        parser = DocxParser(file_path="test.docx")
        
        with patch('datamax.parser.docx_parser.os.path.exists', return_value=True), \
             patch('datamax.parser.docx_parser.os.path.getsize', return_value=1024), \
             patch.object(parser, 'read_docx_file', side_effect=PermissionError("权限拒绝")):
            
            with pytest.raises(Exception, match="无权限访问文件"):
                parser.parse("test.docx")

    @patch('datamax.parser.docx_parser.subprocess.Popen')
    def test_subprocess_error_handling(self, mock_popen):
        """测试subprocess错误处理"""
        # 设置模拟subprocess错误
        mock_popen.side_effect = OSError("Command not found")
        
        parser = DocxParser(file_path="test.docx")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            docx_path = os.path.join(temp_dir, "test.docx")
            with pytest.raises(OSError, match="Command not found"):
                parser.docx_to_txt(docx_path, temp_dir)

    @patch('datamax.parser.docx_parser.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data="测试内容")
    def test_encoding_detection(self, mock_file, mock_detect):
        """测试文件编码检测"""
        # 测试检测不到编码的情况
        mock_detect.return_value = {"encoding": None}
        
        parser = DocxParser(file_path="test.docx")
        content = parser.read_txt_file("test.txt")
        
        # 应该默认使用utf-8编码
        assert content == "测试内容"
        mock_detect.assert_called_once() 