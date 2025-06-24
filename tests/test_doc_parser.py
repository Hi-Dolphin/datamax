import sys
import pathlib
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datamax.parser.doc_parser import DocParser
from datamax.parser.base import MarkdownOutputVo, LifeCycle


class TestDocParser:
    """DOC解析器测试类"""

    def test_doc_parser_initialization(self):
        """测试DocParser初始化"""
        file_path = "test.doc"
        parser = DocParser(file_path=file_path, to_markdown=False)
        assert parser.file_path == file_path
        assert parser.to_markdown == False

        # 测试markdown模式
        parser_md = DocParser(file_path=file_path, to_markdown=True)
        assert parser_md.to_markdown == True

    @patch('datamax.parser.doc_parser.subprocess.Popen')
    @patch('datamax.parser.doc_parser.os.path.exists')
    def test_doc_to_txt_success(self, mock_exists, mock_popen):
        """测试DOC到TXT转换成功的情况"""
        # 设置模拟
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"conversion complete", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_exists.return_value = True

        parser = DocParser(file_path="test.doc")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = os.path.join(temp_dir, "test.doc")
            result = parser.doc_to_txt(doc_path, temp_dir)
            expected_path = os.path.join(temp_dir, "test.txt")
            assert result == expected_path

    @patch('datamax.parser.doc_parser.subprocess.Popen')
    def test_doc_to_txt_failure(self, mock_popen):
        """测试DOC到TXT转换失败的情况"""
        # 设置模拟失败的转换
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"conversion failed")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        parser = DocParser(file_path="test.doc")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = os.path.join(temp_dir, "test.doc")
            with pytest.raises(Exception, match="Error Output"):
                parser.doc_to_txt(doc_path, temp_dir)

    @patch('builtins.open', new_callable=mock_open, read_data="测试文档内容\n第二行内容")
    @patch('datamax.parser.doc_parser.chardet.detect')
    def test_read_txt_file_success(self, mock_detect, mock_file):
        """测试读取TXT文件成功"""
        mock_detect.return_value = {"encoding": "utf-8"}
        
        parser = DocParser(file_path="test.doc")
        content = parser.read_txt_file("test.txt")
        
        assert content == "测试文档内容\n第二行内容"
        mock_detect.assert_called_once()

    def test_read_txt_file_not_found(self):
        """测试读取不存在的TXT文件"""
        parser = DocParser(file_path="test.doc")
        
        with pytest.raises(Exception, match="文件未找到"):
            parser.read_txt_file("nonexistent.txt")

    @patch('datamax.parser.doc_parser.shutil.copy')
    @patch('datamax.parser.doc_parser.tempfile.TemporaryDirectory')
    def test_read_doc_file_success(self, mock_temp_dir, mock_copy):
        """测试读取DOC文件成功"""
        # 设置临时目录模拟
        temp_path = "/tmp/test"
        mock_temp_dir.return_value.__enter__.return_value = temp_path
        
        parser = DocParser(file_path="test.doc")
        
        # 模拟doc_to_txt和read_txt_file方法
        with patch.object(parser, 'doc_to_txt', return_value="/tmp/test/tmp.txt") as mock_doc_to_txt, \
             patch.object(parser, 'read_txt_file', return_value="文档内容") as mock_read_txt:
            
            content = parser.read_doc_file("test.doc")
            
            assert content == "文档内容"
            mock_copy.assert_called_once()
            mock_doc_to_txt.assert_called_once()
            mock_read_txt.assert_called_once()

    def test_read_doc_file_not_found(self):
        """测试读取不存在的DOC文件"""
        parser = DocParser(file_path="test.doc")
        
        with pytest.raises(Exception, match="文件未找到"):
            parser.read_doc_file("nonexistent.doc")

    @patch('datamax.parser.doc_parser.os.path.exists')
    @patch('datamax.parser.doc_parser.os.path.getsize')
    def test_parse_file_not_exists(self, mock_getsize, mock_exists):
        """测试解析不存在的文件"""
        mock_exists.return_value = False
        
        parser = DocParser(file_path="test.doc")
        
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            parser.parse("nonexistent.doc")

    @patch('datamax.parser.doc_parser.os.path.exists')
    @patch('datamax.parser.doc_parser.os.path.getsize')
    def test_parse_success_without_markdown(self, mock_getsize, mock_exists):
        """测试成功解析DOC文件（不转换为markdown）"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        parser = DocParser(file_path="test.doc", to_markdown=False)
        
        with patch.object(parser, 'read_doc_file', return_value="文档内容测试") as mock_read, \
             patch.object(parser, 'get_file_extension', return_value="doc") as mock_ext, \
             patch.object(parser, 'generate_lifecycle') as mock_lifecycle:
            
            # 设置lifecycle模拟
            mock_lifecycle_obj = MagicMock()
            mock_lifecycle_obj.to_dict.return_value = {
                "update_time": "2024-01-01 12:00:00",
                "life_type": ["LLM_ORIGIN"],
                "life_metadata": {"source_file": "test.doc"}
            }
            mock_lifecycle.return_value = mock_lifecycle_obj
            
            result = parser.parse("test.doc")
            
            assert result["extension"] == "doc"
            assert result["content"] == "文档内容测试"
            assert "lifecycle" in result
            mock_read.assert_called_once_with(doc_path="test.doc")
            mock_ext.assert_called_once_with("test.doc")

    @patch('datamax.parser.doc_parser.os.path.exists')
    @patch('datamax.parser.doc_parser.os.path.getsize')
    def test_parse_success_with_markdown(self, mock_getsize, mock_exists):
        """测试成功解析DOC文件（转换为markdown）"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        parser = DocParser(file_path="test.doc", to_markdown=True)
        
        with patch.object(parser, 'read_doc_file', return_value="标题\n内容") as mock_read, \
             patch.object(parser, 'get_file_extension', return_value="doc") as mock_ext, \
             patch.object(parser, 'generate_lifecycle') as mock_lifecycle:
            
            # 设置lifecycle模拟
            mock_lifecycle_obj = MagicMock()
            mock_lifecycle_obj.to_dict.return_value = {
                "update_time": "2024-01-01 12:00:00",
                "life_type": ["LLM_ORIGIN"],
                "life_metadata": {"source_file": "test.doc"}
            }
            mock_lifecycle.return_value = mock_lifecycle_obj
            
            result = parser.parse("test.doc")
            
            assert result["extension"] == "doc"
            assert result["content"] == "标题\n内容"  # format_as_markdown在这种情况下保持原样
            assert "lifecycle" in result

    def test_format_as_markdown(self):
        """测试markdown格式化功能"""
        parser = DocParser(file_path="test.doc")
        
        # 测试空内容
        assert parser.format_as_markdown("") == ""
        assert parser.format_as_markdown("   ") == "   "
        
        # 测试普通内容
        content = "标题1\n\n内容段落1\n内容段落2\n\n标题2\n内容段落3"
        result = parser.format_as_markdown(content)
        expected = "标题1\n\n内容段落1\n内容段落2\n\n标题2\n内容段落3"
        assert result == expected

    def test_integration_with_temp_file(self, tmp_path):
        """集成测试：使用临时文件测试基本功能（不依赖soffice）"""
        # 创建临时DOC文件（实际上是txt文件，用于测试）
        doc_file = tmp_path / "test.doc"
        doc_file.write_text("这是一个测试文档\n包含多行内容", encoding="utf-8")
        
        parser = DocParser(file_path=str(doc_file), to_markdown=False)
        
        # 模拟soffice转换过程
        with patch.object(parser, 'read_doc_file', return_value="这是一个测试文档\n包含多行内容"):
            result = parser.parse(str(doc_file))
            
            assert result["extension"] == "doc"
            assert "这是一个测试文档" in result["content"]
            assert result["lifecycle"][0]["life_metadata"]["source_file"] == str(doc_file)

    def test_error_handling_in_parse(self):
        """测试parse方法的错误处理"""
        parser = DocParser(file_path="test.doc")
        
        with patch('datamax.parser.doc_parser.os.path.exists', return_value=True), \
             patch('datamax.parser.doc_parser.os.path.getsize', return_value=1024), \
             patch.object(parser, 'read_doc_file', side_effect=Exception("读取文件时出错")):
            
            with pytest.raises(Exception):
                parser.parse("test.doc")

    @patch('datamax.parser.doc_parser.subprocess.Popen')
    def test_subprocess_error_handling(self, mock_popen):
        """测试subprocess错误处理"""
        # 设置模拟subprocess错误
        mock_popen.side_effect = OSError("Command not found")
        
        parser = DocParser(file_path="test.doc")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = os.path.join(temp_dir, "test.doc")
            with pytest.raises(OSError, match="Command not found"):
                parser.doc_to_txt(doc_path, temp_dir)

    @patch('datamax.parser.doc_parser.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data="测试内容")
    def test_encoding_detection(self, mock_file, mock_detect):
        """测试文件编码检测"""
        # 测试检测不到编码的情况
        mock_detect.return_value = {"encoding": None}
        
        parser = DocParser(file_path="test.doc")
        content = parser.read_txt_file("test.txt")
        
        # 应该默认使用utf-8编码
        assert content == "测试内容"
        mock_detect.assert_called_once() 