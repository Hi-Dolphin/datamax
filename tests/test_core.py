# tests/test_core.py

import pytest
from pathlib import Path

# 注意导入路径，对应 datamax/parser/core.py
from datamax.parser.core import DataMax
# BaseLife 里定义了预置列表 PREDEFINED_DOMAINS :contentReference[oaicite:0]{index=0}
from datamax.parser.base import PREDEFINED_DOMAINS

@pytest.fixture
def dummy_file(tmp_path):
    f = tmp_path / "foo.txt"
    f.write_text("hello world")
    return str(f)

def test_default_domain(dummy_file):
    """不传 domain 时应为默认 'Technology'"""
    dm = DataMax(file_path=dummy_file)
    assert dm.domain == "Technology"

def test_predefined_domains(dummy_file):
    """传入预置列表中的 domain 应被正确接受"""
    for dom in PREDEFINED_DOMAINS:
        dm = DataMax(file_path=dummy_file, domain=dom)
        assert dm.domain == dom

def test_custom_domain_warning(dummy_file, capsys):
    """传入非预置的自定义 domain，应有警告提示且属性正确"""
    custom = "Journalism"
    dm = DataMax(file_path=dummy_file, domain=custom)
    captured = capsys.readouterr()
    assert "不在预置列表" in captured.out
    assert dm.domain == custom

def test_clean_data_uses_domain(monkeypatch, dummy_file):
    """
    clean_data 中对 generate_lifecycle 的调用要使用 self.domain 而非硬编码，
    我们通过 monkeypatch 记录每次调用的 domain 值。
    """
    calls = []
    def fake_gl(self, source_file, domain, life_type, usage_purpose):
        calls.append(domain)
        class LC:
            def to_dict(self): return {}
        return LC()

    # 替换 DataMax.generate_lifecycle :contentReference[oaicite:1]{index=1}
    monkeypatch.setattr(DataMax, "generate_lifecycle", fake_gl)

    dm = DataMax(file_path=dummy_file, domain="Health")
    # clean_data 需要 dm.parsed_data 才不报 “No data to clean.”
    dm.parsed_data = {"content": "some text"}

    dm.clean_data(method_list=["filter"])
    # 至少有一次调用 domain 为 "Health"
    assert "Health" in calls
