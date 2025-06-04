import json
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datamax.parser.json_parser import JsonParser

def test_json_parser(tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    parser = JsonParser(file_path=str(json_file))
    result = parser.parse(file_path=str(json_file))
    assert result["title"] == "json"
    assert "\"a\"" in result["content"]
    assert result["lifecycle"][0]["life_metadata"]["source_file"] == str(json_file)

