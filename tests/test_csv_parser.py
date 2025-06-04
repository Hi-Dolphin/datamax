import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datamax.parser.csv_parser import CsvParser

def test_csv_parser(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    parser = CsvParser(file_path=str(csv_file))
    result = parser.parse(file_path=str(csv_file))
    assert result["title"] == "csv"
    assert "a" in result["content"]
    assert result["lifecycle"][0]["life_metadata"]["source_file"] == str(csv_file)

