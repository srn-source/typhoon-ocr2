import json

from typhoon_ocr2 import PARSE_PROMPT, cli
from typhoon_ocr2.prompts import v15_prompt
from conftest import FakeServer, attach


def test_parse_prompt_is_the_trained_prompt():
    assert PARSE_PROMPT.startswith("- Extract layout information from the provided PDF image.")
    assert PARSE_PROMPT.endswith("- Final output must be a single JSON object.")
    assert "'page_footnote', 'footer']" in PARSE_PROMPT


def test_v15_prompt_language():
    assert "Describe in English." in v15_prompt("English")


def test_cli_json(tmp_path, page_image, monkeypatch, capsys):
    p = tmp_path / "page.png"
    page_image.save(p)
    srv = FakeServer()
    real = cli.TyphoonOCR
    monkeypatch.setattr(cli, "TyphoonOCR", lambda **kw: attach(real(**kw), srv))
    assert cli.main([str(p), "--base-url", "http://x/v1", "--model", "m", "--format", "json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["page"] == 1 and out[0]["width"] == 400 and len(out[0]["elements"]) == 3
    assert out[0]["truncated"] is False and out[0]["markdown"]


def test_cli_error_exit_code(tmp_path, capsys):
    assert cli.main([str(tmp_path / "missing.png"), "--base-url", "http://x/v1", "--model", "m"]) == 1
    assert "error:" in capsys.readouterr().err
