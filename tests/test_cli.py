import json
import subprocess
import sys


def run(*args):
    return subprocess.run([sys.executable, "-m", "scratchbpe", *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def test_full_terminal_flow(tmp_path):
    prefix = tmp_path / "toy"
    demo = run("demo", "--output", prefix)
    assert demo.returncode == 0, demo.stderr
    assert "[258, 100, 258, 97, 99]" in demo.stdout
    assert "reload verified" in demo.stdout
    encoded = run("encode", str(prefix) + ".model", "--text", "aaabdaaabac")
    assert json.loads(encoded.stdout) == [258, 100, 258, 97, 99]
    decoded = run("decode", str(prefix) + ".model", "--ids", encoded.stdout)
    assert decoded.stdout.strip() == "aaabdaaabac"


def test_utf8_file_flow_preserves_crlf_and_spaces(tmp_path):
    source = tmp_path / "input.txt"
    raw = "😀 नमस्ते\r\n\tspace   \n".encode("utf-8")
    source.write_bytes(raw)
    prefix = tmp_path / "regex"
    trained = run("train", "--input", source, "--output", prefix, "--special", "<|endoftext|>=1000")
    assert trained.returncode == 0, trained.stderr
    assert json.loads(trained.stdout)["round_trip"]
    model = str(prefix) + ".model"
    encoded = run("encode", model, "--input", source)
    output = tmp_path / "decoded.txt"
    decoded = run("decode", model, "--ids", encoded.stdout, "--output", output)
    assert decoded.returncode == 0, decoded.stderr
    assert output.read_bytes() == raw
    special = run("encode", model, "--text", "<|endoftext|>")
    assert special.returncode == 2 and "explicitly" in special.stderr


def test_invalid_ids_and_missing_model_are_clean_errors(tmp_path):
    demo = run("demo", "--output", tmp_path / "toy")
    assert demo.returncode == 0
    bad = run("decode", tmp_path / "toy.model", "--ids", "{}")
    assert bad.returncode == 2 and "JSON array" in bad.stderr
    missing = run("encode", tmp_path / "missing.model", "--text", "hi")
    assert missing.returncode == 2 and "error:" in missing.stderr


def test_invalid_saved_regex_is_a_clean_error(tmp_path):
    prefix = tmp_path / "toy"
    assert run("demo", "--output", prefix).returncode == 0
    path = tmp_path / "toy.model"
    model = json.loads(path.read_text(encoding="utf-8"))
    model["pattern"] = "["
    path.write_text(json.dumps(model), encoding="utf-8")
    result = run("encode", path, "--text", "hello")
    assert result.returncode == 2
    assert "Invalid regex pattern" in result.stderr
    assert "Traceback" not in result.stderr
