import json
import random

import pytest

from scratchbpe import BasicTokenizer, RegexTokenizer, load_tokenizer
from scratchbpe.core import GPT2_PATTERN, pair_counts, replace_pair

CASES = ["", "x", "   \t\r\n\n", "hello world!!!? (안녕하세요!) lol123 😉", "😀👩🏽‍💻🦙", "नमस्ते दुनिया! 中文 日本語 العربية русский", "café cafe\u0301", "\0\x01\x7f", "𝕦𝕟𝕚𝕔𝕠𝕕𝕖"]


@pytest.mark.parametrize("factory", [BasicTokenizer, RegexTokenizer])
@pytest.mark.parametrize("text", CASES)
def test_roundtrip_before_and_after_training(factory, text):
    tokenizer = factory()
    assert tokenizer.decode(tokenizer.encode(text), errors="strict") == text
    tokenizer.train(text, 300)
    assert tokenizer.decode(tokenizer.encode(text), errors="strict") == text


@pytest.mark.parametrize("factory", [BasicTokenizer, RegexTokenizer])
def test_exact_tutorial_example(factory):
    tokenizer = factory()
    tokenizer.train("aaabdaaabac", 259)
    assert list(tokenizer.merges.items()) == [((97, 97), 256), ((256, 97), 257), ((257, 98), 258)]
    assert tokenizer.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert tokenizer.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"


def test_overlaps_and_pair_frequency():
    assert pair_counts([[1, 1, 1, 1], [1, 2]]) == {(1, 1): 3, (1, 2): 1}
    assert replace_pair([1, 1, 1, 1, 1], (1, 1), 256) == [256, 256, 1]
    assert pair_counts([[1], [2]]) == {}


def test_encode_uses_merge_rank_not_frequency():
    tokenizer = BasicTokenizer()
    tokenizer.train("abababcc", 258)
    assert tokenizer.encode("ccabab") == [99, 99, 257]


def test_regex_boundaries_and_no_dropped_input():
    tokenizer = RegexTokenizer()
    assert tokenizer.split("hello123456!!!") == ["hello", "123", "456", "!!!"]
    tokenizer.train("hello123456!!!" * 20, 400)
    assert all(not (any(chr(b).isalpha() for b in raw) and any(chr(b).isdigit() for b in raw)) for raw in tokenizer.vocab.values())
    with pytest.raises(ValueError, match="unmatched"):
        RegexTokenizer(r"[a-z]+").encode("abc 123")
    with pytest.raises(ValueError, match="empty"):
        RegexTokenizer(r".*?").encode("abc")


@pytest.mark.parametrize("factory", [BasicTokenizer, RegexTokenizer])
def test_deterministic_random_roundtrips(factory):
    rng = random.Random(42)
    alphabet = "abc123 \t\r\n😀नम世界é\u0301\0"
    tokenizer = factory()
    tokenizer.train("".join(rng.choices(alphabet, k=500)), 310)
    for _ in range(50):
        text = "".join(rng.choices(alphabet, k=rng.randrange(120)))
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_modes_collisions_and_roundtrip(tmp_path):
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world hello world", 270)
    tokenizer.register_special_tokens({"<|endoftext|>": 1000, "<|a|>": 1001})
    text = "<|endoftext|>hello<|a|> world"
    with pytest.raises(ValueError, match="explicitly"):
        tokenizer.encode(text)
    ids = tokenizer.encode(text, "all")
    assert ids[0] == 1000 and 1001 in ids
    assert tokenizer.decode(ids) == text
    assert 1000 not in tokenizer.encode(text, "none")
    subset = tokenizer.encode(text, {"<|a|>"})
    assert 1000 not in subset and 1001 in subset
    tokenizer.save(tmp_path / "special")
    restored = load_tokenizer(tmp_path / "special.model")
    assert restored.encode(text, "all") == ids
    assert restored.decode(ids) == text
    for tokens in ({"x": 1}, {"x": 1000, "y": 1000}, {"": 1000}, {"x": True}):
        with pytest.raises(ValueError):
            tokenizer.register_special_tokens(tokens)
    with pytest.raises(ValueError):
        tokenizer.encode(text, {"unknown"})
    with pytest.raises(ValueError):
        tokenizer.train("hello", 1001)


def test_longest_special_and_newline_persistence(tmp_path):
    tokenizer = RegexTokenizer()
    tokenizer.register_special_tokens({"<a>": 500, "<a>b": 501, "line\nbreak": 502})
    tokenizer.save(tmp_path / "model")
    restored = load_tokenizer(tmp_path / "model.model")
    assert restored.encode("<a>bline\nbreak", "all") == [501, 502]
    assert restored.decode([501, 502]) == "<a>bline\nbreak"


def test_custom_regex_recompiled_on_load(tmp_path):
    tokenizer = RegexTokenizer(GPT2_PATTERN)
    tokenizer.train("hello123456 world", 270)
    tokenizer.save(tmp_path / "gpt2")
    restored = RegexTokenizer()
    restored.load(tmp_path / "gpt2.model")
    assert restored.split("123456") == ["123456"]
    assert restored.encode("hello123456 world") == tokenizer.encode("hello123456 world")


def test_save_is_deterministic_and_portable(tmp_path):
    tokenizer = BasicTokenizer()
    tokenizer.train("aaabdaaabac", 259)
    first, _ = tokenizer.save(tmp_path / "first")
    second, _ = tokenizer.save(tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    assert load_tokenizer(first).merges == tokenizer.merges
    with pytest.raises(ValueError, match="kind differs"):
        RegexTokenizer().load(first)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(version=99),
    lambda d: d.update(byte_ids=[0] * 256),
    lambda d: d.update(merges=[[9999, 97, 256]]),
    lambda d: d.update(merges=[[97, 97, 257]]),
    lambda d: d.update(merges=[[97, 97, 256], [97, 97, 257]]),
    lambda d: d.update(special_tokens={"x": 0}),
    lambda d: d.update(pretrained="yes"),
])
def test_reject_invalid_models(tmp_path, mutation):
    path, _ = BasicTokenizer().save(tmp_path / "model")
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_tokenizer(path)


def test_decode_partial_utf8_and_invalid_tokens():
    tokenizer = BasicTokenizer()
    assert tokenizer.decode([240]) == "�"
    assert tokenizer.decode_bytes([240]) == bytes([240])
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([240], errors="strict")
    for ids in ([10000], [-1], [True], [1.0], ["hello"]):
        with pytest.raises(ValueError):
            tokenizer.decode(ids)


def test_empty_and_exhausted_training():
    tokenizer = BasicTokenizer()
    tokenizer.train("", 300)
    assert len(tokenizer.vocab) == 256
    tokenizer.train("a", 300)
    assert len(tokenizer.vocab) == 256
    with pytest.raises(ValueError):
        tokenizer.train("abc", 255)
