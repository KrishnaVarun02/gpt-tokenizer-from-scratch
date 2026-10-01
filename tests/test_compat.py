import os
import random

import pytest

from scratchbpe import load_tokenizer
from scratchbpe.compat import GPT4Tokenizer, recover_merges


def test_recover_merges_with_synthetic_rank_table():
    ranks = {bytes([b]): b for b in range(256)}
    ranks.update({b"aa": 256, b"aaa": 257, b"aaab": 258})
    assert recover_merges(ranks) == {(97, 97): 256, (256, 97): 257, (257, 98): 258}


@pytest.fixture(scope="module")
def compatibility():
    if os.environ.get("RUN_COMPAT_TESTS") != "1":
        pytest.skip("Set RUN_COMPAT_TESTS=1 to download/check public cl100k_base data")
    tiktoken = pytest.importorskip("tiktoken")
    return GPT4Tokenizer(), tiktoken.get_encoding("cl100k_base")


@pytest.mark.integration
@pytest.mark.parametrize("text", ["", " ", "\t\r\n\n  ", "hello123!!!? (안녕하세요!) 😉", "😀👩🏽‍💻", "नमस्ते दुनिया! 中文 日本語 العربية русский", "café cafe\u0301\0", "1234567890", "we'LL test contractions!\n"])
def test_exact_cl100k_ids(compatibility, text):
    ours, reference = compatibility
    ids = ours.encode(text)
    assert ids == reference.encode(text)
    assert ours.decode(ids) == reference.decode(ids) == text


@pytest.mark.integration
def test_all_specials_and_offline_export(compatibility, tmp_path):
    ours, reference = compatibility
    text = "<|endoftext|>hello<|fim_prefix|>world<|fim_middle|>😀<|fim_suffix|><|endofprompt|>"
    ids = ours.encode(text, "all")
    assert ids == reference.encode(text, allowed_special="all")
    assert ours.decode(ids) == text
    assert ours.encode(text, "none") == reference.encode(text, disallowed_special=())
    path, _ = ours.save(tmp_path / "cl100k")
    restored = load_tokenizer(path)
    assert restored.encode(text, "all") == ids
    assert restored.decode(ids) == text
    with pytest.raises(ValueError, match="cannot be retrained"):
        restored.train("hi", 256)


@pytest.mark.integration
def test_randomized_comparison(compatibility):
    ours, reference = compatibility
    rng = random.Random(2024)
    alphabet = "abc123456789 !?\r\n\t😀👩🏽‍💻é\u0301नम中文العربية"
    for _ in range(50):
        text = "".join(rng.choices(alphabet, k=rng.randrange(200)))
        assert ours.encode(text) == reference.encode(text)
