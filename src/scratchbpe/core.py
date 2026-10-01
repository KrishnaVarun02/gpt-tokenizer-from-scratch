"""The core algorithm never calls a tokenizer library.

Train: count adjacent pairs and merge the most frequent pair.
Encode: apply available merges in ascending learned rank, within regex chunks.
Decode: join token bytes before decoding UTF-8 (individual tokens can split it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import regex

# Patterns from minbpe / OpenAI tiktoken, MIT; notices in third_party/.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def pair_counts(chunks: Iterable[list[int]]) -> dict[tuple[int, int], int]:
    """Count overlapping pairs; insertion order gives deterministic tie breaks."""
    counts: dict[tuple[int, int], int] = {}
    for chunk in chunks:
        for pair in zip(chunk, chunk[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def replace_pair(ids: list[int], pair: tuple[int, int], token: int) -> list[int]:
    """Replace non-overlapping occurrences, scanning left to right."""
    result = []
    cursor = 0
    while cursor < len(ids):
        if cursor + 1 < len(ids) and (ids[cursor], ids[cursor + 1]) == pair:
            result.append(token)
            cursor += 2
        else:
            result.append(ids[cursor])
            cursor += 1
    return result


class BasicTokenizer:
    """Trainable UTF-8 BPE. Vocabulary size is an upper bound for tiny corpora."""

    def __init__(self) -> None:
        self.pattern: str | None = None
        self.byte_ids = list(range(256))
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.pretrained = False

    def split(self, text: str) -> list[str]:
        return [text] if text else []

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if self.pretrained:
            raise ValueError("Pretrained compatibility models cannot be retrained")
        if type(vocab_size) is not int or vocab_size < 256:
            raise ValueError("vocab_size must be an integer of at least 256")
        if any(token < vocab_size for token in self.special_tokens.values()):
            raise ValueError("Special token IDs must be outside the requested vocabulary")
        chunks = [list(part.encode("utf-8")) for part in self.split(text)]
        merges: dict[tuple[int, int], int] = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for token in range(256, vocab_size):
            counts = pair_counts(chunks)
            if not counts:
                break
            pair = max(counts, key=counts.__getitem__)
            chunks = [replace_pair(chunk, pair, token) for chunk in chunks]
            merges[pair] = token
            vocab[token] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {token - 255}: {pair} -> {token}, count={counts[pair]}, bytes={vocab[token]!r}")
        self.merges, self.vocab = merges, vocab
        self.byte_ids = list(range(256))

    def _encode_chunk(self, part: str) -> list[int]:
        ids = [self.byte_ids[b] for b in part.encode("utf-8")]
        while len(ids) > 1:
            candidates = (pair for pair in zip(ids, ids[1:]) if pair in self.merges)
            pair = min(candidates, key=self.merges.__getitem__, default=None)
            if pair is None:
                break
            ids = replace_pair(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        return [token for part in self.split(text) for token in self._encode_chunk(part)]

    def encode(self, text: str, allowed_special: str | set[str] = "none_raise") -> list[int]:
        if allowed_special == "all":
            allowed = self.special_tokens
        elif allowed_special == "none":
            allowed = {}
        elif allowed_special == "none_raise":
            if any(s in text for s in self.special_tokens):
                raise ValueError("Special token in input; choose allowed_special explicitly")
            allowed = {}
        elif isinstance(allowed_special, set):
            if not allowed_special <= self.special_tokens.keys():
                raise ValueError("Unknown special token in allowed_special")
            allowed = {s: self.special_tokens[s] for s in allowed_special}
        else:
            raise ValueError("allowed_special must be all, none, none_raise, or a set")
        if not allowed:
            return self.encode_ordinary(text)
        # Longest first makes overlapping user-defined specials unambiguous.
        boundary = "(" + "|".join(regex.escape(s) for s in sorted(allowed, key=lambda s: (-len(s), s))) + ")"
        result: list[int] = []
        for part in regex.split(boundary, text):
            result.extend([allowed[part]] if part in allowed else self.encode_ordinary(part))
        return result

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        if not isinstance(special_tokens, dict):
            raise ValueError("special_tokens must be a dictionary")
        used: set[int] = set()
        for spelling, token in special_tokens.items():
            if not isinstance(spelling, str) or not spelling:
                raise ValueError("Special token spellings must be nonempty strings")
            if type(token) is not int or token < 0 or token in self.vocab or token in used:
                raise ValueError("Special token IDs must be unique nonnegative integers outside the vocabulary")
            used.add(token)
        self.special_tokens = dict(special_tokens)

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        specials = {i: s.encode("utf-8") for s, i in self.special_tokens.items()}
        parts = []
        for token in ids:
            if type(token) is not int or (token not in self.vocab and token not in specials):
                raise ValueError(f"Unknown token ID: {token!r}")
            parts.append(self.vocab[token] if token in self.vocab else specials[token])
        return b"".join(parts)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        """Write portable versioned JSON and a human-readable vocabulary."""
        model, vocab = Path(str(prefix) + ".model"), Path(str(prefix) + ".vocab")
        model.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "scratchbpe", "version": 1, "pattern": self.pattern,
            "byte_ids": self.byte_ids, "pretrained": self.pretrained,
            "merges": [[*pair, token] for pair, token in sorted(self.merges.items(), key=lambda p: p[1])],
            "special_tokens": self.special_tokens,
        }
        model.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        entries = {**self.vocab, **{i: s.encode("utf-8") for s, i in self.special_tokens.items()}}
        with vocab.open("w", encoding="utf-8", newline="\n") as out:
            for token, raw in sorted(entries.items()):
                # Hex is lossless even when a token contains incomplete UTF-8.
                out.write(f"{token}\t{raw.hex()}\t{json.dumps(raw.decode('utf-8', errors='replace'), ensure_ascii=True)}\n")
        return model, vocab

    def load(self, path: str | Path) -> None:
        replacement = load_tokenizer(path)
        if (replacement.pattern is None) != (self.pattern is None):
            raise ValueError("Tokenizer kind differs; use load_tokenizer(path) instead")
        self.__dict__.update(replacement.__dict__)


class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern: str = GPT4_PATTERN) -> None:
        super().__init__()
        self.pattern = pattern
        try:
            self._compiled = regex.compile(pattern)
        except regex.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

    def split(self, text: str) -> list[str]:
        chunks, cursor = [], 0
        for match in self._compiled.finditer(text):
            if match.start() != cursor or match.start() == match.end():
                raise ValueError("Regex must partition the entire text without empty matches")
            chunks.append(match.group(0))
            cursor = match.end()
        if cursor != len(text):
            raise ValueError("Regex left unmatched input; refusing to lose text")
        return chunks


def load_tokenizer(path: str | Path) -> BasicTokenizer:
    """Validate every merge before constructing vocabulary; never use pickle."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "scratchbpe" or data.get("version") != 1:
        raise ValueError("Unsupported tokenizer model format/version")
    pattern = data.get("pattern")
    if pattern is not None and not isinstance(pattern, str):
        raise ValueError("Invalid regex pattern")
    tokenizer = BasicTokenizer() if pattern is None else RegexTokenizer(pattern)
    byte_ids = data.get("byte_ids")
    if not isinstance(byte_ids, list) or len(byte_ids) != 256 or any(type(i) is not int for i in byte_ids) or set(byte_ids) != set(range(256)):
        raise ValueError("byte_ids must be a permutation of 0..255")
    tokenizer.byte_ids = byte_ids
    tokenizer.vocab = {token: bytes([byte]) for byte, token in enumerate(byte_ids)}
    merges = data.get("merges")
    if not isinstance(merges, list):
        raise ValueError("merges must be a list")
    for expected, row in enumerate(merges, 256):
        if not isinstance(row, list) or len(row) != 3 or any(type(i) is not int for i in row):
            raise ValueError("Invalid merge row")
        left, right, token = row
        if token != expected or left not in tokenizer.vocab or right not in tokenizer.vocab or (left, right) in tokenizer.merges:
            raise ValueError("Invalid merge order, pair, or token ID")
        tokenizer.merges[(left, right)] = token
        tokenizer.vocab[token] = tokenizer.vocab[left] + tokenizer.vocab[right]
    tokenizer.register_special_tokens(data.get("special_tokens", {}))
    if type(data.get("pretrained")) is not bool:
        raise ValueError("pretrained must be a boolean")
    tokenizer.pretrained = data["pretrained"]
    return tokenizer
