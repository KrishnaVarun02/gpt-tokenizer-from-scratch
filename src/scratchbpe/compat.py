"""Recover the cl100k_base merge tree, then use our own encoder/decoder.

Only this optional module imports tiktoken. It supplies pretrained byte ranks,
not our training, encoding, decoding, or persistence implementation.
"""

from .core import RegexTokenizer


def recover_merges(ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
    """Replay lower-ranked merges to recover each parent token's two children.

    This algorithm follows the MIT-licensed minbpe compatibility exercise;
    see third_party/MINBPE_LICENSE and REFERENCE.md.
    """
    merges = {}
    for raw, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if len(raw) == 1:
            continue
        pieces = [bytes([byte]) for byte in raw]
        while len(pieces) > 2:
            candidates = []
            for offset in range(len(pieces) - 1):
                candidate = ranks.get(pieces[offset] + pieces[offset + 1])
                if candidate is not None and candidate < rank:
                    candidates.append((candidate, offset))
            if not candidates:
                raise ValueError(f"Cannot recover merge for rank {rank}")
            _, offset = min(candidates)
            pieces[offset : offset + 2] = [pieces[offset] + pieces[offset + 1]]
        merges[(ranks[pieces[0]], ranks[pieces[1]])] = rank
    return merges


class GPT4Tokenizer(RegexTokenizer):
    """Match the tutorial's cl100k_base encoding, not every modern GPT model."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("Install the compat extra: uv sync --extra compat") from exc
        try:
            reference = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise RuntimeError("Unable to load public cl100k_base data; check network access and TIKTOKEN_CACHE_DIR") from exc
        # These private table attributes are intentionally isolated here and the
        # tiktoken version is pinned. No tiktoken encode/decode calls occur here.
        ranks = reference._mergeable_ranks
        self.byte_ids = [ranks[bytes([b])] for b in range(256)]
        self.vocab = {token: bytes([byte]) for byte, token in enumerate(self.byte_ids)}
        self.merges = recover_merges(ranks)
        for (left, right), token in sorted(self.merges.items(), key=lambda item: item[1]):
            self.vocab[token] = self.vocab[left] + self.vocab[right]
        if self.vocab != {rank: raw for raw, rank in ranks.items()}:
            raise ValueError("Recovered vocabulary differs from cl100k_base")
        self.register_special_tokens(reference._special_tokens)
        self.pretrained = True
