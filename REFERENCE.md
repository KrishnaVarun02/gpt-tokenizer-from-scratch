# Reference fidelity

Accessed on 2026-10-01:

- Supplied `Build a GPT Tokenizer.png`, inspected visually. It identifies the lesson and byte BPE objective; it is a reference collage, not the requested application UI.
- [YouTube lesson](https://www.youtube.com/watch?v=zduSFxRajkE): fetched public HTML/player metadata and creator description, including chapter timestamps. Retrieved its public English auto-generated VTT captions using yt-dlp and read the algorithm, regex, special token, and compatibility sections. Initial direct timedtext retrieval returned empty content; yt-dlp succeeded. **No audiovisual playback was performed.** Auto-captions can misrecognize identifiers, so source code is authoritative for names and constants.
- [minbpe repository](https://github.com/karpathy/minbpe/tree/1acefe89412b20245db5a22d2a02001e547dc602), cloned and inspected at commit `1acefe89412b20245db5a22d2a02001e547dc602`: README, `exercise.md`, `lecture.md`, `minbpe/{base,basic,regex,gpt4}.py`, `tests/test_tokenizer.py`, `requirements.txt`, and MIT license. The repository's `lecture.md` contains only an introduction, so it was not treated as the full transcript.
- [tiktoken 0.8.0 MIT license](https://github.com/openai/tiktoken/blob/0.8.0/LICENSE), and the installed package's `cl100k_base` tables for the compatibility check.

Timestamps below are creator description chapters, corroborated by captions; source paths are pinned links. No unverified line-perfect notebook reproduction is claimed.

| Demonstrated feature | Video/source reference | Implementation | Verification |
|---|---|---|---|
| UTF-8 bytes; train by counting frequent adjacent pairs and merging | 00:18:15, 00:23:50, 00:28:35, 00:30:36, 00:34:58; [basic.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py) | `core.py`: `pair_counts`, `replace_pair`, `BasicTokenizer.train` | Exact toy IDs, overlapping pairs, deterministic ties, empty inputs |
| Join vocabulary bytes before UTF-8 decode | 00:42:47; [basic.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py) | `decode_bytes`, `decode` | Unicode/emoji/multilingual round trips, partial UTF-8 behavior |
| Encoding follows learned merge rank | 00:48:21; [basic.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py) | `_encode_chunk` | Rank-order case and exact toy output |
| Regex chunks prevent category-spanning merges; GPT-2/GPT-4 patterns | 00:57:36, 01:11:38; [regex.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/regex.py) | `RegexTokenizer`, `GPT2_PATTERN`, `GPT4_PATTERN` | Numeric triplet boundaries, whitespace, full input coverage |
| Explicit special-token permission | 01:18:26; [regex.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/regex.py) | `register_special_tokens`, `encode` | Default rejection, all/none/subset, persistence, collisions |
| Recover cl100k merges and byte permutation | 01:25:28; [exercise steps 3–4](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/exercise.md), [gpt4.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/gpt4.py) | `compat.py`: `recover_merges`, `GPT4Tokenizer` | Public table comparison, all five special tokens, randomized texts, exact README example |
| Train custom vocabulary and inspect merges; save/load | 01:27:00 vocabulary discussion; [base.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/base.py), [train.py](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/train.py) | `.model`/`.vocab`, terminal `demo`, `train`, `encode`, `decode` | Actual CLI subprocess flow, byte-preserving file round trip, custom regex reload |

## Deliberate adaptations and gaps

- Independent source layout and a portable terminal command wrap the tutorial's notebook/Python methods. No new graphical app is invented. The `aaabdaaabac` example and compatibility output match the companion repository exactly.
- Original small multilingual training corpus replaces the lecture's copied blog prose and Wikipedia article. Those third-party texts are not redistributed. Learned vocabulary on this corpus therefore differs; the algorithm and workflow match.
- Python 3.13.5, regex 2024.11.6, tiktoken 0.8.0, pytest 8.3.5 and a full `uv.lock` replace upstream's unpinned `regex`/`tiktoken` requirements. Optional compatibility depends on the pinned package's private `_mergeable_ranks`/`_special_tokens` table attributes, isolated in one module.
- JSON model format adds strict validation, compiled-pattern restoration, newline-containing special tokens, and byte mapping persistence. It does not read upstream minbpe v1 files. Unlike the inspected upstream `GPT4Tokenizer`, compatibility models can be exported and special tokens decode correctly.
- Training on exhausted/empty input stops safely instead of calling `max()` on an empty pair dictionary. Longest-match special handling and explicit collision errors make edge behavior defined.
- GPT-2's split pattern is provided, but GPT-2 pretrained merge compatibility, SentencePiece/Llama, multimodal quantization, and language-model training are not implemented or claimed. The lesson describes them beyond the requested GPT byte-level BPE core.
- No missing private source or paid credential blocks this project. Full audiovisual playback and the hosted Colab were not inspected; reviewed captions plus creator code support the implemented scope.

## Attribution and redistribution

The algorithm implementation was written for this project following the tutorial and exercise. Regex patterns and compatibility recovery approach derive from MIT-licensed minbpe/OpenAI material. Preserve `third_party/MINBPE_LICENSE` (Copyright 2024 Andrej) and `third_party/TIKTOKEN_LICENSE` (Copyright 2022 OpenAI) when redistributing. The repository does not include downloaded captions, screenshots, reference clone, pretrained vocabulary, or tutorial corpora.
