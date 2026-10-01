# GPT tokenizer from scratch

An independent Python implementation of Andrej Karpathy's [GPT Tokenizer lesson](https://www.youtube.com/watch?v=zduSFxRajkE) and [minbpe exercise](https://github.com/karpathy/minbpe). It trains byte-level BPE, encodes and decodes text, persists models, respects GPT-style regex boundaries, handles special tokens explicitly, and reproduces the lesson's `cl100k_base` compatibility exercise.

The core algorithm is Python code in `src/scratchbpe/core.py`. No tokenizer library trains or encodes text for it. The optional `tiktoken` dependency supplies pretrained tables and an independent comparison; regex splitting uses `regex` because Python's built-in `re` lacks Unicode property classes.

## Prerequisites

- Python **3.13.5** (pinned in `.python-version`; package also supports 3.11–3.13).
- [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.11.18** for the commands below. `uv sync` creates an isolated `.venv` and installs the pinned `uv.lock` dependencies. It can download the pinned Python if needed.
- No Docker, WSL2, Kubernetes, account, GPU, or API key. Windows runs natively in PowerShell. macOS Intel and Apple Silicon use native Python wheels; no container architecture emulation is required.
- Internet is needed for installation and the **first** optional compatibility check's public tokenizer-table download. Core tests and training then run offline.

## macOS: zsh or bash

Run inside this repository:

```sh
uv sync --locked
uv run --no-sync pytest -q -m 'not integration'
uv run --no-sync scratchbpe demo
uv run --no-sync scratchbpe train --input examples/corpus.txt --vocab-size 320 --output models/corpus
uv run --no-sync scratchbpe encode models/toy.model --text aaabdaaabac
uv run --no-sync scratchbpe decode models/toy.model --ids '[258,100,258,97,99]'
```

## Windows: PowerShell

Run inside this repository. No shell activation or execution-policy change is necessary:

```powershell
$env:PYTHONUTF8 = '1'
uv sync --locked
uv run --no-sync pytest -q -m 'not integration'
uv run --no-sync scratchbpe demo
uv run --no-sync scratchbpe train --input examples/corpus.txt --vocab-size 320 --output models/corpus
uv run --no-sync scratchbpe encode models/toy.model --text aaabdaaabac
uv run --no-sync scratchbpe decode models/toy.model --ids '[258,100,258,97,99]'
```

The demo prints three learned merges, `[258, 100, 258, 97, 99]`, the original `aaabdaaabac`, and confirmation that the saved model reloaded successfully. Training reports actual vocabulary size, input bytes, token count, and its round-trip result. This is a terminal/Python project; there is no web service to start.

## Python API and special tokens

```python
from scratchbpe import BasicTokenizer, RegexTokenizer, load_tokenizer

basic = BasicTokenizer()
basic.train("aaabdaaabac", vocab_size=259)
assert basic.encode("aaabdaaabac") == [258, 100, 258, 97, 99]

tokenizer = RegexTokenizer()  # GPT-4/cl100k regex; GPT2_PATTERN is also available
text = "hello123!!!? (안녕하세요!) 😉"
tokenizer.train(text * 10, vocab_size=300)
assert tokenizer.decode(tokenizer.encode(text)) == text
tokenizer.register_special_tokens({"<|endoftext|>": 1000})
ids = tokenizer.encode("<|endoftext|>" + text, allowed_special="all")
assert ids[0] == 1000
tokenizer.save("models/custom")
restored = load_tokenizer("models/custom.model")
assert restored.encode("<|endoftext|>" + text, "all") == ids
```

Special spellings in text raise by default. Pass `"all"` to recognize every registered special, `"none"` to encode their literal text, or a set such as `{"<|endoftext|>"}` to recognize only those spellings. In set mode other spellings are ordinary text, matching minbpe's behavior. IDs must be unique and outside the learned vocabulary. CLI registration uses `--special '<|endoftext|>=1000'`; encoding uses `--allowed-special all`.

## Exact tutorial GPT compatibility (optional, no paid service)

This targets **`cl100k_base`**, the encoding used in the lesson, not all models called GPT. The optional module reconstructs merges and the 256-byte ID permutation from the reference tables, then calls our encoder and decoder. It deliberately does not call `tiktoken.encode`/`decode` internally. The `compare` command and integration tests call the reference separately.

macOS:

```sh
uv sync --locked --extra compat
export TIKTOKEN_CACHE_DIR=.tiktoken-cache
RUN_COMPAT_TESTS=1 uv run --no-sync pytest -q -m integration
uv run --no-sync scratchbpe compare --text 'hello123!!!? (안녕하세요!) 😉'
uv run --no-sync scratchbpe compare --text '<|endoftext|>hello world' --allowed-special all
```

PowerShell:

```powershell
uv sync --locked --extra compat
$env:TIKTOKEN_CACHE_DIR = '.tiktoken-cache'
$env:RUN_COMPAT_TESTS = '1'
uv run --no-sync pytest -q -m integration
uv run --no-sync scratchbpe compare --text 'hello123!!!? (안녕하세요!) 😉'
uv run --no-sync scratchbpe compare --text '<|endoftext|>hello world' --allowed-special all
```

The first comparison should print `match: true` with IDs `[15339,4513,12340,30,320,31495,230,75265,243,92245,16715,57037]`. The second prints `[100257,15339,1917]`. Download failure is reported as an error, never replaced with a fabricated vocabulary. Once cached, the same check can run offline. `GPT4Tokenizer().save(...)` also exports an offline model that `load_tokenizer` opens without `tiktoken`.

## Persistence, text fidelity, and limits

- `.model` is versioned JSON with byte IDs, merges, regex, special tokens, and a pretrained flag. It is **not compatible with minbpe's text model format**. Load rejects invalid or forward-referencing merges and ID collisions; it never deserializes Python code.
- `.vocab` is a readable table: ID, lossless byte hex, escaped text preview. A single token can contain incomplete UTF-8; decoding joins bytes first. `decode(..., errors="strict")` rejects invalid combined UTF-8, while the tutorial-compatible default replaces it. `decode_bytes` is lossless for arbitrary byte tokens.
- Regex matches must cover all text, so custom patterns cannot silently discard characters. Input files preserve CRLF, tabs, NUL, Unicode, and whitespace. `decode --output FILE` writes exact bytes with no extra newline.
- Pair ties resolve by first occurrence, as in minbpe. Tiny or empty training inputs can exhaust pairs before the requested vocabulary size; the size is an upper bound. No normalization occurs: composed and decomposed Unicode remain distinct.
- This is a readable educational implementation, not a high-throughput production tokenizer. It materializes corpus chunks and repeatedly scans them. Pretrained compatibility models cannot be retrained.
- SentencePiece/Llama, multimodal tokenization, and Transformer training discussed later in the video are outside this byte-level GPT implementation. They are not represented as implemented features.

## Tests and cleanup

`uv run --no-sync pytest -q` runs deterministic tests and visibly skips optional integration checks unless `RUN_COMPAT_TESTS=1`. CI runs both groups on Windows, macOS, and Linux. [VERIFICATION.md](VERIFICATION.md) distinguishes actual local results from CI status; [REFERENCE.md](REFERENCE.md) maps tutorial behavior to implementation and tests.

Commands exit after completion; there are no running servers or cloud resources. Remove generated files on macOS:

```sh
rm -rf .venv models .pytest_cache .tiktoken-cache
```

PowerShell:

```powershell
Remove-Item -Recurse -Force .venv, models, .pytest_cache, .tiktoken-cache -ErrorAction SilentlyContinue
Remove-Item Env:RUN_COMPAT_TESTS -ErrorAction SilentlyContinue
Remove-Item Env:TIKTOKEN_CACHE_DIR -ErrorAction SilentlyContinue
```

MIT license. The original corpus is included under this project's license. Required minbpe and tiktoken notices are in `third_party/`; screenshots, transcripts, original Wikipedia article, pretrained tables, and videos are not redistributed.
