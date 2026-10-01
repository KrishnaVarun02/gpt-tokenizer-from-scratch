# Verification record

Local checks on 2026-10-01 used **macOS 26.6.2 (25G83), Apple Silicon arm64, CPython 3.13.5, uv 0.11.18** in the repository's isolated `.venv`. No API key, paid model call, cloud deployment, or external messaging is involved.

| Check actually run | Result |
|---|---|
| `uv lock` and `uv sync --locked --extra compat` | Resolved 13 packages; isolated install succeeded |
| `uv run --no-sync pytest -q -m 'not integration'` | **42 passed, 11 deselected**, 0.88 seconds |
| `uv run --no-sync scratchbpe demo` | Three expected merges; `[258, 100, 258, 97, 99]`; original text restored; saved model reload verified |
| `uv run --no-sync scratchbpe train --input examples/corpus.txt --vocab-size 320 --output models/corpus` | Vocabulary 320, input 640 bytes, output 402 tokens, `round_trip: true` |
| `uv run --no-sync scratchbpe encode models/toy.model --text aaabdaaabac` | `[258, 100, 258, 97, 99]` |
| `uv run --no-sync scratchbpe decode models/toy.model --ids '[258,100,258,97,99]'` | `aaabdaaabac` |
| Public `cl100k_base` compatibility tests | Download/check initiated; final result pending below/CI |

Deterministic tests cover Unicode, emoji, multilingual strings, combining characters, empty input, repeated spaces, tabs, CRLF, NUL, incomplete UTF-8 tokens, randomized round trips, rank ordering, overlapping pairs, regex boundaries, special-token permission/collisions, malformed saved models, and actual CLI subprocesses. The CLI file test compares original and decoded **bytes**, including CRLF, rather than normalized text.

The optional integration suite is deliberately separate: `RUN_COMPAT_TESTS=1 uv run --no-sync pytest -q -m integration`. It compares exact token IDs to the real public `cl100k_base` table for fixed and randomized texts, all five specials, and an exported/reloaded model. No synthetic table is represented as that public vocabulary. Its first download was slow in this environment; the initial curl fallback timed out after 45 seconds with a partial file. Public table SHA-256 must equal `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7` before use; tiktoken verifies this hash.

## Cross-platform evidence

Local native macOS execution above is verified. Windows and Linux have not been run on this local machine. `.github/workflows/test.yml` runs native setup, deterministic tests, public-table compatibility, toy demonstration, and corpus training on all three operating systems. Workflow creation alone is not evidence of successful execution; repository delivery must include the actual latest-commit Actions results. macOS Intel and Windows ARM have not been specifically verified.

The Python applications use no platform-specific shell calls or absolute paths. Windows commands in README avoid virtual-environment activation and set UTF-8 explicitly. No container test is presented as proof of native Windows or macOS execution.
