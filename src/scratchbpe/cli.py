"""Portable terminal interface; JSON token arrays work in PowerShell and zsh."""

import argparse
import json
import sys
from pathlib import Path

from .core import BasicTokenizer, RegexTokenizer, load_tokenizer


def _text(args):
    if args.text is not None:
        return args.text
    # Preserve CRLF and all whitespace; read_text otherwise normalizes newlines.
    return Path(args.input).read_bytes().decode("utf-8")


def _input(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", help="UTF-8 input file, preserved byte for byte")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and inspect byte-level BPE built from scratch")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Reproduce the tutorial toy BPE example")
    demo.add_argument("--output", default="models/toy")
    train = commands.add_parser("train")
    _input(train)
    train.add_argument("--kind", choices=["basic", "regex"], default="regex")
    train.add_argument("--vocab-size", type=int, default=320)
    train.add_argument("--output", default="models/custom")
    train.add_argument("--special", action="append", default=[], help="TOKEN=ID, e.g. '<|endoftext|>=1000'")
    train.add_argument("--verbose", action="store_true")
    encode = commands.add_parser("encode")
    encode.add_argument("model")
    _input(encode)
    encode.add_argument("--allowed-special", choices=["all", "none", "none_raise"], default="none_raise")
    decode = commands.add_parser("decode")
    decode.add_argument("model")
    decode.add_argument("--ids", required=True, help="JSON array of integer token IDs")
    decode.add_argument("--output", help="Write UTF-8 bytes to a file without an added newline")
    compat = commands.add_parser("compare", help="Compare our encoder to tiktoken's cl100k_base")
    _input(compat)
    compat.add_argument("--allowed-special", choices=["all", "none", "none_raise"], default="none_raise")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            tokenizer = BasicTokenizer()
            text = "aaabdaaabac"
            tokenizer.train(text, 259, verbose=True)
            ids = tokenizer.encode(text)
            print("tokens:", ids)
            print("decoded:", tokenizer.decode(ids))
            model, vocab = tokenizer.save(args.output)
            restored = load_tokenizer(model)
            if restored.encode(text) != ids or restored.decode(ids) != text:
                raise ValueError("Saved model failed round-trip verification")
            print(f"saved: {model}, {vocab}; reload verified")
        elif args.command == "train":
            tokenizer = BasicTokenizer() if args.kind == "basic" else RegexTokenizer()
            text = _text(args)
            tokenizer.train(text, args.vocab_size, args.verbose)
            special = {}
            for entry in args.special:
                spelling, token = entry.rsplit("=", 1)
                if spelling in special:
                    raise ValueError("Duplicate special token")
                special[spelling] = int(token)
            tokenizer.register_special_tokens(special)
            tokenizer.save(args.output)
            ids = tokenizer.encode(text, "none")
            if tokenizer.decode(ids) != text:
                raise ValueError("Training round trip failed")
            print(json.dumps({"vocabulary": len(tokenizer.vocab), "bytes": len(text.encode('utf-8')), "tokens": len(ids), "round_trip": True, "model": args.output + '.model'}))
        elif args.command == "encode":
            print(json.dumps(load_tokenizer(args.model).encode(_text(args), args.allowed_special)))
        elif args.command == "decode":
            ids = json.loads(args.ids)
            if not isinstance(ids, list):
                raise ValueError("--ids must be a JSON array")
            tokenizer = load_tokenizer(args.model)
            if args.output:
                Path(args.output).write_bytes(tokenizer.decode_bytes(ids))
            else:
                print(tokenizer.decode(ids))
        elif args.command == "compare":
            from .compat import GPT4Tokenizer
            import tiktoken

            text = _text(args)
            tokenizer = GPT4Tokenizer()
            ours = tokenizer.encode(text, args.allowed_special)
            reference = tiktoken.get_encoding("cl100k_base")
            kwargs = {"disallowed_special": ()} if args.allowed_special == "none" else {"allowed_special": "all" if args.allowed_special == "all" else set()}
            expected = reference.encode(text, **kwargs)
            equal = ours == expected and tokenizer.decode(ours) == text
            print(json.dumps({"encoding": "cl100k_base", "ours": ours, "reference": expected, "match": equal}))
            if not equal:
                return 1
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
