"""CLI: prepare LJSpeech features (download/extract, tokenize, mel cache)."""

from __future__ import annotations

import argparse

from dsflow.config import DataConfig
from dsflow.data.ljspeech import ensure_ljspeech, preprocess_ljspeech
from dsflow.train import build_tokenizer, metadata_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LJSpeech features for DSFlow training")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None, help="limit utterances (debug)")
    parser.add_argument("--no-phonemes", action="store_true")
    args = parser.parse_args()

    cfg = DataConfig(data_root=args.data_root, cache_dir=args.cache_dir, use_phonemes=not args.no_phonemes)
    ensure_ljspeech(cfg)
    tokenizer = build_tokenizer(cfg, metadata_texts(cfg))
    records = preprocess_ljspeech(cfg, tokenizer, max_files=args.max_files, workers=args.workers)
    print(f"[prepare] {len(records)} utterances, vocab={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
