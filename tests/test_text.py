from dsflow.text import BOS, EOS, TextTokenizer, _char_units, phonemize


def test_char_fallback_units():
    assert _char_units("Hello, World! 123") == ["h", "e", "l", "l", "o", "w", "o", "r", "l", "d", "1", "2", "3"]


def test_tokenizer_roundtrip():
    tok = TextTokenizer.from_corpus(["hello world", "test sentence"], use_phonemes=False)
    ids = tok.encode("hello world", phones=[])
    assert ids == [BOS, EOS]
    assert tok.decode(ids[:2]) == ["<bos>", "<eos>"]
    assert tok.vocab_size == 4 + 11  # 4 specials + 11 distinct letters in the corpus


def test_phonemize_available_or_clean_fallback():
    phones = phonemize("hello world")
    if phones is not None:
        assert all(len(p) > 0 for p in phones)
        assert all(p.isalpha() for p in phones)


def test_tokenizer_save_load(tmp_path):
    tok = TextTokenizer.from_corpus(["hi there"], use_phonemes=False)
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = TextTokenizer.load(path)
    assert loaded.symbols == tok.symbols
