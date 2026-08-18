import torch
import pytest

from dsflow.audio import save_wav


@pytest.fixture
def fake_ljspeech(tmp_path):
    """Create a tiny LJSpeech-like tree (wavs + metadata.csv) under tmp_path."""
    wavs = tmp_path / "LJSpeech-1.1" / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(4):
        fid = f"LJ00{i + 1}_001"
        seconds = 1.0 + 0.5 * i
        t = torch.linspace(0, 2 * 3.14159 * 220 * seconds, int(22050 * seconds))
        wave = torch.randn(int(22050 * seconds)) * 0.1 + 0.2 * torch.sin(t)
        save_wav(wavs / f"{fid}.wav", wave, 22050)
        rows.append(f"{fid}|fake|sentence number {i + 1}.")
    (tmp_path / "LJSpeech-1.1" / "metadata.csv").write_text("\n".join(rows) + "\n")
    return tmp_path
