from dsflow import __version__
from dsflow.config import DataConfig, InferConfig, ModelConfig, TrainConfig


def test_package_importable():
    assert __version__ == "0.1.0"


def test_configs_instantiate():
    data = DataConfig()
    model = ModelConfig()
    train = TrainConfig()
    infer = InferConfig()
    assert data.mel.sample_rate == 22050
    assert model.decoder_dim == 512
    assert train.steps == 2000
    assert infer.steps == 1
