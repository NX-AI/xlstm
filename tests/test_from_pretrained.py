import importlib


def test_load_from_pretrained_reads_yaml_config(tmp_path, monkeypatch):
    from_pretrained = importlib.import_module("xlstm.xlstm_large.from_pretrained")

    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.yaml").write_text(
        "embedding_dim: 64\nnum_heads: 4\nnum_blocks: 2\nvocab_size: 128\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(from_pretrained, "load_file", lambda _: {"weight": "value"})

    class FakeModel:
        def __init__(self, config):
            self.config = config
            self.state_dict = None

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

    monkeypatch.setattr(from_pretrained, "xLSTMLarge", FakeModel)

    model = from_pretrained.load_from_pretrained(tmp_path)

    assert model.config.embedding_dim == 64
    assert model.config.num_heads == 4
    assert model.config.num_blocks == 2
    assert model.config.vocab_size == 128
    assert model.config.mode == "inference"
    assert model.state_dict == {"weight": "value"}
