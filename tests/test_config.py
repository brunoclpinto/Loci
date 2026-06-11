"""Tests for loci/config.py: loading, precedence, validation."""
import pytest
from pathlib import Path

from loci.config import Config, load


class TestDefaults:
    def test_returns_config_instance(self):
        assert isinstance(load(), Config)

    def test_default_values(self):
        cfg = load()
        assert cfg.models.n_ctx == 4096
        assert cfg.retrieval.vec_top_k == 12
        assert cfg.ingest.chunk_tokens == 512
        assert cfg.logging.jsonl is True
        assert cfg.bench.runs_per_question == 3

    def test_no_provenance_when_no_file(self, monkeypatch):
        # Simulate an environment where no config file is found
        monkeypatch.setattr("loci.config._find_config_file", lambda _: None)
        cfg = load()
        assert cfg._provenance == {}


class TestFileLoading:
    def test_override_from_file(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nvec_top_k = 20\n")
        cfg = load(cfg_file)
        assert cfg.retrieval.vec_top_k == 20

    def test_provenance_points_to_file(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nvec_top_k = 20\n")
        cfg = load(cfg_file)
        assert str(cfg_file) in cfg._provenance["retrieval.vec_top_k"]

    def test_partial_section_leaves_defaults(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nvec_top_k = 5\n")
        cfg = load(cfg_file)
        assert cfg.retrieval.fts_top_k == 12  # untouched

    def test_unknown_section_raises_valueerror(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[badSection]\nfoo = 1\n")
        with pytest.raises(ValueError, match="Unknown config section"):
            load(cfg_file)

    def test_unknown_key_raises_valueerror(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nnonexistent_key = 99\n")
        with pytest.raises(ValueError, match="Unknown config key"):
            load(cfg_file)

    def test_missing_path_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load(tmp_path / "does_not_exist.toml")

    def test_invalid_toml_raises_valueerror(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("this is not [valid toml !!!")
        with pytest.raises(ValueError, match="Invalid TOML"):
            load(cfg_file)

    def test_path_values_become_path_objects(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[paths]\nmodels_dir = "/tmp/loci_models"\n')
        cfg = load(cfg_file)
        assert isinstance(cfg.paths.models_dir, Path)
        assert cfg.paths.models_dir == Path("/tmp/loci_models")


class TestEnvOverrides:
    def test_env_overrides_int(self, monkeypatch):
        monkeypatch.setenv("LOCI_RETRIEVAL__VEC_TOP_K", "99")
        cfg = load()
        assert cfg.retrieval.vec_top_k == 99

    def test_env_overrides_bool_false(self, monkeypatch):
        monkeypatch.setenv("LOCI_LOGGING__JSONL", "false")
        cfg = load()
        assert cfg.logging.jsonl is False

    def test_env_overrides_bool_true(self, monkeypatch):
        monkeypatch.setenv("LOCI_MODELS__USE_MMAP", "true")
        cfg = load()
        assert cfg.models.use_mmap is True

    def test_env_unknown_section_ignored(self, monkeypatch):
        monkeypatch.setenv("LOCI_FAKESECTION__FOO", "bar")
        cfg = load()  # should not raise

    def test_env_provenance(self, monkeypatch):
        monkeypatch.setenv("LOCI_RETRIEVAL__FTS_TOP_K", "7")
        cfg = load()
        assert "env:LOCI_RETRIEVAL__FTS_TOP_K" in cfg._provenance["retrieval.fts_top_k"]


class TestPrecedence:
    def test_env_beats_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nvec_top_k = 20\n")
        monkeypatch.setenv("LOCI_RETRIEVAL__VEC_TOP_K", "30")
        cfg = load(cfg_file)
        assert cfg.retrieval.vec_top_k == 30
        assert cfg._provenance["retrieval.vec_top_k"].startswith("env:")

    def test_override_dict_beats_env(self, monkeypatch):
        monkeypatch.setenv("LOCI_RETRIEVAL__VEC_TOP_K", "30")
        cfg = load(overrides={"retrieval.vec_top_k": 50})
        assert cfg.retrieval.vec_top_k == 50
        assert cfg._provenance["retrieval.vec_top_k"] == "cli-flag"

    def test_override_dict_beats_file(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nvec_top_k = 20\n")
        cfg = load(cfg_file, overrides={"retrieval.vec_top_k": 100})
        assert cfg.retrieval.vec_top_k == 100
        assert cfg._provenance["retrieval.vec_top_k"] == "cli-flag"


class TestCLICommands:
    def test_config_init_creates_file(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from loci.cli import app
        import loci.cli as cli_mod

        dest = tmp_path / "loci_config.toml"
        monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", dest)

        runner = CliRunner()
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        assert dest.exists()
        assert "[retrieval]" in dest.read_text()

    def test_config_show_no_error(self, tmp_path):
        from typer.testing import CliRunner
        from loci.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "retrieval.vec_top_k" in result.output

    def test_config_show_with_file(self, tmp_path):
        from typer.testing import CliRunner
        from loci.cli import app

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[retrieval]\nvec_top_k = 77\n")

        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "--config", str(cfg_file)])
        assert result.exit_code == 0
        assert "77" in result.output

    def test_config_show_bad_file_exits_1(self, tmp_path):
        from typer.testing import CliRunner
        from loci.cli import app

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[unknownSection]\nfoo = 1\n")

        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "--config", str(cfg_file)])
        assert result.exit_code == 1
