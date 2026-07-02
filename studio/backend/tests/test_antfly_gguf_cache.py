import asyncio
from types import SimpleNamespace

from routes import models as models_route


def test_cached_gguf_includes_antfly_generator_and_direct_layouts(
    tmp_path,
    monkeypatch,
):
    generator_repo = tmp_path / "generators" / "unsloth" / "gemma-4-12b-it-GGUF"
    generator_repo.mkdir(parents = True)
    (generator_repo / "gemma-4-12b-it-UD-Q4_K_XL.gguf").write_bytes(b"a" * 7)

    direct_repo = tmp_path / "ggml-org" / "gemma-4-E4B-it-GGUF"
    direct_repo.mkdir(parents = True)
    (direct_repo / "gemma-4-E4B-it-Q4_K_M.gguf").write_bytes(b"b" * 11)

    ignored_repo = tmp_path / "antflydb" / "clipclap"
    ignored_repo.mkdir(parents = True)
    (ignored_repo / "clipclap-clap.Q4_K.gguf").write_bytes(b"c" * 13)

    monkeypatch.setenv("ANTFLY_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test"))

    by_repo = {entry["repo_id"]: entry for entry in result["cached"]}
    assert by_repo["unsloth/gemma-4-12b-it-GGUF"]["size_bytes"] == 7
    assert by_repo["ggml-org/gemma-4-E4B-it-GGUF"]["size_bytes"] == 11
    assert "antflydb/clipclap" not in by_repo


def test_cached_gguf_includes_legacy_termite_default_cache(
    tmp_path,
    monkeypatch,
):
    legacy_repo = (
        tmp_path
        / ".termite"
        / "models"
        / "generators"
        / "unsloth"
        / "gemma-4-26B-A4B-it-GGUF"
    )
    legacy_repo.mkdir(parents = True)
    (legacy_repo / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf").write_bytes(b"d" * 17)

    monkeypatch.delenv("ANTFLY_MODELS_DIR", raising = False)
    monkeypatch.delenv("TERMITE_MODELS_DIR", raising = False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test"))

    by_repo = {entry["repo_id"]: entry for entry in result["cached"]}
    assert by_repo["unsloth/gemma-4-26B-A4B-it-GGUF"]["size_bytes"] == 17
    assert by_repo["unsloth/gemma-4-26B-A4B-it-GGUF"]["cache_path"] == str(
        legacy_repo
    )


def test_gguf_variants_marks_antfly_cached_variant_downloaded(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "generators" / "Unsloth" / "Gemma-4-12B-it-GGUF"
    repo.mkdir(parents = True)
    (repo / "gemma-4-12b-it-UD-Q4_K_XL.gguf").write_bytes(b"a" * 100)

    monkeypatch.setenv("ANTFLY_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        models_route,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (
            [
                SimpleNamespace(
                    filename = "gemma-4-12b-it-UD-Q4_K_XL.gguf",
                    quant = "UD-Q4_K_XL",
                    size_bytes = 100,
                ),
                SimpleNamespace(
                    filename = "gemma-4-12b-it-Q8_0.gguf",
                    quant = "Q8_0",
                    size_bytes = 200,
                ),
            ],
            False,
        ),
    )

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "unsloth/gemma-4-12b-it-GGUF",
            current_subject = "test",
        )
    )

    by_quant = {variant.quant: variant for variant in result.variants}
    assert by_quant["UD-Q4_K_XL"].downloaded is True
    assert by_quant["Q8_0"].downloaded is False
