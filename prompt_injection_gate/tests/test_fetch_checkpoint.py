"""fetch_checkpoint.install: only a hash-verified checkpoint is ever put in place (no network)."""
import hashlib
import os
import zipfile

import pytest

from prompt_injection_gate import fetch_checkpoint as fc


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "ARTIFACTS", str(tmp_path))
    weights = b"fake weights"
    m = {"dir": "ckpt-test", "model_sha256": hashlib.sha256(weights).hexdigest()}
    return tmp_path, m, weights


def test_verified_archive_is_installed(env):
    tmp, m, weights = env
    _zip(tmp / "a.zip", {"ckpt-test/model.safetensors": weights, "ckpt-test/gate_model.json": "{}"})
    fc.install(str(tmp / "a.zip"), m)
    assert fc.present(m)
    assert (tmp / "ckpt-test" / "model.safetensors").read_bytes() == weights


def test_hash_mismatch_is_refused_and_leaves_nothing(env):
    tmp, m, _ = env
    _zip(tmp / "a.zip", {"ckpt-test/model.safetensors": b"other weights", "ckpt-test/gate_model.json": "{}"})
    with pytest.raises(SystemExit, match="does not match"):
        fc.install(str(tmp / "a.zip"), m)
    assert not fc.present(m)
    assert sorted(os.listdir(tmp)) == ["a.zip"]          # temp folder cleaned up


def test_path_escaping_the_folder_is_refused(env):
    tmp, m, weights = env
    _zip(tmp / "a.zip", {"ckpt-test/model.safetensors": weights, "../evil.txt": "x"})
    with pytest.raises(SystemExit, match="unsafe path"):
        fc.install(str(tmp / "a.zip"), m)
    assert not (tmp.parent / "evil.txt").exists()


def test_existing_checkpoint_is_not_downloaded_again(env, monkeypatch):
    tmp, m, weights = env
    _zip(tmp / "a.zip", {"ckpt-test/model.safetensors": weights, "ckpt-test/gate_model.json": "{}"})
    fc.install(str(tmp / "a.zip"), m)
    monkeypatch.setattr(fc, "load_manifest", lambda: m)
    monkeypatch.setattr(fc, "download", lambda *a: pytest.fail("must not download"))
    assert fc.ensure() is True
