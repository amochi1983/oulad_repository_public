# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Acquisition must preserve the shipped provenance after success or corruption."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def prepare_fixture(tmp_path):
    path = Path(__file__).resolve().parents[1] / "src/download.py"
    spec = importlib.util.spec_from_file_location("pinned_download_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    archive = raw / "anonymisedData.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in module.FILES:
            z.writestr(name + ".csv", "id,value\n1,7\n")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            (raw / name).write_bytes(z.read(name))
    metadata = {
        "files": [{"name": archive.name, "size": archive.stat().st_size,
                   "computed_md5": hashlib.md5(archive.read_bytes()).hexdigest(),
                   "download_url": "https://example.invalid/not-used"}],
        "url": "https://example.invalid/metadata-not-used",
        "license": {"name": "CC BY 4.0"}, "citation": "Test fixture only"
    }
    (raw / "figshare_metadata.json").write_text(json.dumps(metadata))
    rows = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "download_date": "fixture", "size": p.stat().st_size}
            for p in [archive] + [raw / (name + ".csv") for name in module.FILES]]
    manifest = raw / "manifest.json"
    manifest.write_text(json.dumps(rows))
    return module, raw, manifest, manifest.read_bytes()


def test_valid_acquisition_preserves_expected_manifest(tmp_path):
    module, raw, manifest, before = prepare_fixture(tmp_path)
    module.main()
    assert manifest.read_bytes() == before
    assert (raw / "last_acquisition.json").is_file()


def test_corrupt_csv_is_not_rebaselined(tmp_path):
    module, raw, manifest, before = prepare_fixture(tmp_path)
    (raw / "studentInfo.csv").write_text("id,value\n1,999\n")
    with pytest.raises(ValueError, match="Pinned raw SHA256 mismatch"):
        module.main()
    assert manifest.read_bytes() == before
    assert not (raw / "last_acquisition.json").exists()
