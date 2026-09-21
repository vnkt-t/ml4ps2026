"""Anonymous derivatives preserve scientific evidence and checksum references."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.package_submission import REPO, copy_binary_anonymous, remap_hashes, sha


def test_signature_anonymization_preserves_source_and_numeric_bits(tmp_path):
    source, target = tmp_path/"original.npz", tmp_path/"anonymous.npz"
    values = np.array([-0., 1e-100, -3.75, np.pi], dtype=np.float64)
    np.savez_compressed(source, a=values, signature=json.dumps({"out":str(REPO/"results/example"), "seed":4}))
    before = sha(source)
    assert copy_binary_anonymous(source, target) == ["signature"]
    assert sha(source) == before
    with np.load(target, allow_pickle=False) as result:
        assert np.array_equal(result["a"].view(np.uint64), values.view(np.uint64))
        assert json.loads(str(result["signature"].item())) == {"out":"results/example", "seed":4}


def test_arbitrary_identifying_array_labels_are_not_silently_rewritten(tmp_path):
    source = tmp_path/"labels.npz"
    np.savez_compressed(source, labels=np.array([str(Path.home()/"private_label")]))
    before = sha(source)
    with pytest.raises(ValueError, match="non-metadata"):
        copy_binary_anonymous(source, tmp_path/"output.npz")
    assert sha(source) == before


def test_anonymous_checksum_updates_are_transitive(tmp_path):
    a, b, c = [tmp_path/name for name in ["a.json", "b.json", "c.json"]]
    a.write_text(json.dumps({"path":"absolute/local/path", "value":7}))
    b.write_text(json.dumps({"input_sha256":sha(a)}))
    c.write_text(json.dumps({"summary_sha256":sha(b)}))
    originals = {p.name:sha(p) for p in [a,b,c]}
    a.write_text(json.dumps({"path":"relative/path", "value":7}))
    remap_hashes(tmp_path, originals)
    assert json.loads(b.read_text())["input_sha256"] == sha(a)
    assert json.loads(c.read_text())["summary_sha256"] == sha(b)
    assert json.loads(a.read_text())["value"] == 7
