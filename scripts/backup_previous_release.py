"""Preserve the existing release locally before replacing the Hub main revision."""
import argparse
import json
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download


def main(a):
    import torch
    import numpy as np
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        for child in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            sys.modules.setdefault("numpy._core." + child, __import__("numpy.core." + child, fromlist=[child]))
    info = HfApi().model_info(a.repo)
    manifest = json.loads(Path(hf_hub_download(a.repo, "manifest.json", revision=info.sha)).read_text())
    expected = {"model.pt": manifest["exported_model"]["size_bytes"],
                "tactile_ae.pt": manifest["tactile_ae"]["size_bytes"]}
    for filename, size in expected.items():
        path = a.existing / filename
        if path.stat().st_size != size:
            raise ValueError(f"Backup candidate size disagrees with published manifest: {filename}")
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if filename == "model.pt":
            assert isinstance(state["module"], dict) and len(state["module"]) > 3000
            assert state["module"]["tactile_expert.tokenizer.slice_embedding"].shape[0] == 18
        else:
            assert state["schema"]["architecture"] == "universal_anatomy_tactile_ae_v3"
            assert state["ema"]["shadow"]
        del state
    a.destination.mkdir(parents=True, exist_ok=False)
    for filename in expected:
        # Same shared filesystem: preserve without another 12GB copy.
        os.link(a.existing / filename, a.destination / filename)
    for filename in ("README.md", "manifest.json", "model_config.json", ".gitattributes"):
        hf_hub_download(a.repo, filename, revision=info.sha, local_dir=a.destination)
    report = {"repo": a.repo, "revision": info.sha, "backup": str(a.destination),
              "checks": "published manifest sizes, readable torch archives, legacy 18-slice model and AE schema",
              "not_checked": "byte-for-byte equality to remote weight payload", "weights": expected}
    (a.destination / "BACKUP_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--existing", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    main(p.parse_args())
