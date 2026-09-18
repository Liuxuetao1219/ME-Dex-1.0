"""Publish verified local exports as one atomic Hub commit; keep Git history."""
import argparse
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--export", type=Path, required=True)
p.add_argument("--backup", type=Path, required=True)
p.add_argument("--token-stdin", action="store_true", help="Read an existing HF token from stdin; never save it")
a = p.parse_args()
backup = json.loads((a.backup / "BACKUP_VERIFICATION.json").read_text())
for name, size in backup["weights"].items():
    assert (a.backup / name).stat().st_size == size
manifest = json.loads((a.export / "manifest.json").read_text())
assert manifest["variant"] == "ME-Dex-1.0" and manifest["step"] == 50000
api = HfApi(endpoint="https://huggingface.co", token=sys.stdin.read().strip() if a.token_stdin else None)
head = api.model_info(a.repo).sha
operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(a.export / name))
              for name in ("model.pt", "tactile_ae.pt", "manifest.json", "model_config.json")]
operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(
    Path(__file__).resolve().parents[1] / "docs/HF_MODEL_CARD.md")))
commit = api.create_commit(repo_id=a.repo, repo_type="model", operations=operations,
                          parent_commit=head, commit_message="Release ME-Dex-1.0 new V3 AE Clean-only Full Attention 50k")
print("PUBLISHED", commit.oid, commit.commit_url, flush=True)
