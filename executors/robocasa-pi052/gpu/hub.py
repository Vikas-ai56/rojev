"""Hugging Face Hub fetches behind this executor's GPU setup hook (`gpu/hook.sh`); run from the executor dir.

  uv run --frozen --no-sync python -m gpu.hub tokenizer [--offline]   Mac: gated PaliGemma tokenizer files
  uv run --frozen --no-sync python -m gpu.hub weights [--offline]     box: π0.5² checkpoint + FAST action tokenizer

`tokenizer` prints one local file path per line (the box gets copies of exactly these files). The gated download
uses the Mac's own Hugging Face login through the hub library's standard lookup; this script never reads, prints or
passes a token. `--offline` consults only the local cache and exits 1 if anything is missing: that is how the hook's
check_local / check_remote decide whether a step is already done.

The FAST action tokenizer is not named in drive.py: the π0.5² checkpoint's processor config loads it on first start,
so it is fetched here ahead of time to keep that first start fast.
"""
import argparse
import sys

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from drive import PALIGEMMA, POLICY

TOKENIZER_FILES = ("added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
                   "tokenizer_config.json")
FAST_ACTION_TOKENIZER = "lerobot/fast-action-tokenizer"
WEIGHT_REPOS = (POLICY, FAST_ACTION_TOKENIZER)


def tokenizer(offline: bool) -> list[str]:
    return [hf_hub_download(PALIGEMMA, name, local_files_only=offline) for name in TOKENIZER_FILES]


def weights(offline: bool) -> list[str]:
    return [snapshot_download(repo, local_files_only=offline) for repo in WEIGHT_REPOS]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["tokenizer", "weights"])
    p.add_argument("--offline", action="store_true", help="local cache only; exit 1 if anything is missing")
    a = p.parse_args()
    fetch = tokenizer if a.what == "tokenizer" else weights
    try:
        paths = fetch(a.offline)
    except LocalEntryNotFoundError as e:
        print(f"not in the local Hugging Face cache: {e}", file=sys.stderr)
        return 1
    print("\n".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
