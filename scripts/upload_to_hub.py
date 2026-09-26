#!/usr/bin/env python3
"""Upload the folders built by prepare_release.py to the Hugging Face Hub.

Run `hf auth login` first. Repositories are created if they do not exist.

Example:
    python scripts/upload_to_hub.py --hub-user ProtonPratt            # public
    python scripts/upload_to_hub.py --hub-user ProtonPratt --private
"""

import argparse
import os

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASES = ["Cognita-Mini-167M", "Cognita-Mini-167M-instruct-lora"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub-user", required=True)
    parser.add_argument("--release-dir", default=os.path.join(REPO_ROOT, "release"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    for name in RELEASES:
        folder = os.path.join(args.release_dir, name)
        if not os.path.isdir(folder):
            raise SystemExit(f"{folder} not found; run scripts/prepare_release.py first")
        repo_id = f"{args.hub_user}/{name}"
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model")
        print(f"Uploaded {folder} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
