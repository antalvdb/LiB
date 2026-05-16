"""Train a LiB tokenizer on a streaming sample of FineWeb-Edu and upload to the Hub.

Usage:
    python train_tokenizer.py [--vocab-size N] [--samples N] [--hub-repo REPO]
    python train_tokenizer.py --upload-only ./saved-tokenizer-dir
"""

import argparse
import tempfile
import os
from datasets import load_dataset
from lib_tokenizers import LiBTokenizerFast

HUB_REPO = "antalvdb/lib-tokenizer"
VOCAB_SIZE = 50_000
NUM_SAMPLES = 100_000   # sentences streamed from FineWeb-Edu
NUM_EPOCHS = 10_000


def upload(save_dir, hub_repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(hub_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=hub_repo,
        repo_type="model",
        commit_message="Retrain with prepend-only space convention",
    )
    print(f"Uploaded to https://huggingface.co/{hub_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--hub-repo", type=str, default=HUB_REPO)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save trained tokenizer here instead of a temp dir")
    parser.add_argument("--upload-only", type=str, default=None, metavar="DIR",
                        help="Skip training; upload an already-saved tokenizer dir")
    args = parser.parse_args()

    if args.upload_only:
        print(f"Uploading {args.upload_only} to {args.hub_repo} …")
        upload(args.upload_only, args.hub_repo)
        return

    print(f"Streaming {args.samples:,} sentences from HuggingFaceFW/fineweb-edu …")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        corpus_path = f.name
        count = 0
        for example in dataset:
            for line in example["text"].splitlines():
                line = line.strip()
                if line:
                    f.write(line + "\n")
                    count += 1
                    if count >= args.samples:
                        break
            if count >= args.samples:
                break
    print(f"Wrote {count:,} lines to {corpus_path}")

    print(f"Training LiB tokenizer (vocab_size={args.vocab_size}, epochs={args.epochs}) …")
    kwargs = dict(
        vocab_size=args.vocab_size,
        num_epochs=args.epochs,
    )
    if args.seed is not None:
        kwargs["seed"] = args.seed

    tokenizer = LiBTokenizerFast.train_new(corpus_path, **kwargs)
    os.unlink(corpus_path)

    print(f"Trained vocab size: {tokenizer.vocab_size}")

    save_dir = args.save_dir or tempfile.mkdtemp(prefix="lib-tokenizer-")
    tokenizer.save_pretrained(save_dir)
    print(f"Saved tokenizer to {save_dir}")

    print(f"Uploading to {args.hub_repo} …")
    upload(save_dir, args.hub_repo)

    if not args.save_dir:
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
