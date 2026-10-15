"""Download a Hugging Face model, or copy an explicitly supplied local model."""

import argparse
import json
from pathlib import Path
import subprocess


def validate_model(directory):
    """Check the HF layout/shard inventory, not the integrity of tensor contents."""
    directory = Path(directory).resolve()

    def require_file(name):
        path = (directory / name).resolve()
        # Local HF snapshots may legitimately symlink files into their cache.
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty model file: {directory / name}")
        return path

    json.loads(require_file("config.json").read_text())
    json.loads(require_file("tokenizer_config.json").read_text())
    if not any((directory / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json",
    )):
        raise ValueError(f"Tokenizer files not found under {directory}")
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (directory / index_name).is_file():
            index = json.loads(require_file(index_name).read_text())
            shards = set(index.get("weight_map", {}).values())
            if not shards:
                raise ValueError(f"Empty weight_map in {directory / index_name}")
            for shard in shards:
                if Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ValueError(f"Invalid shard path: {shard}")
                require_file(shard)
            return
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (directory / name).is_file():
            require_file(name)
            return
    raise ValueError(f"HF weights or a weight index not found under {directory}")


def copy_model(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    validate_model(source)
    if source == target:
        return
    if source in target.parents or target in source.parents:
        raise ValueError("Source and destination model directories must not contain each other.")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"Refusing to overwrite a non-empty destination: {target}. "
                         "Use MODEL_PATH pointing to the existing model, or choose an empty destination.")
    target.mkdir(parents=True, exist_ok=True)
    # Dereference HF cache symlinks so the copy works without the source cache.
    subprocess.run(["cp", "-rL", "--preserve=mode,timestamps", "--reflink=auto", "--",
                    str(source) + "/.", str(target)], check=True)
    validate_model(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    try:
        if args.source is not None:
            copy_model(args.source, args.output_dir)
        else:
            from huggingface_hub import snapshot_download

            # Re-running this command reuses completed files and download metadata.
            snapshot_download(repo_id=args.model_id, local_dir=str(args.output_dir))
            validate_model(args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Model preparation failed: {exc}\n")
    print(f"Hugging Face model is ready under {args.output_dir}")


if __name__ == "__main__":
    main()
