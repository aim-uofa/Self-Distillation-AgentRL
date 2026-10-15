#!/usr/bin/env python3
"""Materialize a shared, deterministic Search-QA evaluation set.

Only stdout is the selected Parquet path, so launchers can use command
substitution. Diagnostics go to stderr. Original datasets are never overwritten.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SOURCES = ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle")
PROTOCOL_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def atomic_target(path):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_target(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_subset(data_root, per_source=500, subset_seed=0, eval_size="all", eval_seed=0):
    """Return (selected_path, manifest); preserve the original Arrow schema/IDs."""
    if per_source < 0:
        raise ValueError("SEARCH_EVAL_PER_SOURCE must be non-negative (0 means full test set)")
    if not 0 <= subset_seed < 2**32 or not 0 <= eval_seed < 2**32:
        raise ValueError("Search evaluation seeds must be in [0, 2**32)")
    eval_size = str(eval_size)
    if eval_size != "all" and (not eval_size.isdigit() or int(eval_size) < 1):
        raise ValueError("EVAL_SIZE must be 'all' or a positive integer")
    source = Path(data_root).resolve() / "searchR1_processed_direct/test.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}; run bash launchers/prepare_search.sh qa")

    # Generation/task seeds must not change the full stratified evaluation set.
    parameters = {
        "protocol_version": PROTOCOL_VERSION,
        "source_sha256": file_sha256(source),
        "per_source": int(per_source),
        "subset_seed": int(subset_seed) if per_source else 0,
        "eval_size": eval_size,
        "eval_seed": int(eval_seed) if eval_size != "all" else None,
    }
    cache = Path(data_root).resolve() / "searchR1_fixed_eval"
    cache.mkdir(parents=True, exist_ok=True)
    name = f"test_per_source{per_source}_seed{parameters['subset_seed']}_{json_sha256(parameters)[:16]}"
    cached_data = cache / f"{name}.parquet"
    manifest_path = cache / f"{name}.manifest.json"
    output = source if per_source == 0 and eval_size == "all" else cached_data

    with locked(cache / f"{name}.lock"):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(manifest.get(key) != value for key, value in parameters.items()):
                raise ValueError(f"Search evaluation cache parameters mismatch: {manifest_path}")
            if not output.is_file() or file_sha256(output) != manifest["parquet_sha256"]:
                raise ValueError(f"Search evaluation cache is missing or corrupted: {output}")
            if pq.read_metadata(output).num_rows != manifest["row_count"]:
                raise ValueError(f"Search evaluation cache row count mismatch: {output}")
            return output, manifest

        table = pq.read_table(source)
        sources = np.asarray(table["data_source"].to_pylist(), dtype=object)
        source_counts = Counter(sources.tolist())
        if set(source_counts) != set(SOURCES):
            raise ValueError(f"Expected exactly the seven Search-QA sources {SOURCES}, got {dict(source_counts)}")
        if per_source:
            indices = []
            for name in SOURCES:
                pool = np.flatnonzero(sources == name)
                if len(pool) > per_source:
                    pool = np.random.RandomState(subset_seed).permutation(pool)[:per_source]
                indices.extend(pool.tolist())
            # Stable original-row order, including all Bamboogle questions.
            indices = np.asarray(sorted(indices), dtype=np.int64)
        else:
            indices = np.arange(len(table), dtype=np.int64)

        if eval_size != "all":
            size = int(eval_size)
            if size > len(indices):
                raise ValueError(
                    f"EVAL_SIZE={size} exceeds selected Search-QA pool size={len(indices)}; "
                    "set SEARCH_EVAL_PER_SOURCE=0 to use the original full pool"
                )
            indices = indices[np.random.RandomState(eval_seed).choice(len(indices), size, replace=False)]
        selected = table.take(pa.array(indices))
        if file_sha256(source) != parameters["source_sha256"]:
            raise RuntimeError("Search-QA source changed while preparing evaluation data; retry")
        if output != source:
            with atomic_target(output) as temporary:
                pq.write_table(selected, temporary)
        manifest = {
            **parameters,
            "source_row_count": len(table),
            "source_counts": dict(source_counts),
            "row_count": len(selected),
            "counts": dict(Counter(selected["data_source"].to_pylist())),
            "source_row_indices": indices.tolist(),
            "selection_sha256": json_sha256([parameters["source_sha256"], indices.tolist()]),
            "parquet_sha256": file_sha256(output),
        }
        write_json(manifest_path, manifest)
        return output, manifest


def prepare_train_stub(data_root):
    """Preserve the existing one-row, evaluation-only training placeholder."""
    source = Path(data_root) / "searchR1_processed_direct/train.parquet"
    target = Path(data_root) / "searchR1_pretrained_eval/train_1.parquet"
    with locked(target.with_suffix(".lock")):
        if not target.exists() or target.stat().st_size == 0:
            first = next(pq.ParquetFile(source).iter_batches(batch_size=1))
            with atomic_target(target) as temporary:
                pq.write_table(pa.Table.from_batches([first]), temporary)
    return target


def bind_run_directory(run_dir, manifest):
    """Never resume old/unidentified or differently sampled Search attempts."""
    run_dir = Path(run_dir)
    identity = {key: manifest[key] for key in (
        "protocol_version", "source_sha256", "per_source", "subset_seed",
        "eval_size", "eval_seed", "row_count", "counts", "selection_sha256",
    )}
    path = run_dir / "search_eval_manifest.json"
    with locked(run_dir / ".search_eval.lock"):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != identity:
                raise ValueError(f"Search evaluation dataset mismatch in {run_dir}; use a new RUN_ID/output directory")
        elif any(run_dir.glob("attempt_*")):
            raise ValueError(
                f"Existing Search attempts have no fixed-dataset manifest in {run_dir}; "
                "use a new RUN_ID/output directory (old results are not modified)"
            )
        else:
            write_json(path, identity)


def check_attempt(attempt_dir):
    """Check completed episode count even when EVAL_SIZE=all."""
    attempt_dir = Path(attempt_dir)
    identity = json.loads((attempt_dir.parent / "search_eval_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((attempt_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("benchmark") != "search" or not manifest.get("completed")
            or manifest.get("record_count") != identity["row_count"]):
        raise ValueError(
            f"Search attempt is incomplete or has the wrong task count: {attempt_dir}; "
            f"expected {identity['row_count']}, got {manifest.get('record_count')}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data"))))
    parser.add_argument("--per-source", type=int, default=os.environ.get("SEARCH_EVAL_PER_SOURCE", "500"))
    parser.add_argument("--subset-seed", type=int, default=os.environ.get("SEARCH_EVAL_SUBSET_SEED", "0"))
    parser.add_argument("--eval-size", default="all")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--prepare-train-subset", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-attempt", type=Path)
    args = parser.parse_args()
    try:
        if args.check_attempt:
            check_attempt(args.check_attempt)
            return
        path, manifest = prepare_subset(
            args.data_root, args.per_source, args.subset_seed, args.eval_size, args.eval_seed,
        )
        if args.run_dir:
            bind_run_directory(args.run_dir, manifest)
        if args.prepare_train_subset:
            prepare_train_stub(args.data_root)
        print(
            f"[Search-QA eval data] rows={manifest['row_count']}, counts={manifest['counts']}, "
            f"subset_seed={manifest['subset_seed']}, fingerprint={manifest['selection_sha256']}\n"
            f"[Search-QA eval data] file={path}", file=sys.stderr,
        )
        print(path)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"Search-QA evaluation setup failed: {error}\n")


if __name__ == "__main__":
    main()
