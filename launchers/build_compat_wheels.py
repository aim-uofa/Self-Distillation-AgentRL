#!/usr/bin/env python3
"""Build metadata-corrected wheels without changing package versions or code."""

import copy
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "compat_wheels"

# Exact upstream artifacts whose metadata defects have been inspected.
ARTIFACTS = [
    (
        "https://files.pythonhosted.org/packages/"
        "11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/"
        "decord-0.6.0-py3-none-manylinux2010_x86_64.whl",
        "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976",
    ),
    (
        "https://files.pythonhosted.org/packages/"
        "f7/5c/0799519ab54e76f0e13d8d46a4f2257e6854e1650e70725a8133638095f3/"
        "textworld-1.7.0-py3-none-manylinux_2_5_x86_64.manylinux1_x86_64."
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "ca7486cff540c4d90865c54a3465e948c2790357aa78ad64de4406387854cbc8",
    ),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["SOURCE_DATE_EPOCH"] = "315532800"
    requirements = []

    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix=".build-") as temp:
        work = Path(temp)

        for url, expected_hash in ARTIFACTS:
            filename = Path(urlparse(url).path).name
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "download",
                    "--no-deps", "--no-cache-dir", "--no-index", "--disable-pip-version-check",
                    "--dest", str(work),
                    f"{url}#sha256={expected_hash}",
                ],
                check=True,
            )

            original = work / filename
            if sha256(original) != expected_hash:
                raise RuntimeError(f"Unexpected upstream artifact: {filename}")

            distribution, version, python_tag, abi_tag, platform_tag = filename[:-4].split("-")
            fixed_name = (
                f"{distribution}-{version}-1agentrl-"
                f"{python_tag}-{abi_tag}-{platform_tag}.whl"
            )
            fixed = work / fixed_name
            metadata_dir = f"{distribution}-{version}.dist-info"
            wheel_path = f"{metadata_dir}/WHEEL"
            record_path = f"{metadata_dir}/RECORD"
            _, _, _, tags = parse_wheel_filename(filename)
            replaced = {wheel_path, record_path, record_path + ".jws", record_path + ".p7s"}

            with ZipFile(original) as source:
                old_metadata = source.read(wheel_path).decode("utf-8")
                # Keep other headers; remove TextWorld's premature blank line.
                headers = [
                    line for line in old_metadata.splitlines()
                    if line and not line.startswith(("Tag:", "Build:"))
                ]
                headers.append("Build: 1agentrl")
                headers.extend(f"Tag: {tag}" for tag in sorted(tags, key=str))
                new_metadata = ("\n".join(headers) + "\n\n").encode("utf-8")

                # WheelFile regenerates RECORD hashes for the new archive.
                with WheelFile(fixed, "w") as target:
                    for member in source.infolist():
                        if member.is_dir() or member.filename in replaced - {wheel_path}:
                            continue
                        content = source.read(member)
                        if member.filename == wheel_path:
                            content = new_metadata
                        # zipfile mutates ZipInfo offsets while writing; retain
                        # the original entry for the byte-for-byte check below.
                        target.writestr(copy.copy(member), content)

                # Validate RECORD and ensure all code, binaries, package metadata
                # and licenses are byte-identical to the pinned upstream wheel.
                with WheelFile(fixed, "r") as verification:
                    for member in verification.infolist():
                        if not member.is_dir():
                            content = verification.read(member)
                            if member.filename not in replaced and content != source.read(member.filename):
                                raise RuntimeError(f"Unexpected payload change: {member.filename}")
                    original_files = {m.filename for m in source.infolist() if not m.is_dir()}
                    if original_files - replaced != set(verification.namelist()) - replaced:
                        raise RuntimeError(f"Unexpected payload inventory change: {filename}")

            destination = OUTPUT / fixed_name
            fixed.replace(destination)
            requirements.append(f"./{fixed_name} --hash=sha256:{sha256(destination)}")

    manifest = OUTPUT / "requirements.txt"
    manifest.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    print(f"Compatibility wheels and hash manifest written to {OUTPUT}")


if __name__ == "__main__":
    main()
